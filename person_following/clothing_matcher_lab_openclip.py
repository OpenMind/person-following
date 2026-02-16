"""
Clothing Matcher with Lab + OpenCLIP.

Two-stage matching:
1. Fast clothing color matching using Lab histograms (L, a, b channels)
2. OpenCLIP embedding for semantic matching (better for cross-view matching)

Uses TensorRT YOLO11s-seg for person segmentation.
Uses OpenCLIP ViT-B-16 for visual-semantic feature extraction.
"""

import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="xFormers is not available")

# Configure logging
logger = logging.getLogger(__name__)

try:
    import pycuda.autoinit  # noqa: F401 - Required for CUDA initialization
    import pycuda.driver as cuda
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT not available")

try:
    import open_clip
    import torch
    from PIL import Image

    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False
    logger.warning("OpenCLIP not available. Install with: pip install open-clip-torch")


class SegmentationError(Exception):
    """
    Exception raised when segmentation fails.

    This exception is raised when the segmentation model cannot produce
    a valid person mask from the input image.

    Parameters
    ----------
    message : str
        Explanation of the segmentation failure.

    """

    pass


class TRTSegmentationModel:
    """
    TensorRT YOLO11-seg model for person segmentation.

    Provides GPU-accelerated person segmentation using a TensorRT-optimized
    YOLO11 segmentation model.

    Parameters
    ----------
    engine_path : str
        Path to the TensorRT engine file (.engine).

    Attributes
    ----------
    logger : trt.Logger
        TensorRT logger instance.
    engine : trt.ICudaEngine
        Deserialized TensorRT engine.
    context : trt.IExecutionContext
        TensorRT execution context.
    input_name : str
        Name of the input tensor.
    input_shape : tuple
        Shape of the input tensor (N, C, H, W).
    input_h : int
        Input height in pixels.
    input_w : int
        Input width in pixels.
    output_names : list of str
        Names of output tensors.
    output_shapes : list of tuple
        Shapes of output tensors.
    inputs : list of dict
        Input buffer information.
    outputs : list of dict
        Output buffer information.
    stream : cuda.Stream
        CUDA stream for async operations.
    """

    def __init__(self, engine_path: str):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")

        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        self.output_names = []
        self.output_shapes = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.engine.get_tensor_shape(name)
                self.output_names.append(name)
                self.output_shapes.append(shape)

        self._allocate_buffers()
        logger.info(f"Segmentation model loaded: {engine_path}")

    def _allocate_buffers(self):
        """
        Allocate CUDA buffers for input and output tensors.
        """
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            if -1 in shape:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    shape = self.input_shape
                else:
                    shape = tuple(max(1, abs(s)) for s in shape)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            entry = {
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "name": name,
                "dtype": dtype,
            }
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int, int, int]:
        """
        Preprocess image for segmentation inference.

        Resizes the image with letterboxing to match the model input size,
        converts BGR to RGB, and normalizes pixel values.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image of shape (height, width, 3).

        Returns
        -------
        blob : numpy.ndarray
            Preprocessed image blob of shape (1, 3, input_h, input_w).
        scale : float
            Scale factor applied during resize.
        pad_w : int
            Horizontal padding offset in pixels.
        pad_h : int
            Vertical padding offset in pixels.
        new_w : int
            Width after scaling (before padding).
        new_h : int
            Height after scaling (before padding).
        """
        h, w = image.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        letterbox = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_h = (self.input_h - new_h) // 2
        pad_w = (self.input_w - new_w) // 2
        letterbox[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Convert BGR to RGB for model
        letterbox_rgb = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)

        blob = letterbox_rgb.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis])

        return blob, scale, pad_w, pad_h, new_w, new_h

    def infer(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], float, Tuple[int, int], Tuple[int, int]]:
        """
        Run segmentation inference on an image.

        Preprocesses the image, runs TensorRT inference, and returns
        the raw outputs along with preprocessing parameters.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image of shape (height, width, 3).

        Returns
        -------
        outputs : list of numpy.ndarray
            List of output tensors from the model.
        scale : float
            Scale factor applied during preprocessing.
        pads : tuple of int
            Padding offsets (pad_w, pad_h).
        sizes : tuple of int
            Scaled sizes before padding (new_w, new_h).
        """
        blob, scale, pad_w, pad_h, new_w, new_h = self.preprocess(image)

        np.copyto(self.inputs[0]["host"], blob.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        self.context.execute_async_v3(self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        outputs = []
        for out in self.outputs:
            actual_shape = self.context.get_tensor_shape(out["name"])
            output = out["host"][: int(np.prod(actual_shape))].reshape(actual_shape)
            outputs.append(output)

        return outputs, scale, (pad_w, pad_h), (new_w, new_h)


class OpenCLIPMatcher:
    """
    OpenCLIP-based visual-semantic feature matcher.

    Provides visual feature extraction using OpenCLIP models for
    person re-identification. Better than DINOv2 for cross-view matching
    because it's trained with text-image pairs and learns semantic concepts.

    Parameters
    ----------
    model_name : str, optional
        OpenCLIP model architecture name, by default 'ViT-B-16'.
    pretrained : str, optional
        Pretrained weights identifier, by default 'laion2b_s34b_b88k'.
    device : str, optional
        Device to run inference on, by default 'cuda'.

    Attributes
    ----------
    device : str
        Device for inference.
    model_name : str
        Model architecture name.
    pretrained : str
        Pretrained weights identifier.
    model : open_clip.CLIP
        Loaded OpenCLIP model.
    preprocess : callable
        Image preprocessing transform.
    embed_dim : int
        Dimension of output embeddings.
    """

    # OpenCLIP mean/std (same as CLIP)
    CLIP_MEAN_BGR = np.array(
        [122, 116, 104], dtype=np.uint8
    )  # [0.48, 0.46, 0.41] * 255

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize OpenCLIP feature extractor.

        Parameters
        ----------
        model_name : str, optional
            CLIP model architecture, by default 'ViT-B-16'.
        pretrained : str, optional
            Pretrained weights name, by default 'laion2b_s34b_b88k'.
        device : str, optional
            Compute device, by default 'cuda'.
        cache_dir : str, optional
            Directory to cache downloaded models.
        """
        if not HAS_OPENCLIP:
            raise RuntimeError(
                "OpenCLIP not available. Install with: pip install open-clip-torch"
            )

        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        logger.info(f"Loading OpenCLIP {model_name} ({pretrained})...")

        kwargs = dict(pretrained=pretrained, device=device)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["cache_dir"] = cache_dir

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, **kwargs
        )
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            dummy_out = self.model.encode_image(dummy)
            self.embed_dim = dummy_out.shape[-1]

        logger.info(f"OpenCLIP {model_name} loaded (dim: {self.embed_dim})")

    def extract_embedding(
        self, image: np.ndarray, mask: np.ndarray, use_mask: bool = True
    ) -> np.ndarray:
        """
        Extract OpenCLIP embedding from image.

        Extracts a normalized feature embedding from the masked person region
        using the OpenCLIP vision encoder.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image of shape (height, width, 3).
        mask : numpy.ndarray
            Binary mask of shape (height, width) indicating the person region.
        use_mask : bool, optional
            If True, mask out background with neutral color, by default True.

        Returns
        -------
        numpy.ndarray
            L2-normalized embedding vector of shape (embed_dim,).
        """
        if mask is None or mask.sum() < 500:
            raise ValueError(
                f"Invalid mask: {mask.sum() if mask is not None else 0} pixels"
            )

        mask_binary = (mask > 0).astype(np.uint8)

        # Find bounding box of mask
        ys, xs = np.where(mask_binary > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()

        # Add padding
        pad = 10
        h, w = image.shape[:2]
        y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
        x1, x2 = max(0, x1 - pad), min(w, x2 + pad)

        if use_mask:
            # Use CLIP mean as background (neutral after normalization)
            masked = np.full_like(image, 0)
            masked[:, :] = self.CLIP_MEAN_BGR
            masked[mask_binary > 0] = image[mask_binary > 0]
            cropped = masked[y1:y2, x1:x2]
        else:
            cropped = image[y1:y2, x1:x2].copy()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Preprocess and get embedding
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(input_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.

        Parameters
        ----------
        emb1 : numpy.ndarray
            First embedding vector of shape (embed_dim,).
        emb2 : numpy.ndarray
            Second embedding vector of shape (embed_dim,).

        Returns
        -------
        float
            Cosine similarity score in range [-1, 1].
        """
        return float(np.dot(emb1, emb2))


class ClothingMatcher:
    """
    Two-stage clothing matcher using Lab color histograms and OpenCLIP.

    Provides robust person matching using a two-stage approach:
    Stage 1: Lab color histograms for fast, lighting-robust matching
    Stage 2: OpenCLIP embedding for semantic matching and cross-view robustness

    Parameters
    ----------
    yolo_seg_engine : str
        Path to the YOLO segmentation TensorRT engine file.
    device : str, optional
        Device to run inference on, by default 'cuda'.
    use_clip : bool, optional
        Whether to use OpenCLIP for verification, by default True.
    clip_model : str, optional
        OpenCLIP model architecture name, by default 'ViT-B-16'.
    clip_pretrained : str, optional
        OpenCLIP pretrained weights identifier, by default 'laion2b_s34b_b88k'.

    Attributes
    ----------
    device : str
        Device for inference.
    seg_model : TRTSegmentationModel
        TensorRT segmentation model.
    clip_matcher : OpenCLIPMatcher or None
        OpenCLIP matcher instance, or None if disabled.
    L_BINS : int
        Number of bins for L channel histogram (16).
    A_BINS : int
        Number of bins for a channel histogram (32).
    B_BINS : int
        Number of bins for b channel histogram (32).
    CLOTHING_THRESHOLD : float
        Default clothing similarity threshold (0.5).
    CLIP_THRESHOLD : float
        Default OpenCLIP similarity threshold (0.7).
    """

    L_BINS = 16
    A_BINS = 32
    B_BINS = 32

    CLOTHING_THRESHOLD = 0.5
    CLIP_THRESHOLD = 0.7  # OpenCLIP threshold

    def __init__(
        self,
        yolo_seg_engine: str,
        device: str = "cuda",
        use_clip: bool = True,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        clip_cache_dir: Optional[str] = None,
    ):
        self.device = device

        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT is required")
        self.seg_model = TRTSegmentationModel(yolo_seg_engine)

        # OpenCLIP matcher
        self.clip_matcher = None
        if use_clip and HAS_OPENCLIP:
            self.clip_matcher = OpenCLIPMatcher(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                cache_dir=clip_cache_dir,
            )

        logger.info("ClothingMatcher initialized")
        logger.info("  - Segmentation: TensorRT YOLO")
        logger.info(
            f"  - OpenCLIP: {'Enabled (' + clip_model + ')' if self.clip_matcher else 'Disabled'}"
        )

    def _identify_outputs(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify detection and proto outputs by shape.

        Analyzes the output tensor shapes to distinguish between
        detection outputs and prototype mask outputs.

        Parameters
        ----------
        outputs : list of numpy.ndarray
            List of output tensors from the segmentation model.

        Returns
        -------
        det_output : numpy.ndarray
            Detection output tensor.
        proto_output : numpy.ndarray
            Prototype mask output tensor.
        """
        det_output = None
        proto_output = None

        for output in outputs:
            shape = output.shape
            if len(shape) == 4:
                proto_output = output
            elif len(shape) == 3:
                det_output = output
            elif len(shape) == 2:
                det_output = output[np.newaxis, ...]

        if det_output is None or proto_output is None:
            raise SegmentationError(
                f"Could not identify outputs. Shapes: {[o.shape for o in outputs]}"
            )

        return det_output, proto_output

    def _fill_mask_contours(self, mask: np.ndarray, min_area: int = 500) -> np.ndarray:
        """
        Fill mask using contours to fix incomplete masks.

        Finds contours of the mask and fills them completely to fix
        issues with YOLO-seg mask probabilities at edges being filtered out.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask of shape (height, width), values 0/1 or 0/255.
        min_area : int, optional
            Minimum contour area to consider in pixels, by default 500.

        Returns
        -------
        numpy.ndarray
            Filled binary mask of shape (height, width) with no holes
            inside the person boundary.

        Notes
        -----
        Problem: YOLO-seg mask probabilities at edges (arms, etc.) may be
        0.3-0.5, which get filtered out by threshold 0.5, leaving holes.
        Solution: Find contours of the mask and fill them completely.
        """
        # Ensure binary mask
        mask_binary = (mask > 0).astype(np.uint8)

        # Find external contours
        contours, hierarchy = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return mask_binary

        # Find largest contour (main person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < min_area:
            return mask_binary

        # Create filled mask
        filled_mask = np.zeros_like(mask_binary)
        cv2.fillPoly(filled_mask, [largest_contour], 1)

        # Optional: Also fill any other significant contours (in case person has separate parts)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area * 0.3:  # 30% of min_area
                cv2.fillPoly(filled_mask, [contour], 1)

        return filled_mask

    def extract_person_mask_from_crop(
        self,
        crop: np.ndarray,
        conf_thresh: float = 0.3,
        validate_centroid: bool = True,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Extract person segmentation mask from a cropped image.

        Runs segmentation on the crop and returns the best person mask
        based on confidence and centroid validation.

        Parameters
        ----------
        crop : numpy.ndarray
            Cropped BGR image of shape (height, width, 3).
        conf_thresh : float, optional
            Minimum confidence threshold for detections, by default 0.3.
        validate_centroid : bool, optional
            If True, validate that mask centroid is near image center,
            by default True.
        debug : bool, optional
            If True, enable debug output, by default False.

        Returns
        -------
        numpy.ndarray
            Binary mask of shape (height, width) with dtype uint8.
        """
        h, w = crop.shape[:2]
        outputs, scale, (pad_w, pad_h), (new_w, new_h) = self.seg_model.infer(crop)

        if len(outputs) < 2:
            raise SegmentationError(f"Expected 2 outputs, got {len(outputs)}")

        det_output, proto_output = self._identify_outputs(outputs)

        if det_output.ndim == 3:
            det_output = det_output[0]
        if det_output.shape[0] == 116 and det_output.shape[1] != 116:
            det_output = det_output.T

        proto = proto_output[0]
        proto_h, proto_w = proto.shape[1], proto.shape[2]

        best_mask = None
        best_conf = 0
        best_centroid_dist = float("inf")
        expected_cx, expected_cy = w / 2, h / 2

        for det in det_output:
            class_scores = det[4:84]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            if class_id != 0 or conf < conf_thresh:
                continue

            mask_coeffs = det[84:116]
            mask_proto = np.tensordot(mask_coeffs, proto, axes=([0], [0]))
            mask_proto = 1.0 / (1.0 + np.exp(-mask_proto))

            scale_h = proto_h / self.seg_model.input_h
            scale_w = proto_w / self.seg_model.input_w

            p_pad_h = int(pad_h * scale_h)
            p_pad_w = int(pad_w * scale_w)
            p_new_h = int(new_h * scale_h)
            p_new_w = int(new_w * scale_w)

            p_pad_h = max(0, min(p_pad_h, proto_h - 1))
            p_pad_w = max(0, min(p_pad_w, proto_w - 1))
            p_end_h = min(p_pad_h + p_new_h, proto_h)
            p_end_w = min(p_pad_w + p_new_w, proto_w)

            if p_end_h <= p_pad_h or p_end_w <= p_pad_w:
                continue

            mask_valid = mask_proto[p_pad_h:p_end_h, p_pad_w:p_end_w]
            if mask_valid.size == 0:
                continue

            mask_resized = cv2.resize(
                mask_valid, (w, h), interpolation=cv2.INTER_LINEAR
            )
            candidate_mask = (mask_resized > 0.5).astype(np.uint8)

            # Fill contours to fix incomplete masks (arms, edges, etc.)
            candidate_mask = self._fill_mask_contours(candidate_mask)

            ys, xs = np.where(candidate_mask > 0)
            if len(ys) == 0:
                continue

            mask_cx = xs.mean()
            mask_cy = ys.mean()
            centroid_dist = np.sqrt(
                (mask_cx - expected_cx) ** 2 + (mask_cy - expected_cy) ** 2
            )

            if validate_centroid:
                max_acceptable_dist = min(w, h) * 0.4
                if centroid_dist < max_acceptable_dist:
                    if conf > best_conf or (
                        conf > best_conf * 0.9 and centroid_dist < best_centroid_dist
                    ):
                        best_conf = conf
                        best_mask = candidate_mask
                        best_centroid_dist = centroid_dist
            else:
                if conf > best_conf:
                    best_conf = conf
                    best_mask = candidate_mask

        if best_mask is None:
            raise SegmentationError(f"No valid person detected (conf >= {conf_thresh})")

        coverage = best_mask.sum() / best_mask.size * 100
        if coverage < 5:
            raise SegmentationError(f"Mask coverage too low: {coverage:.1f}%")
        if coverage > 95:
            raise SegmentationError(f"Mask coverage too high: {coverage:.1f}%")

        return best_mask

    def _compute_lab_histograms(
        self, lab_pixels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute normalized Lab histograms.

        Parameters
        ----------
        lab_pixels : numpy.ndarray
            Lab pixel values of shape (N, 3).

        Returns
        -------
        l_hist : numpy.ndarray
            Normalized L channel histogram of shape (L_BINS,).
        a_hist : numpy.ndarray
            Normalized a channel histogram of shape (A_BINS,).
        b_hist : numpy.ndarray
            Normalized b channel histogram of shape (B_BINS,).
        """
        l_pixels = lab_pixels[:, 0]
        a_pixels = lab_pixels[:, 1]
        b_pixels = lab_pixels[:, 2]

        l_hist, _ = np.histogram(l_pixels, bins=self.L_BINS, range=(0, 256))
        a_hist, _ = np.histogram(a_pixels, bins=self.A_BINS, range=(0, 256))
        b_hist, _ = np.histogram(b_pixels, bins=self.B_BINS, range=(0, 256))

        l_hist = l_hist.astype(np.float32) / (l_hist.sum() + 1e-6)
        a_hist = a_hist.astype(np.float32) / (a_hist.sum() + 1e-6)
        b_hist = b_hist.astype(np.float32) / (b_hist.sum() + 1e-6)

        return l_hist, a_hist, b_hist

    def extract_clothing_features(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Extract Lab color features from masked clothing region.

        Computes Lab color histograms and other features from the
        clothing region defined by the mask.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image of shape (height, width, 3).
        mask : numpy.ndarray
            Binary mask of shape (height, width).

        Returns
        -------
        dict
            Feature dictionary containing:
            - 'l_hist', 'a_hist', 'b_hist' : numpy.ndarray
                Full body Lab histograms.
            - 'upper_l', 'upper_a', 'upper_b' : numpy.ndarray
                Upper body Lab histograms.
            - 'lower_l', 'lower_a', 'lower_b' : numpy.ndarray
                Lower body Lab histograms.
            - 'avg_lab' : numpy.ndarray
                Average Lab values (3,).
            - 'avg_color' : numpy.ndarray
                Average BGR values (3,).
            - 'avg_l', 'avg_a', 'avg_b' : float
                Individual average Lab channel values.
            - 'edge_density' : float
                Edge density in masked region.
            - 'color_variance' : float
                Color variance in masked region.
            - 'pixel_count' : int
                Total pixels in mask.
            - 'upper_pixels', 'lower_pixels' : int
                Pixels in upper/lower body regions.
        """
        if mask is None:
            raise ValueError("Mask cannot be None")

        h, w = image.shape[:2]
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs ({h}, {w})")

        mask_bool = mask.astype(bool)
        pixel_count = mask_bool.sum()
        if pixel_count < 500:
            raise ValueError(f"Too few pixels: {pixel_count}")

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        masked_lab = lab[mask_bool]

        l_hist, a_hist, b_hist = self._compute_lab_histograms(masked_lab)

        # Upper/lower body split
        ys, xs = np.where(mask_bool)
        y_mid = (ys.min() + ys.max()) // 2

        upper_mask = mask_bool.copy()
        upper_mask[y_mid:, :] = False
        lower_mask = mask_bool.copy()
        lower_mask[:y_mid, :] = False

        upper_pixels = upper_mask.sum()
        lower_pixels = lower_mask.sum()

        if upper_pixels >= 200:
            upper_lab = lab[upper_mask]
            upper_l, upper_a, upper_b = self._compute_lab_histograms(upper_lab)
        else:
            upper_l, upper_a, upper_b = l_hist.copy(), a_hist.copy(), b_hist.copy()

        if lower_pixels >= 200:
            lower_lab = lab[lower_mask]
            lower_l, lower_a, lower_b = self._compute_lab_histograms(lower_lab)
        else:
            lower_l, lower_a, lower_b = l_hist.copy(), a_hist.copy(), b_hist.copy()

        avg_lab = masked_lab.mean(axis=0)
        avg_bgr = np.array([image[:, :, c][mask_bool].mean() for c in range(3)])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges[mask_bool].sum() / (pixel_count * 255 + 1e-6)

        color_variance = masked_lab.std(axis=0).mean()

        return {
            "l_hist": l_hist,
            "a_hist": a_hist,
            "b_hist": b_hist,
            "upper_l": upper_l,
            "upper_a": upper_a,
            "upper_b": upper_b,
            "lower_l": lower_l,
            "lower_a": lower_a,
            "lower_b": lower_b,
            "avg_lab": avg_lab,
            "avg_color": avg_bgr,
            "avg_l": float(avg_lab[0]),
            "avg_a": float(avg_lab[1]),
            "avg_b": float(avg_lab[2]),
            "edge_density": edge_density,
            "color_variance": color_variance,
            "pixel_count": pixel_count,
            "upper_pixels": upper_pixels,
            "lower_pixels": lower_pixels,
        }

    def extract_clip_embedding(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract OpenCLIP embedding from image.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image of shape (height, width, 3).
        mask : numpy.ndarray
            Binary mask of shape (height, width).

        Returns
        -------
        numpy.ndarray
            L2-normalized embedding vector.
        """
        if self.clip_matcher is None:
            raise RuntimeError("OpenCLIP not initialized")
        return self.clip_matcher.extract_embedding(image, mask, use_mask=True)

    def _compare_histograms(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """
        Compare histograms using Bhattacharyya distance and intersection.

        Parameters
        ----------
        h1 : numpy.ndarray
            First histogram.
        h2 : numpy.ndarray
            Second histogram.

        Returns
        -------
        float
            Combined similarity score in range [0, 1].
        """
        h1 = np.ascontiguousarray(h1.astype(np.float32))
        h2 = np.ascontiguousarray(h2.astype(np.float32))

        bhatt_dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        bhatt_sim = 1.0 - bhatt_dist

        intersect = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)

        return 0.6 * bhatt_sim + 0.4 * intersect

    def compute_clothing_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """
        Compute Lab histogram similarity between two feature sets.

        Computes a weighted similarity score based on full-body histograms,
        regional histograms (upper/lower body), and other features.

        Parameters
        ----------
        feat1 : dict
            First feature dictionary from extract_clothing_features.
        feat2 : dict
            Second feature dictionary from extract_clothing_features.

        Returns
        -------
        float
            Similarity score in range [0, 1].
        """
        l_sim = self._compare_histograms(feat1["l_hist"], feat2["l_hist"])
        a_sim = self._compare_histograms(feat1["a_hist"], feat2["a_hist"])
        b_sim = self._compare_histograms(feat1["b_hist"], feat2["b_hist"])

        upper_l_sim = self._compare_histograms(feat1["upper_l"], feat2["upper_l"])
        upper_a_sim = self._compare_histograms(feat1["upper_a"], feat2["upper_a"])
        upper_b_sim = self._compare_histograms(feat1["upper_b"], feat2["upper_b"])

        lower_l_sim = self._compare_histograms(feat1["lower_l"], feat2["lower_l"])
        lower_a_sim = self._compare_histograms(feat1["lower_a"], feat2["lower_a"])
        lower_b_sim = self._compare_histograms(feat1["lower_b"], feat2["lower_b"])

        lab_diff = np.abs(feat1["avg_lab"] - feat2["avg_lab"])
        weighted_lab_diff = 0.2 * lab_diff[0] + 0.4 * lab_diff[1] + 0.4 * lab_diff[2]
        avg_lab_sim = 1.0 - min(weighted_lab_diff / 50.0, 1.0)

        edge_diff = abs(feat1["edge_density"] - feat2["edge_density"])
        edge_sim = 1.0 - min(edge_diff * 5, 1.0)

        var_diff = abs(feat1["color_variance"] - feat2["color_variance"])
        var_sim = 1.0 - min(var_diff / 20.0, 1.0)

        upper_sim = (upper_l_sim + upper_a_sim + upper_b_sim) / 3.0
        lower_sim = (lower_l_sim + lower_a_sim + lower_b_sim) / 3.0
        regional_sim = 0.5 * upper_sim + 0.5 * lower_sim

        other_sim = 0.5 * avg_lab_sim + 0.3 * edge_sim + 0.2 * var_sim

        similarity = (
            0.40 * (0.15 * l_sim + 0.425 * a_sim + 0.425 * b_sim)
            + 0.35 * regional_sim
            + 0.25 * other_sim
        )

        return float(np.clip(similarity, 0.0, 1.0))

    def compute_clip_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute OpenCLIP embedding similarity.

        Parameters
        ----------
        emb1 : numpy.ndarray
            First embedding vector.
        emb2 : numpy.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity score.
        """
        if self.clip_matcher is None:
            raise RuntimeError("OpenCLIP not initialized")
        return self.clip_matcher.compute_similarity(emb1, emb2)
