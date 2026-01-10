"""
TensorRT YOLO11n Person Detector.

Only detects class 0 (person) from COCO.
Returns detections compatible with BoxMOT tracker format.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

# Configure logging
logger = logging.getLogger(__name__)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_the: float) -> List[int]:
    """
    Perform greedy Non-Maximum Suppression on bounding boxes.

    Parameters
    ----------
    boxes : numpy.ndarray
        Bounding boxes of shape (N, 4) with format [x1, y1, x2, y2].
    scores : numpy.ndarray
        Confidence scores of shape (N,).
    iou_the : float
        IoU threshold for suppression. Boxes with IoU greater than
        this threshold with a higher-scoring box are suppressed.

    Returns
    -------
    list of int
        Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_the)[0]
        order = order[inds + 1]
    return keep


class TRTYOLODetector:
    """
    TensorRT YOLO11n detector for person detection only.

    Provides GPU-accelerated person detection using a TensorRT-optimized
    YOLO11n model. Only detects COCO class 0 (person).

    Parameters
    ----------
    engine_path : str
        Path to the TensorRT engine file (.engine).
    size : int, optional
        Input image size (square), by default 640.
    conf_thresh : float, optional
        Confidence threshold for detections, by default 0.5.
    nms_thresh : float, optional
        IoU threshold for NMS, by default 0.45.

    Attributes
    ----------
    size : int
        Input image size.
    conf_thresh : float
        Confidence threshold.
    nms_thresh : float
        NMS IoU threshold.
    engine : trt.ICudaEngine
        Deserialized TensorRT engine.
    context : trt.IExecutionContext
        TensorRT execution context.
    stream : cuda.Stream
        CUDA stream for async operations.
    v10_api : bool
        Whether using TensorRT 10+ API.
    PERSON_CLASS : int
        COCO class index for person (0).
    """

    PERSON_CLASS = 0  # COCO class index for person

    def __init__(
        self,
        engine_path: str,
        size: int = 640,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.45,
    ):
        self.size = size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Check TRT version API
        self.v10_api = hasattr(self.engine, "num_io_tensors")

        if self.v10_api:
            self.io_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            self.in_name = [
                n
                for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ][0]
            self.out_name = [
                n
                for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ][0]
        else:
            self.bindings_map = {
                self.engine.get_binding_name(i): i
                for i in range(self.engine.num_bindings)
            }
            self.in_idx = next(
                i
                for n, i in self.bindings_map.items()
                if self.engine.binding_is_input(i)
            )
            self.out_idx = next(
                i
                for n, i in self.bindings_map.items()
                if not self.engine.binding_is_input(i)
            )

        logger.info(f"YOLO detector loaded: {engine_path}")

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Letterbox preprocessing for YOLO input.

        Resizes the image with letterboxing to maintain aspect ratio,
        converts BGR to RGB, and normalizes pixel values.

        Parameters
        ----------
        img_bgr : numpy.ndarray
            Input BGR image of shape (height, width, 3).

        Returns
        -------
        blob : numpy.ndarray
            Preprocessed image blob of shape (1, 3, size, size).
        scale : float
            Scale factor applied during resize.
        pad_w : int
            Horizontal padding offset in pixels.
        pad_h : int
            Vertical padding offset in pixels.
        """
        H, W = img_bgr.shape[:2]
        scale = min(self.size / H, self.size / W)
        new_w, new_h = int(W * scale), int(H * scale)
        pad_w = (self.size - new_w) // 2
        pad_h = (self.size - new_h) // 2

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR -> RGB, normalize, NHWC -> NCHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw), scale, pad_w, pad_h

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Decode YOLO output and filter to person class only.

        Processes raw YOLO output, extracts person detections,
        applies confidence filtering, coordinate transformation,
        and Non-Maximum Suppression.

        Parameters
        ----------
        output : numpy.ndarray
            Raw YOLO output of shape [1, 84, num_anchors] or [1, num_anchors, 84].
            84 = 4 (bbox) + 80 (classes).
        scale : float
            Scale factor from preprocessing.
        pad_w : int
            Horizontal padding from preprocessing.
        pad_h : int
            Vertical padding from preprocessing.
        orig_h : int
            Original image height.
        orig_w : int
            Original image width.

        Returns
        -------
        numpy.ndarray
            Detection array of shape (N, 6) with format
            [x1, y1, x2, y2, confidence, class_id].
            Class ID is always 0 (person).
        """
        # YOLO11 output: [1, 84, num_anchors] or [1, num_anchors, 84]
        # 84 = 4 (bbox) + 80 (classes)
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] == 84:
            output = output.T  # [84, N] -> [N, 84]

        # Extract bbox and class scores
        cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        class_scores = output[:, 4:]  # [N, 80]

        # Get person class score (class 0)
        person_scores = class_scores[:, self.PERSON_CLASS]

        # Filter by confidence
        mask = person_scores >= self.conf_thresh
        if not np.any(mask):
            return np.zeros((0, 6), np.float32)  # [x1, y1, x2, y2, conf, class]

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = person_scores[mask]

        # Convert xywh -> xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove letterbox and scale to original
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Class is always 0 (person)
        classes = np.zeros_like(scores)

        dets = np.stack([x1, y1, x2, y2, scores, classes], axis=1).astype(np.float32)

        # NMS
        keep = nms_boxes(dets[:, :4], dets[:, 4], self.nms_thresh)
        if not keep:
            return np.zeros((0, 6), np.float32)

        return dets[keep]

    def detect(self, img_bgr: np.ndarray, max_num: int = 0) -> np.ndarray:
        """
        Run detection on a single frame.

        Performs person detection on the input image using the TensorRT
        YOLO model with preprocessing, inference, and postprocessing.

        Parameters
        ----------
        img_bgr : numpy.ndarray
            Input BGR image of shape (height, width, 3).
        max_num : int, optional
            Maximum number of detections to return. If 0, returns all
            detections. If positive, keeps the largest detections by area.
            By default 0.

        Returns
        -------
        numpy.ndarray
            Detection array of shape (N, 6) with format
            [x1, y1, x2, y2, confidence, class_id].
            Format is compatible with BoxMOT tracker.
        """
        H, W = img_bgr.shape[:2]
        inp, scale, pad_w, pad_h = self._preprocess(img_bgr)

        # Allocate device memory
        d_in = cuda.mem_alloc(inp.nbytes)
        cuda.memcpy_htod_async(d_in, inp, self.stream)

        if self.v10_api:
            self.context.set_input_shape(self.in_name, tuple(inp.shape))
            self.context.set_tensor_address(self.in_name, int(d_in))

            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            out_size = int(np.prod(out_shape)) * 4
            d_out = cuda.mem_alloc(out_size)
            self.context.set_tensor_address(self.out_name, int(d_out))

            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.set_binding_shape(self.in_idx, tuple(inp.shape))
            out_shape = tuple(self.context.get_binding_shape(self.out_idx))
            out_size = int(np.prod(out_shape)) * 4
            d_out = cuda.mem_alloc(out_size)

            bindings = [None] * self.engine.num_bindings
            bindings[self.in_idx] = int(d_in)
            bindings[self.out_idx] = int(d_out)

            self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream.handle
            )

        out_host = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(out_host, d_out, self.stream)
        self.stream.synchronize()

        dets = self._postprocess(out_host, scale, pad_w, pad_h, H, W)

        if max_num > 0 and len(dets) > max_num:
            # Keep largest by area
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            idx = np.argsort(areas)[::-1][:max_num]
            dets = dets[idx]

        return dets
