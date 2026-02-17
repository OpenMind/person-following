"""
Model Manager for Person Following System.

Handles automatic downloading of ONNX models and compilation to TensorRT engines.
Ensures compatibility across different machines and TensorRT versions.
"""

import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import tensorrt as trt

logger = logging.getLogger(__name__)


class TRTVersionMismatchError(Exception):
    """Raised when TensorRT engine version doesn't match runtime."""

    pass


class ModelManager:
    """
    Manages ONNX model downloads and TensorRT engine compilation.

    Features:
    - Auto-downloads ONNX models from configured URLs
    - Checks TensorRT version compatibility
    - Auto-recompiles engines when version mismatch detected
    - Supports multiple models with different configurations
    """

    # Default model configurations
    DEFAULT_MODELS = {
        "yolo11n": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
            "onnx_name": "yolo11n.onnx",
            "engine_name": "yolo11n.engine",
            "input_shape": "1x3x640x640",
            "fp16": True,
            "workspace": 4096,  # MB
            "description": "YOLO11n detection model",
        },
        "yolo11s-seg": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.onnx",
            "onnx_name": "yolo11s-seg.onnx",
            "engine_name": "yolo11s-seg.engine",
            "input_shape": "1x3x640x640",
            "fp16": True,
            "workspace": 4096,  # MB
            "description": "YOLO11s segmentation model",
        },
    }

    def __init__(
        self,
        model_dir: str = "/opt/person_following/model",
        engine_dir: str = "/opt/person_following/engine",
        trtexec_path: str = "/usr/src/tensorrt/bin/trtexec",
        force_recompile: bool = False,
    ):
        """
        Initialize model manager.

        Parameters
        ----------
        model_dir : str
            Directory to store ONNX models.
        engine_dir : str
            Directory to store TensorRT engines.
        trtexec_path : str
            Path to trtexec binary.
        force_recompile : bool
            Force recompilation even if engine exists.
        """
        self.model_dir = Path(model_dir)
        self.engine_dir = Path(engine_dir)
        self.trtexec_path = Path(trtexec_path)
        self.force_recompile = force_recompile

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.engine_dir.mkdir(parents=True, exist_ok=True)

        # Get TensorRT version
        self.trt_version = self._get_trt_version()
        logger.info(f"TensorRT version: {self.trt_version}")

        # Check trtexec availability
        if not self.trtexec_path.exists():
            # Try to find in PATH
            trtexec_in_path = shutil.which("trtexec")
            if trtexec_in_path:
                self.trtexec_path = Path(trtexec_in_path)
                logger.info(f"Found trtexec in PATH: {self.trtexec_path}")
            else:
                logger.warning(f"trtexec not found at {self.trtexec_path} or in PATH")

    def _get_trt_version(self) -> str:
        """Get TensorRT version string."""
        return trt.__version__

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _check_engine_compatibility(self, engine_path: Path) -> bool:
        """
        Check if TensorRT engine is compatible with current runtime.

        Parameters
        ----------
        engine_path : Path
            Path to engine file.

        Returns
        -------
        bool
            True if compatible, False otherwise.
        """
        if not engine_path.exists():
            return False

        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    logger.warning(f"Failed to deserialize engine: {engine_path}")
                    return False
                logger.info(f"Engine {engine_path.name} is compatible")
                return True
        except Exception as e:
            logger.warning(f"Engine compatibility check failed: {e}")
            return False

    def _download_file(self, url: str, dest_path: Path, desc: str = "model") -> bool:
        """
        Download file from URL with progress.

        Parameters
        ----------
        url : str
            Download URL.
        dest_path : Path
            Destination file path.
        desc : str
            Description for logging.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Downloading {desc} from {url}")
            logger.info(f"Destination: {dest_path}")

            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100.0 / total_size, 100)
                    if block_num % 50 == 0 or percent >= 100:
                        logger.info(
                            f"Download progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)"
                        )

            urlretrieve(url, dest_path, reporthook=_progress)
            logger.info(f"Download complete: {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def _compile_engine(
        self,
        onnx_path: Path,
        engine_path: Path,
        fp16: bool = True,
        workspace: int = 4096,
        input_shape: Optional[str] = None,
        input_name: str = "images",
    ) -> bool:
        """
        Compile ONNX model to TensorRT engine using trtexec.

        Parameters
        ----------
        onnx_path : Path
            Path to ONNX model.
        engine_path : Path
            Output engine path.
        fp16 : bool
            Enable FP16 precision.
        workspace : int
            Max workspace size in MB.
        input_shape : str, optional
            Input shape specification (e.g., "1x3x640x640").
        input_name : str
            Name of input tensor (default: "images").

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self.trtexec_path.exists():
            logger.error(f"trtexec not found at {self.trtexec_path}")
            return False

        if not onnx_path.exists():
            logger.error(f"ONNX model not found: {onnx_path}")
            return False

        logger.info("=" * 60)
        logger.info("Compiling TensorRT engine:")
        logger.info(f"  ONNX:      {onnx_path}")
        logger.info(f"  Engine:    {engine_path}")
        logger.info(f"  FP16:      {fp16}")
        logger.info(f"  Workspace: {workspace} MB")
        if input_shape:
            logger.info(f"  Input Shape: {input_name}:{input_shape}")
        logger.info(f"  TRT Version: {self.trt_version}")
        logger.info("=" * 60)

        # Build trtexec command
        cmd = [
            str(self.trtexec_path),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]

        trt_major = int(self.trt_version.split(".")[0])
        if trt_major >= 10:
            cmd.append(f"--memPoolSize=workspace:{workspace}M")
        else:
            cmd.append(f"--workspace={workspace}")

        if fp16:
            cmd.append("--fp16")

        if input_shape:
            shape_spec = f"{input_name}:{input_shape}"
            cmd.extend(
                [
                    f"--minShapes={shape_spec}",
                    f"--optShapes={shape_spec}",
                    f"--maxShapes={shape_spec}",
                ]
            )

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Log output
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.debug(f"trtexec: {line}")

            if engine_path.exists():
                logger.info(f"✓ Engine compiled successfully: {engine_path}")
                logger.info(
                    f"  Size: {engine_path.stat().st_size / 1024 / 1024:.1f} MB"
                )
                return True
            else:
                logger.error("Engine compilation failed: output file not found")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"trtexec failed with return code {e.returncode}")
            logger.error(f"Output:\n{e.stdout}")
            return False
        except Exception as e:
            logger.error(f"Engine compilation error: {e}")
            return False

    def prepare_model(
        self, model_key: str, custom_config: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Path]]:
        """
        Prepare model: download ONNX if needed, compile to engine.

        Parameters
        ----------
        model_key : str
            Model key from DEFAULT_MODELS or custom key.
        custom_config : dict, optional
            Custom model configuration (overrides DEFAULT_MODELS).

        Returns
        -------
        tuple of (bool, Path or None)
            (success, engine_path) - engine path if successful, None otherwise.
        """
        # Get model config
        if custom_config:
            config = custom_config
        elif model_key in self.DEFAULT_MODELS:
            config = self.DEFAULT_MODELS[model_key]
        else:
            logger.error(f"Unknown model key: {model_key}")
            return False, None

        onnx_path = self.model_dir / config["onnx_name"]
        engine_path = self.engine_dir / config["engine_name"]

        logger.info(f"Preparing model: {config.get('description', model_key)}")

        # Step 1: Download ONNX if missing
        if not onnx_path.exists():
            logger.info("ONNX model not found, downloading...")
            if not self._download_file(config["url"], onnx_path, config["onnx_name"]):
                return False, None
        else:
            logger.info(f"ONNX model found: {onnx_path}")

        # Step 2: Check engine compatibility
        needs_compile = False

        if self.force_recompile:
            logger.info("Force recompile enabled")
            needs_compile = True
        elif not engine_path.exists():
            logger.info("Engine not found")
            needs_compile = True
        elif not self._check_engine_compatibility(engine_path):
            logger.warning("Engine version mismatch detected")
            needs_compile = True
        else:
            logger.info(f"Using existing compatible engine: {engine_path}")
            return True, engine_path

        # Step 3: Compile engine
        if needs_compile:
            logger.info("Compiling TensorRT engine (this may take a few minutes)...")

            # Backup old engine if exists
            if engine_path.exists():
                backup_path = engine_path.with_suffix(
                    f".engine.backup.{self.trt_version}"
                )
                logger.info(f"Backing up old engine to: {backup_path}")
                shutil.move(engine_path, backup_path)

            success = self._compile_engine(
                onnx_path,
                engine_path,
                fp16=config.get("fp16", True),
                workspace=config.get("workspace", 4096),
                input_shape=config.get("input_shape"),
                input_name=config.get("input_name", "images"),
            )

            if not success:
                logger.error("Failed to compile engine")
                return False, None

        return True, engine_path

    def prepare_all_models(self) -> Dict[str, Optional[Path]]:
        """
        Prepare all default models.

        Returns
        -------
        dict
            Dictionary mapping model keys to engine paths (None if failed).
        """
        results = {}
        for model_key in self.DEFAULT_MODELS:
            success, engine_path = self.prepare_model(model_key)
            results[model_key] = engine_path if success else None
        return results


def main():
    """Test model manager."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Model Manager for Person Following")
    parser.add_argument(
        "--model-dir",
        default="/opt/person_following/model",
        help="Directory for ONNX models",
    )
    parser.add_argument(
        "--engine-dir",
        default="/opt/person_following/engine",
        help="Directory for TensorRT engines",
    )
    parser.add_argument(
        "--force-recompile", action="store_true", help="Force recompilation of engines"
    )
    parser.add_argument(
        "--model", default="all", help="Model to prepare (all, yolo11n, yolo11s-seg)"
    )

    args = parser.parse_args()

    manager = ModelManager(
        model_dir=args.model_dir,
        engine_dir=args.engine_dir,
        force_recompile=args.force_recompile,
    )

    if args.model == "all":
        logger.info("Preparing all models...")
        results = manager.prepare_all_models()

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY:")
        for model_key, engine_path in results.items():
            status = "✓ SUCCESS" if engine_path else "✗ FAILED"
            logger.info(f"  {model_key:20s} {status}")
            if engine_path:
                logger.info(f"    → {engine_path}")
        logger.info("=" * 60)

        # Exit with error if any failed
        if any(path is None for path in results.values()):
            sys.exit(1)
    else:
        success, engine_path = manager.prepare_model(args.model)
        if success:
            logger.info(f"\n✓ Model ready: {engine_path}")
            sys.exit(0)
        else:
            logger.error(f"\n✗ Failed to prepare model: {args.model}")
            sys.exit(1)


if __name__ == "__main__":
    main()
