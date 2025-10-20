"""
Platform detection and training configuration for the Knowledge Engine.

This module provides intelligent platform detection and configuration
to ensure optimal training settings based on available hardware and OS.

Key Features:
- Automatic OS and GPU detection
- QLoRA capability validation
- Platform-specific training configuration
- Clear feedback on capabilities and limitations
"""

import platform
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

from shared.logging import get_logger

logger = get_logger("knowledge-service")


@dataclass
class PlatformCapabilities:
    """
    Platform capabilities and hardware information.

    Attributes:
        os_type: Operating system (linux, darwin, windows)
        has_cuda: NVIDIA CUDA GPU available
        has_mps: Apple Metal Performance Shaders available
        has_bitsandbytes: bitsandbytes library available
        device_type: Best device for training (cuda, mps, cpu)
        supports_fp16: FP16 training supported
        supports_qlora: QLoRA training supported
    """
    os_type: str
    has_cuda: bool
    has_mps: bool
    has_bitsandbytes: bool
    device_type: str
    supports_fp16: bool
    supports_qlora: bool


@dataclass
class TrainingConfiguration:
    """
    Platform-optimized training configuration.

    Attributes:
        use_fp16: Enable FP16 mixed precision training
        use_qlora: Enable QLoRA (4-bit quantization)
        optimizer: Optimizer type to use
        device_map: Device mapping strategy
        torch_dtype: PyTorch tensor dtype
        warnings: List of capability warnings
        recommendations: List of recommendations
    """
    use_fp16: bool
    use_qlora: bool
    optimizer: str
    device_map: str
    torch_dtype: str
    warnings: List[str]
    recommendations: List[str]


class PlatformDetector:
    """
    Detects platform capabilities and provides optimized training configuration.

    This class analyzes the current system to determine:
    - Operating system and architecture
    - Available GPU hardware (CUDA, MPS)
    - Installed ML libraries (bitsandbytes)
    - Optimal training settings

    Usage:
        detector = PlatformDetector()
        config = detector.get_training_config(use_qlora=True)

        if config.warnings:
            for warning in config.warnings:
                logger.warning(warning)
    """

    def __init__(self):
        """Initialize platform detector and detect capabilities."""
        self.capabilities = self._detect_capabilities()
        self._log_capabilities()

    def _detect_capabilities(self) -> PlatformCapabilities:
        """
        Detect all platform capabilities.

        Returns:
            PlatformCapabilities object with detected information
        """
        os_type = platform.system().lower()  # linux, darwin, windows

        # Detect CUDA
        has_cuda = self._detect_cuda()

        # Detect MPS (Apple Silicon)
        has_mps = self._detect_mps()

        # Detect bitsandbytes availability
        has_bitsandbytes = self._detect_bitsandbytes()

        # Determine best device
        if has_cuda:
            device_type = "cuda"
        elif has_mps:
            device_type = "mps"
        else:
            device_type = "cpu"

        # Determine FP16 support (only on CUDA)
        supports_fp16 = has_cuda

        # Determine QLoRA support (requires CUDA + bitsandbytes)
        supports_qlora = has_cuda and has_bitsandbytes

        return PlatformCapabilities(
            os_type=os_type,
            has_cuda=has_cuda,
            has_mps=has_mps,
            has_bitsandbytes=has_bitsandbytes,
            device_type=device_type,
            supports_fp16=supports_fp16,
            supports_qlora=supports_qlora
        )

    def _detect_cuda(self) -> bool:
        """
        Detect if NVIDIA CUDA is available.

        Returns:
            True if CUDA GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _detect_mps(self) -> bool:
        """
        Detect if Apple MPS (Metal Performance Shaders) is available.

        Returns:
            True if MPS is available, False otherwise
        """
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def _detect_bitsandbytes(self) -> bool:
        """
        Detect if bitsandbytes library is available.

        Returns:
            True if bitsandbytes can be imported, False otherwise
        """
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    def _log_capabilities(self):
        """Log detected platform capabilities."""
        caps = self.capabilities

        logger.info("=" * 60)
        logger.info("Platform Capabilities Detected")
        logger.info("=" * 60)
        logger.info(f"Operating System: {caps.os_type}")
        logger.info(f"GPU - CUDA: {'✓' if caps.has_cuda else '✗'}")
        logger.info(f"GPU - MPS (Apple Silicon): {'✓' if caps.has_mps else '✗'}")
        logger.info(f"Library - bitsandbytes: {'✓' if caps.has_bitsandbytes else '✗'}")
        logger.info(f"Primary Device: {caps.device_type.upper()}")
        logger.info(f"FP16 Training: {'Supported' if caps.supports_fp16 else 'Not Supported'}")
        logger.info(f"QLoRA Training: {'Supported' if caps.supports_qlora else 'Not Supported'}")
        logger.info("=" * 60)

    def get_training_config(
        self,
        requested_qlora: bool = False,
        requested_fp16: Optional[bool] = None
    ) -> TrainingConfiguration:
        """
        Get optimized training configuration for this platform.

        Args:
            requested_qlora: Whether QLoRA was requested
            requested_fp16: Whether FP16 was explicitly requested (None = auto)

        Returns:
            TrainingConfiguration with platform-optimized settings
        """
        caps = self.capabilities
        warnings = []
        recommendations = []

        # Determine QLoRA usage
        use_qlora = requested_qlora and caps.supports_qlora
        if requested_qlora and not caps.supports_qlora:
            warnings.append(
                f"QLoRA requested but not supported on {caps.os_type} with {caps.device_type.upper()}. "
                "QLoRA requires Linux/Windows with NVIDIA CUDA GPU and bitsandbytes library."
            )
            recommendations.append(
                "For QLoRA training: Use a Linux system with NVIDIA GPU and run 'uv sync --extra gpu'"
            )

        # Determine FP16 usage
        if requested_fp16 is not None:
            use_fp16 = requested_fp16 and caps.supports_fp16
            if requested_fp16 and not caps.supports_fp16:
                warnings.append(
                    f"FP16 requested but not supported on {caps.device_type.upper()}. "
                    "FP16 training requires NVIDIA CUDA GPU."
                )
        else:
            # Auto-detect: use FP16 if available and using QLoRA
            use_fp16 = use_qlora and caps.supports_fp16

        # Select optimizer
        if use_qlora:
            optimizer = "paged_adamw_8bit"
        else:
            optimizer = "adamw_torch"

        # Select device mapping
        if caps.device_type == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"  # Works for both CUDA and MPS

        # Select torch dtype
        if use_qlora:
            torch_dtype = "auto"  # Let quantization handle it
        elif caps.has_cuda:
            torch_dtype = "float16"
        else:
            torch_dtype = "float32"  # CPU and MPS use float32

        # Platform-specific recommendations
        if caps.device_type == "cpu":
            recommendations.append(
                "Training on CPU will be significantly slower. "
                "Consider using a system with GPU for better performance."
            )
        elif caps.device_type == "mps":
            recommendations.append(
                "Training on Apple Silicon (MPS). Performance is good but slower than CUDA GPUs. "
                "For production training, consider using a Linux system with NVIDIA GPU."
            )

        return TrainingConfiguration(
            use_fp16=use_fp16,
            use_qlora=use_qlora,
            optimizer=optimizer,
            device_map=device_map,
            torch_dtype=torch_dtype,
            warnings=warnings,
            recommendations=recommendations
        )

    def validate_and_adjust_config(self, config: Dict) -> Dict:
        """
        Validate requested training config and adjust for platform.

        Args:
            config: Requested training configuration dictionary

        Returns:
            Adjusted configuration dictionary compatible with platform
        """
        training_config = config.get("training_config", {})
        requested_qlora = training_config.get("use_qlora", False)

        # Get platform-optimized configuration
        platform_config = self.get_training_config(requested_qlora=requested_qlora)

        # Log warnings
        for warning in platform_config.warnings:
            logger.warning(warning)

        # Log recommendations
        for recommendation in platform_config.recommendations:
            logger.info(f"💡 {recommendation}")

        # Update config with platform-adjusted values
        adjusted_config = config.copy()
        adjusted_config["training_config"] = training_config.copy()
        adjusted_config["training_config"]["use_qlora"] = platform_config.use_qlora
        adjusted_config["platform"] = {
            "device": self.capabilities.device_type,
            "os": self.capabilities.os_type,
            "fp16": platform_config.use_fp16,
            "optimizer": platform_config.optimizer,
            "device_map": platform_config.device_map,
            "torch_dtype": platform_config.torch_dtype
        }

        logger.info(
            f"Training will use: {self.capabilities.device_type.upper()} "
            f"| FP16: {platform_config.use_fp16} "
            f"| QLoRA: {platform_config.use_qlora} "
            f"| Optimizer: {platform_config.optimizer}"
        )

        return adjusted_config

    def get_platform_info(self) -> Dict:
        """
        Get platform information for API responses.

        Returns:
            Dictionary with platform information
        """
        caps = self.capabilities
        return {
            "os": caps.os_type,
            "device": caps.device_type,
            "cuda_available": caps.has_cuda,
            "mps_available": caps.has_mps,
            "fp16_supported": caps.supports_fp16,
            "qlora_supported": caps.supports_qlora
        }