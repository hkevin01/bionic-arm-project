"""
Neural Decoder Module
======================

Decodes motor intent from EEG features using neural networks and
Kalman filtering for smooth, continuous velocity output.

Key Components:
    1. EEGNet: Compact CNN for EEG classification
    2. Kalman Filter: Smooths discrete classifications into continuous commands
    3. Continuous Decoder: Maps features directly to velocity

Mathematical Background:
    
    Kalman Filter State Estimation:
        Predict:  x̂_k|k-1 = F x̂_k-1
                  P_k|k-1 = F P_k-1 F^T + Q
        
        Update:   K = P_k|k-1 H^T (H P_k|k-1 H^T + R)^-1
                  x̂_k = x̂_k|k-1 + K(z_k - H x̂_k|k-1)
                  P_k = (I - K H) P_k|k-1
        
        Where:
            x = state (velocity), z = measurement (decoded output)
            F = state transition, H = observation matrix
            Q = process noise, R = measurement noise
            P = state covariance, K = Kalman gain

Performance Targets:
    - Inference latency: <10ms on GPU, <30ms on CPU
    - Model size: <1MB for embedded deployment
    - Update rate: 30-60 Hz

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch, fall back to numpy-only mode
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-only decoder.")


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class DecoderConfig:
    """
    Configuration for the neural decoder.
    
    Attributes:
        n_features: Number of input features
        n_outputs: Number of output dimensions (e.g., 7 for 7-DOF arm velocity)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
        use_kalman: Whether to apply Kalman filtering to output
        process_noise: Kalman filter process noise (Q)
        measurement_noise: Kalman filter measurement noise (R)
        output_smoothing: Exponential smoothing factor (0-1)
        velocity_scale: Scaling factors for each output dimension
        velocity_limits: Maximum velocity per dimension (rad/s or m/s)
        confidence_threshold: Minimum confidence to output non-zero velocity
    """
    n_features: int = 64
    n_outputs: int = 7
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    use_kalman: bool = True
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    output_smoothing: float = 0.3
    velocity_scale: Optional[np.ndarray] = None
    velocity_limits: Optional[np.ndarray] = None
    confidence_threshold: float = 0.5
    
    def __post_init__(self) -> None:
        """Set defaults for arrays."""
        if self.velocity_scale is None:
            self.velocity_scale = np.ones(self.n_outputs)
        if self.velocity_limits is None:
            self.velocity_limits = np.ones(self.n_outputs) * 1.0  # rad/s


@dataclass
class EEGNetConfig:
    """
    Configuration for EEGNet architecture.
    
    EEGNet is a compact convolutional neural network designed specifically
    for EEG-based BCI applications. It uses depthwise and separable
    convolutions for efficiency.
    
    Architecture:
        Input: (batch, 1, n_channels, n_samples)
        
        Block 1: Temporal convolution → BatchNorm → Depthwise spatial
        Block 2: Separable convolution → BatchNorm → Average pooling
        Classification: Flatten → Dropout → Dense
    
    Reference:
        Lawhern et al., "EEGNet: A Compact Convolutional Network for 
        EEG-based Brain-Computer Interfaces", 2018
    
    Attributes:
        n_channels: Number of EEG channels
        n_samples: Number of time samples per trial
        n_classes: Number of output classes
        F1: Number of temporal filters
        D: Depth multiplier (filters per channel in depthwise conv)
        F2: Number of separable filters (typically F1 * D)
        kernel_length: Temporal filter length (samples)
        dropout_rate: Dropout probability
        pool_size: Pooling window size
    """
    n_channels: int = 32
    n_samples: int = 250
    n_classes: int = 4
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout_rate: float = 0.5
    pool_size: int = 8
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.kernel_length > self.n_samples:
            logger.warning(
                f"kernel_length ({self.kernel_length}) > n_samples ({self.n_samples}). "
                f"Reducing kernel_length to {self.n_samples // 4}"
            )
            self.kernel_length = self.n_samples // 4


# =============================================================================
# Kalman Filter
# =============================================================================

class KalmanFilter:
    """
    Kalman filter for smoothing velocity estimates.
    
    Implements a simple constant velocity model where the state is
    the velocity vector. Process noise allows for velocity changes,
    while measurement noise accounts for decoder uncertainty.
    
    Mathematical Model:
        State: x = [v1, v2, ..., vn]  (velocities)
        
        State transition: x_k = F * x_{k-1} + w
            F = I (constant velocity)
            w ~ N(0, Q) (process noise)
        
        Measurement: z_k = H * x_k + v
            H = I (direct observation)
            v ~ N(0, R) (measurement noise)
    
    Tuning Guidelines:
        - Higher Q: More responsive, less smooth
        - Higher R: Smoother, more lag
        - Q/R ratio determines filter behavior
    
    Example:
        >>> kf = KalmanFilter(n_dims=7, process_noise=0.01, measurement_noise=0.1)
        >>> for measurement in decoded_velocities:
        ...     kf.predict()
        ...     smoothed = kf.update(measurement)
    """
    
    def __init__(
        self,
        n_dims: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ) -> None:
        """
        Initialize Kalman filter.
        
        Args:
            n_dims: Number of state dimensions
            process_noise: Process noise variance (Q diagonal)
            measurement_noise: Measurement noise variance (R diagonal)
        """
        self.n_dims = n_dims
        
        # State estimate: velocity
        self.x = np.zeros(n_dims)
        
        # State covariance
        self.P = np.eye(n_dims) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(n_dims) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(n_dims) * measurement_noise
        
        # State transition matrix (identity for constant velocity)
        self.F = np.eye(n_dims)
        
        # Observation matrix (direct observation)
        self.H = np.eye(n_dims)
        
        # Innovation for diagnostics
        self._last_innovation = np.zeros(n_dims)
        self._innovation_history: List[np.ndarray] = []
        
        logger.debug(
            f"KalmanFilter initialized: {n_dims} dims, "
            f"Q={process_noise:.4f}, R={measurement_noise:.4f}"
        )
    
    def predict(self) -> np.ndarray:
        """
        Prediction step: project state forward.
        
        Returns:
            Predicted state estimate
        """
        # Project state forward
        self.x = self.F @ self.x
        
        # Project covariance forward
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step: incorporate new measurement.
        
        Args:
            z: Measurement vector (decoded velocity)
            
        Returns:
            Updated state estimate
        """
        # Ensure z is numpy array
        z = np.asarray(z).flatten()
        
        if len(z) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dims, got {len(z)}")
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        self._last_innovation = y.copy()
        self._innovation_history.append(y)
        
        # Keep only recent history
        if len(self._innovation_history) > 100:
            self._innovation_history.pop(0)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudoinverse")
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n_dims) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def filter(self, z: np.ndarray) -> np.ndarray:
        """
        Predict and update in one step.
        
        Args:
            z: Measurement vector
            
        Returns:
            Filtered state estimate
        """
        self.predict()
        return self.update(z)
    
    def reset(self) -> None:
        """Reset filter state to initial conditions."""
        self.x = np.zeros(self.n_dims)
        self.P = np.eye(self.n_dims) * 1.0
        self._last_innovation = np.zeros(self.n_dims)
        self._innovation_history.clear()
        logger.debug("KalmanFilter reset")
    
    def set_state(self, x: np.ndarray) -> None:
        """
        Set filter state directly.
        
        Useful for initialization based on prior knowledge.
        
        Args:
            x: New state estimate
        """
        self.x = np.asarray(x).flatten().copy()
    
    @property
    def innovation(self) -> np.ndarray:
        """Last innovation (measurement residual)."""
        return self._last_innovation
    
    @property
    def innovation_std(self) -> np.ndarray:
        """Standard deviation of recent innovations (for monitoring)."""
        if len(self._innovation_history) < 2:
            return np.zeros(self.n_dims)
        return np.std(self._innovation_history, axis=0)
    
    def get_confidence(self) -> float:
        """
        Estimate filter confidence based on covariance.
        
        Returns:
            Confidence score (0-1), higher is more confident
        """
        # Use trace of covariance as uncertainty measure
        uncertainty = np.trace(self.P)
        # Convert to confidence (inverse, bounded)
        confidence = 1.0 / (1.0 + uncertainty)
        return float(confidence)


# =============================================================================
# Neural Network Models (PyTorch)
# =============================================================================

if TORCH_AVAILABLE:
    
    class EEGNet(nn.Module):
        """
        EEGNet: Compact CNN for EEG classification.
        
        Architecture designed specifically for EEG with:
        - Temporal convolution (captures frequency information)
        - Depthwise spatial convolution (learns channel combinations)
        - Separable convolution (efficient feature refinement)
        
        Total parameters: ~2,500 (very compact)
        
        Reference:
            Lawhern et al., "EEGNet: A Compact Convolutional Network for 
            EEG-based Brain-Computer Interfaces", Journal of Neural 
            Engineering, 2018
        """
        
        def __init__(self, config: EEGNetConfig) -> None:
            """
            Initialize EEGNet.
            
            Args:
                config: EEGNet configuration
            """
            super().__init__()
            self.config = config
            
            # Block 1: Temporal + Spatial filtering
            # Temporal convolution across time
            self.temporal_conv = nn.Conv2d(
                1, config.F1,
                kernel_size=(1, config.kernel_length),
                padding=(0, config.kernel_length // 2),
                bias=False
            )
            self.temporal_bn = nn.BatchNorm2d(config.F1)
            
            # Depthwise convolution across channels
            self.spatial_conv = nn.Conv2d(
                config.F1, config.F1 * config.D,
                kernel_size=(config.n_channels, 1),
                groups=config.F1,
                bias=False
            )
            self.spatial_bn = nn.BatchNorm2d(config.F1 * config.D)
            self.pool1 = nn.AvgPool2d(kernel_size=(1, config.pool_size))
            self.dropout1 = nn.Dropout(config.dropout_rate)
            
            # Block 2: Separable convolution
            # Depthwise
            self.separable_depth = nn.Conv2d(
                config.F1 * config.D, config.F1 * config.D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=config.F1 * config.D,
                bias=False
            )
            # Pointwise
            self.separable_point = nn.Conv2d(
                config.F1 * config.D, config.F2,
                kernel_size=(1, 1),
                bias=False
            )
            self.separable_bn = nn.BatchNorm2d(config.F2)
            self.pool2 = nn.AvgPool2d(kernel_size=(1, config.pool_size))
            self.dropout2 = nn.Dropout(config.dropout_rate)
            
            # Calculate flattened size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, config.n_channels, config.n_samples)
                dummy = self._forward_features(dummy)
                self._flat_size = dummy.numel()
            
            # Classification layer
            self.classifier = nn.Linear(self._flat_size, config.n_classes)
            
            # Initialize weights
            self._init_weights()
            
            logger.info(
                f"EEGNet initialized: {self._count_parameters()} parameters, "
                f"input ({config.n_channels}, {config.n_samples}) → {config.n_classes} classes"
            )
        
        def _init_weights(self) -> None:
            """Initialize weights using Xavier/He initialization."""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        def _count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through feature extraction layers."""
            # Block 1
            x = self.temporal_conv(x)
            x = self.temporal_bn(x)
            x = self.spatial_conv(x)
            x = self.spatial_bn(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            
            # Block 2
            x = self.separable_depth(x)
            x = self.separable_point(x)
            x = self.separable_bn(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            return x
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor, shape (batch, 1, n_channels, n_samples)
                   or (batch, n_channels, n_samples)
            
            Returns:
                Class logits, shape (batch, n_classes)
            """
            # Add channel dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            # Feature extraction
            x = self._forward_features(x)
            
            # Flatten
            x = x.flatten(1)
            
            # Classification
            x = self.classifier(x)
            
            return x
        
        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """
            Get class probabilities.
            
            Args:
                x: Input tensor
                
            Returns:
                Class probabilities, shape (batch, n_classes)
            """
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
        
        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            """
            Extract features without classification.
            
            Useful for transfer learning or combining with other decoders.
            
            Args:
                x: Input tensor
                
            Returns:
                Feature vector, shape (batch, n_features)
            """
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self._forward_features(x)
            return x.flatten(1)
    
    
    class FeatureMapper(nn.Module):
        """
        Neural network that maps EEG features to velocity commands.
        
        Takes extracted features (e.g., CSP log-variance) and outputs
        continuous velocity estimates for each DOF.
        
        Architecture:
            Features → Linear → LayerNorm → GELU → Dropout
                    → Linear → LayerNorm → GELU → Dropout
                    → Linear → Velocity
        """
        
        def __init__(self, config: DecoderConfig) -> None:
            """
            Initialize feature mapper.
            
            Args:
                config: Decoder configuration
            """
            super().__init__()
            self.config = config
            
            layers = []
            in_dim = config.n_features
            
            for hidden_dim in config.hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout)
                ])
                in_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, config.n_outputs))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize output layer to small values
            nn.init.xavier_normal_(self.network[-1].weight, gain=0.1)
            nn.init.zeros_(self.network[-1].bias)
            
            logger.info(
                f"FeatureMapper: {config.n_features} → {config.hidden_dims} → {config.n_outputs}"
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Feature tensor, shape (batch, n_features)
                
            Returns:
                Velocity tensor, shape (batch, n_outputs)
            """
            return self.network(x)


# =============================================================================
# Main Decoder Classes
# =============================================================================

class ContinuousDecoder:
    """
    Complete continuous decoder combining neural network feature mapping
    with Kalman filtering for smooth velocity output.
    
    Pipeline:
        Features → Neural Network → Raw Velocity
                → Kalman Filter → Smoothed Velocity
                → Scaling & Limiting → Output Command
    
    Thread Safety:
        Not thread-safe. Use separate instances for concurrent processing.
    
    Example:
        >>> config = DecoderConfig(n_features=16, n_outputs=7)
        >>> decoder = ContinuousDecoder(config)
        >>> decoder.load_weights("model.pt")
        >>> 
        >>> for features in feature_stream:
        ...     velocity, confidence = decoder.decode(features)
        ...     send_to_arm(velocity)
    """
    
    def __init__(
        self,
        config: DecoderConfig,
        device: str = "cuda"
    ) -> None:
        """
        Initialize continuous decoder.
        
        Args:
            config: Decoder configuration
            device: Device for neural network ("cuda" or "cpu")
        """
        self.config = config
        
        # Set device
        if TORCH_AVAILABLE:
            self.device = torch.device(
                device if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = None
        
        # Neural network for feature mapping
        if TORCH_AVAILABLE:
            self.feature_mapper = FeatureMapper(config).to(self.device)
            self.feature_mapper.eval()
        else:
            self.feature_mapper = None
            logger.warning("Using simple linear decoder (PyTorch not available)")
            # Fallback: simple linear weights
            self._linear_weights = np.random.randn(
                config.n_features, config.n_outputs
            ) * 0.01
            self._linear_bias = np.zeros(config.n_outputs)
        
        # Kalman filter for smoothing
        if config.use_kalman:
            self.kalman = KalmanFilter(
                n_dims=config.n_outputs,
                process_noise=config.process_noise,
                measurement_noise=config.measurement_noise
            )
        else:
            self.kalman = None
        
        # Velocity scaling and limits
        self.velocity_scale = np.asarray(config.velocity_scale).flatten()
        self.velocity_limits = np.asarray(config.velocity_limits).flatten()
        
        # State
        self._last_output = np.zeros(config.n_outputs)
        self._confidence = 0.0
        self._inference_times: List[float] = []
        
        # Exponential smoothing state
        self._smoothed_output = np.zeros(config.n_outputs)
        
        logger.info(
            f"ContinuousDecoder initialized: {config.n_features} features → "
            f"{config.n_outputs} outputs, device={self.device}, "
            f"kalman={config.use_kalman}"
        )
    
    def decode(
        self,
        features: np.ndarray,
        return_raw: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Decode EEG features to velocity command.
        
        Args:
            features: EEG features, shape (n_features,)
            return_raw: Whether to return raw (unfiltered) output as well
            
        Returns:
            Tuple of (velocity_command, confidence)
            velocity_command shape: (n_outputs,)
            
        Raises:
            ValueError: If features have wrong shape
        """
        features = np.asarray(features).flatten()
        
        if len(features) != self.config.n_features:
            raise ValueError(
                f"Expected {self.config.n_features} features, got {len(features)}"
            )
        
        start_time = time.perf_counter()
        
        # Neural network inference
        if TORCH_AVAILABLE and self.feature_mapper is not None:
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                raw_output = self.feature_mapper(x).cpu().numpy().squeeze()
        else:
            # Fallback linear decoder
            raw_output = features @ self._linear_weights + self._linear_bias
        
        # Kalman filter prediction and update
        if self.kalman is not None:
            self.kalman.predict()
            filtered_output = self.kalman.update(raw_output)
            confidence = self.kalman.get_confidence()
        else:
            filtered_output = raw_output
            confidence = 1.0
        
        # Exponential smoothing
        alpha = self.config.output_smoothing
        self._smoothed_output = alpha * filtered_output + (1 - alpha) * self._smoothed_output
        
        # Apply scaling and limits
        scaled_output = self._smoothed_output * self.velocity_scale
        clipped_output = np.clip(scaled_output, -self.velocity_limits, self.velocity_limits)
        
        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            clipped_output *= confidence / self.config.confidence_threshold
        
        # Record timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(elapsed_ms)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        
        self._last_output = clipped_output
        self._confidence = confidence
        
        if return_raw:
            return clipped_output, confidence, raw_output
        return clipped_output, confidence
    
    def decode_batch(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode batch of features (without Kalman filtering).
        
        Args:
            features: Feature batch, shape (n_samples, n_features)
            
        Returns:
            Tuple of (velocities, confidences)
            velocities shape: (n_samples, n_outputs)
        """
        n_samples = features.shape[0]
        velocities = np.zeros((n_samples, self.config.n_outputs))
        confidences = np.ones(n_samples)
        
        for i in range(n_samples):
            velocities[i], confidences[i] = self.decode(features[i])
        
        return velocities, confidences
    
    def load_weights(self, path: str) -> None:
        """
        Load pretrained weights.
        
        Args:
            path: Path to weights file (.pt or .npz)
        """
        if TORCH_AVAILABLE and self.feature_mapper is not None:
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.feature_mapper.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.feature_mapper.load_state_dict(checkpoint)
                
                # Load optional parameters
                if "velocity_scale" in checkpoint:
                    self.velocity_scale = np.asarray(checkpoint["velocity_scale"])
                if "velocity_limits" in checkpoint:
                    self.velocity_limits = np.asarray(checkpoint["velocity_limits"])
            else:
                self.feature_mapper.load_state_dict(checkpoint)
            
            self.feature_mapper.eval()
            logger.info(f"Loaded weights from {path}")
        else:
            # Load numpy weights
            data = np.load(path)
            self._linear_weights = data["weights"]
            self._linear_bias = data["bias"]
            if "velocity_scale" in data:
                self.velocity_scale = data["velocity_scale"]
            if "velocity_limits" in data:
                self.velocity_limits = data["velocity_limits"]
            logger.info(f"Loaded numpy weights from {path}")
    
    def save_weights(self, path: str) -> None:
        """
        Save current weights.
        
        Args:
            path: Path to save weights
        """
        if TORCH_AVAILABLE and self.feature_mapper is not None:
            torch.save({
                "model_state_dict": self.feature_mapper.state_dict(),
                "velocity_scale": self.velocity_scale,
                "velocity_limits": self.velocity_limits,
                "config": self.config
            }, path)
        else:
            np.savez(
                path,
                weights=self._linear_weights,
                bias=self._linear_bias,
                velocity_scale=self.velocity_scale,
                velocity_limits=self.velocity_limits
            )
        logger.info(f"Saved weights to {path}")
    
    def reset(self) -> None:
        """Reset decoder state."""
        if self.kalman is not None:
            self.kalman.reset()
        self._last_output = np.zeros(self.config.n_outputs)
        self._smoothed_output = np.zeros(self.config.n_outputs)
        self._confidence = 0.0
        logger.debug("Decoder state reset")
    
    @property
    def last_output(self) -> np.ndarray:
        """Last decoded velocity output."""
        return self._last_output.copy()
    
    @property
    def confidence(self) -> float:
        """Current decoder confidence."""
        return self._confidence
    
    @property
    def mean_inference_time_ms(self) -> float:
        """Mean inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return float(np.mean(self._inference_times))
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information for monitoring.
        
        Returns:
            Dictionary with diagnostic data
        """
        diag = {
            "last_output": self._last_output.tolist(),
            "confidence": self._confidence,
            "mean_inference_ms": self.mean_inference_time_ms,
            "device": str(self.device),
        }
        
        if self.kalman is not None:
            diag["kalman_state"] = self.kalman.x.tolist()
            diag["kalman_innovation_std"] = self.kalman.innovation_std.tolist()
        
        return diag


class ClassifierDecoder:
    """
    Discrete classifier-based decoder.
    
    Uses EEGNet or similar classifier to predict motor imagery class,
    then maps classes to velocity profiles.
    
    Class Mapping:
        0 (rest)  → zero velocity
        1 (left)  → negative velocity on specified joints
        2 (right) → positive velocity on specified joints
        3 (both)  → combined velocity pattern
    
    Example:
        >>> decoder = ClassifierDecoder(eegnet_config, class_mapping)
        >>> decoder.load_model("eegnet.pt")
        >>> velocity = decoder.decode(eeg_trial)
    """
    
    def __init__(
        self,
        model_config: EEGNetConfig,
        class_velocity_mapping: Dict[int, np.ndarray],
        confidence_threshold: float = 0.6
    ) -> None:
        """
        Initialize classifier decoder.
        
        Args:
            model_config: EEGNet configuration
            class_velocity_mapping: Dict mapping class ID to velocity vector
            confidence_threshold: Minimum probability to execute movement
        """
        self.model_config = model_config
        self.class_mapping = class_velocity_mapping
        self.confidence_threshold = confidence_threshold
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = EEGNet(model_config).to(self.device)
            self.model.eval()
        else:
            self.model = None
            logger.warning("PyTorch not available for ClassifierDecoder")
        
        self._last_class = 0
        self._last_confidence = 0.0
    
    def decode(self, trial: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Decode EEG trial to velocity command.
        
        Args:
            trial: EEG data, shape (n_channels, n_samples)
            
        Returns:
            Tuple of (velocity, predicted_class, confidence)
        """
        if self.model is None:
            # Fallback: return zero velocity
            n_outputs = next(iter(self.class_mapping.values())).shape[0]
            return np.zeros(n_outputs), 0, 0.0
        
        with torch.no_grad():
            x = torch.from_numpy(trial).float().unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(x).cpu().numpy().squeeze()
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        # Get velocity for predicted class
        if predicted_class in self.class_mapping:
            velocity = self.class_mapping[predicted_class].copy()
        else:
            velocity = np.zeros_like(next(iter(self.class_mapping.values())))
        
        # Scale by confidence if below threshold
        if confidence < self.confidence_threshold:
            velocity *= confidence / self.confidence_threshold
        
        self._last_class = predicted_class
        self._last_confidence = confidence
        
        return velocity, predicted_class, confidence
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        if self.model is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {path}")
    
    @property
    def last_prediction(self) -> Tuple[int, float]:
        """Last predicted class and confidence."""
        return self._last_class, self._last_confidence
