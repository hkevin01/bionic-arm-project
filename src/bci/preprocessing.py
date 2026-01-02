"""
EEG Signal Preprocessing Module
================================

Comprehensive preprocessing pipeline for EEG signals including:
- Bandpass and notch filtering
- Artifact detection and rejection
- Spatial filtering (Common Average Reference, Laplacian)
- Baseline correction and normalization

Mathematical Background:
    1. Butterworth Filter: H(s) = 1 / sqrt(1 + (s/ωc)^2n)
    2. Notch Filter: Removes specific frequency (50/60 Hz power line)
    3. Common Average Reference: x_car[i] = x[i] - mean(x)
    4. Z-score normalization: z = (x - μ) / σ

Performance Considerations:
    - Uses scipy.signal for efficient IIR filtering
    - Supports chunk-based processing for real-time use
    - Filter states maintained for continuous processing
    - Memory-efficient with in-place operations where possible

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy import signal
from scipy.stats import zscore
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations and Constants
# =============================================================================

class FilterType(Enum):
    """Types of frequency filters available."""
    LOWPASS = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    NOTCH = auto()


class ReferenceType(Enum):
    """Types of spatial reference schemes."""
    NONE = auto()
    COMMON_AVERAGE = auto()      # CAR: subtract mean of all channels
    LINKED_MASTOID = auto()      # Average of mastoid electrodes
    LAPLACIAN = auto()           # Surface Laplacian (requires electrode positions)
    BIPOLAR = auto()             # Difference between adjacent channels


class ArtifactType(Enum):
    """Types of artifacts that can be detected."""
    EYE_BLINK = auto()
    EYE_MOVEMENT = auto()
    MUSCLE = auto()
    ELECTRODE_POP = auto()
    SATURATION = auto()
    FLAT_LINE = auto()
    HIGH_FREQUENCY_NOISE = auto()


# EEG frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'mu': (8.0, 12.0),       # Sensorimotor rhythm
    'beta': (13.0, 30.0),
    'low_beta': (13.0, 20.0),
    'high_beta': (20.0, 30.0),
    'gamma': (30.0, 100.0),
    'low_gamma': (30.0, 50.0),
    'high_gamma': (50.0, 100.0),
}


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class FilterConfig:
    """
    Configuration for a single filter.
    
    Attributes:
        filter_type: Type of filter (lowpass, highpass, bandpass, notch)
        low_freq: Low cutoff frequency in Hz (for highpass/bandpass)
        high_freq: High cutoff frequency in Hz (for lowpass/bandpass)
        order: Filter order (higher = sharper cutoff, more delay)
        notch_freq: Center frequency for notch filter
        notch_q: Quality factor for notch filter (higher = narrower notch)
    """
    filter_type: FilterType
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None
    order: int = 4
    notch_freq: Optional[float] = None
    notch_q: float = 30.0
    
    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if self.filter_type == FilterType.BANDPASS:
            if self.low_freq is None or self.high_freq is None:
                raise ValueError("Bandpass filter requires both low_freq and high_freq")
            if self.low_freq >= self.high_freq:
                raise ValueError(f"low_freq ({self.low_freq}) must be < high_freq ({self.high_freq})")
        elif self.filter_type == FilterType.HIGHPASS:
            if self.low_freq is None:
                raise ValueError("Highpass filter requires low_freq")
        elif self.filter_type == FilterType.LOWPASS:
            if self.high_freq is None:
                raise ValueError("Lowpass filter requires high_freq")
        elif self.filter_type == FilterType.NOTCH:
            if self.notch_freq is None:
                raise ValueError("Notch filter requires notch_freq")


@dataclass
class ArtifactConfig:
    """
    Configuration for artifact detection.
    
    Attributes:
        enable_detection: Whether to detect artifacts
        amplitude_threshold_uv: Maximum amplitude before marking as artifact
        flat_line_threshold_uv: Minimum variation to detect flat line
        flat_line_duration_samples: Samples needed to confirm flat line
        zscore_threshold: Z-score threshold for statistical outlier detection
        gradient_threshold_uv: Maximum sample-to-sample change
        reject_artifacts: Whether to reject/interpolate detected artifacts
    """
    enable_detection: bool = True
    amplitude_threshold_uv: float = 100.0      # µV, typical EEG is ±50µV
    flat_line_threshold_uv: float = 0.5        # Below this = flat line
    flat_line_duration_samples: int = 50       # ~200ms at 250Hz
    zscore_threshold: float = 4.0              # Statistical outlier
    gradient_threshold_uv: float = 50.0        # Sudden jumps
    reject_artifacts: bool = False             # Mark only vs. reject


@dataclass
class PreprocessorConfig:
    """
    Complete preprocessing pipeline configuration.
    
    Attributes:
        sampling_rate: Signal sampling rate in Hz
        n_channels: Number of EEG channels
        filters: List of filter configurations to apply in order
        reference: Spatial reference scheme
        reference_channels: Indices of reference channels (for linked mastoid)
        artifact_config: Artifact detection configuration
        normalize: Whether to apply z-score normalization
        normalize_window_seconds: Window for running normalization
        baseline_correction: Whether to apply baseline correction
        baseline_window_seconds: Window for baseline estimation
    """
    sampling_rate: int = 250
    n_channels: int = 32
    filters: List[FilterConfig] = field(default_factory=list)
    reference: ReferenceType = ReferenceType.COMMON_AVERAGE
    reference_channels: Optional[List[int]] = None
    artifact_config: ArtifactConfig = field(default_factory=ArtifactConfig)
    normalize: bool = True
    normalize_window_seconds: float = 2.0
    baseline_correction: bool = True
    baseline_window_seconds: float = 0.5
    
    def __post_init__(self) -> None:
        """Set up default filters if none provided."""
        if not self.filters:
            # Default preprocessing chain:
            # 1. Bandpass 0.5-50 Hz (remove DC and high-frequency noise)
            # 2. Notch at power line frequency
            self.filters = [
                FilterConfig(
                    filter_type=FilterType.BANDPASS,
                    low_freq=0.5,
                    high_freq=50.0,
                    order=4
                ),
                FilterConfig(
                    filter_type=FilterType.NOTCH,
                    notch_freq=60.0,  # Adjust for your region
                    notch_q=30.0
                )
            ]
    
    @classmethod
    def for_motor_imagery(cls, sampling_rate: int = 250, n_channels: int = 32) -> 'PreprocessorConfig':
        """
        Create configuration optimized for motor imagery BCI.
        
        Motor imagery uses mu (8-12 Hz) and beta (13-30 Hz) rhythms,
        so we optimize filtering for these bands.
        """
        return cls(
            sampling_rate=sampling_rate,
            n_channels=n_channels,
            filters=[
                FilterConfig(FilterType.BANDPASS, low_freq=4.0, high_freq=40.0, order=4),
                FilterConfig(FilterType.NOTCH, notch_freq=60.0),
            ],
            reference=ReferenceType.COMMON_AVERAGE,
            normalize=True,
            normalize_window_seconds=1.0,
        )


# =============================================================================
# Filter Implementation
# =============================================================================

class DigitalFilter:
    """
    Stateful digital IIR filter for continuous processing.
    
    Maintains filter state between calls to process() for seamless
    chunk-based filtering without transients at chunk boundaries.
    
    Mathematical Details:
        Uses Butterworth filter design for maximally flat passband.
        Transfer function: H(s) = 1 / sqrt(1 + (ω/ωc)^2n)
        
        For real-time filtering, uses Direct Form II transposed
        implementation via scipy.signal.sosfilt for numerical stability.
    """
    
    def __init__(self, config: FilterConfig, sampling_rate: float, n_channels: int) -> None:
        """
        Initialize filter with configuration.
        
        Args:
            config: Filter configuration
            sampling_rate: Signal sampling rate in Hz
            n_channels: Number of channels to filter
        """
        self.config = config
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        # Design filter
        self._sos = self._design_filter()
        
        # Initialize filter state for each channel
        # State shape: (n_sections, 2) per channel
        self._zi = self._init_state()
        
        logger.debug(
            f"Created {config.filter_type.name} filter: "
            f"order={config.order}, fs={sampling_rate}"
        )
    
    def _design_filter(self) -> np.ndarray:
        """
        Design the filter and return second-order sections.
        
        Returns:
            SOS (second-order sections) filter coefficients
        """
        nyquist = self.sampling_rate / 2.0
        
        if self.config.filter_type == FilterType.BANDPASS:
            low = self.config.low_freq / nyquist
            high = self.config.high_freq / nyquist
            
            # Validate frequencies are in valid range
            if low <= 0 or high >= 1:
                raise ValueError(
                    f"Filter frequencies must be between 0 and Nyquist ({nyquist} Hz)"
                )
            
            sos = signal.butter(
                self.config.order,
                [low, high],
                btype='bandpass',
                output='sos'
            )
            
        elif self.config.filter_type == FilterType.HIGHPASS:
            low = self.config.low_freq / nyquist
            if low <= 0 or low >= 1:
                raise ValueError(f"Cutoff frequency {self.config.low_freq} Hz is invalid")
            
            sos = signal.butter(
                self.config.order,
                low,
                btype='highpass',
                output='sos'
            )
            
        elif self.config.filter_type == FilterType.LOWPASS:
            high = self.config.high_freq / nyquist
            if high <= 0 or high >= 1:
                raise ValueError(f"Cutoff frequency {self.config.high_freq} Hz is invalid")
            
            sos = signal.butter(
                self.config.order,
                high,
                btype='lowpass',
                output='sos'
            )
            
        elif self.config.filter_type == FilterType.NOTCH:
            freq = self.config.notch_freq / nyquist
            if freq <= 0 or freq >= 1:
                raise ValueError(f"Notch frequency {self.config.notch_freq} Hz is invalid")
            
            # Design notch using iirnotch
            b, a = signal.iirnotch(freq, self.config.notch_q)
            # Convert to SOS for numerical stability
            sos = signal.tf2sos(b, a)
            
        else:
            raise ValueError(f"Unknown filter type: {self.config.filter_type}")
        
        return sos
    
    def _init_state(self) -> np.ndarray:
        """
        Initialize filter state for all channels.
        
        Returns:
            Initial filter state array
        """
        # Get state shape for one channel
        zi_single = signal.sosfilt_zi(self._sos)
        
        # Replicate for all channels
        # Shape: (n_channels, n_sections, 2)
        zi = np.zeros((self.n_channels, *zi_single.shape))
        for ch in range(self.n_channels):
            zi[ch] = zi_single
        
        return zi
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply filter to data, maintaining state between calls.
        
        Args:
            data: Input data, shape (n_channels, n_samples)
            
        Returns:
            Filtered data, same shape as input
            
        Note:
            Modifies internal state, so consecutive calls produce
            continuous filtered output without edge artifacts.
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[0]}"
            )
        
        filtered = np.zeros_like(data)
        
        for ch in range(self.n_channels):
            filtered[ch], self._zi[ch] = signal.sosfilt(
                self._sos,
                data[ch],
                zi=self._zi[ch]
            )
        
        return filtered
    
    def reset(self) -> None:
        """Reset filter state to initial conditions."""
        self._zi = self._init_state()


# =============================================================================
# Artifact Detection
# =============================================================================

@dataclass
class ArtifactMarker:
    """
    Marker for detected artifact.
    
    Attributes:
        artifact_type: Type of artifact detected
        channel: Channel index (or -1 for global)
        start_sample: Start index of artifact
        end_sample: End index of artifact
        severity: Severity score (0-1)
        details: Additional information
    """
    artifact_type: ArtifactType
    channel: int
    start_sample: int
    end_sample: int
    severity: float = 1.0
    details: Optional[str] = None


class ArtifactDetector:
    """
    Detects various types of artifacts in EEG signals.
    
    Implements multiple detection algorithms:
    - Amplitude thresholding: Detects saturation and large transients
    - Flat line detection: Detects disconnected electrodes
    - Statistical outliers: Detects unusual signal characteristics
    - Gradient analysis: Detects sudden jumps (electrode pops)
    
    Usage:
        detector = ArtifactDetector(config)
        markers = detector.detect(data)
        clean_mask = detector.get_clean_mask(data)
    """
    
    def __init__(self, config: ArtifactConfig) -> None:
        """
        Initialize artifact detector.
        
        Args:
            config: Artifact detection configuration
        """
        self.config = config
        
        # Running statistics for adaptive thresholding
        self._running_mean: Optional[np.ndarray] = None
        self._running_std: Optional[np.ndarray] = None
        self._n_samples_seen = 0
    
    def detect(
        self,
        data: np.ndarray,
        return_details: bool = False
    ) -> Tuple[np.ndarray, List[ArtifactMarker]]:
        """
        Detect artifacts in data.
        
        Args:
            data: EEG data, shape (n_channels, n_samples)
            return_details: Whether to return detailed markers
            
        Returns:
            Tuple of:
                - Boolean mask, shape (n_channels, n_samples), True = artifact
                - List of ArtifactMarker (if return_details=True, else empty)
        """
        n_channels, n_samples = data.shape
        artifact_mask = np.zeros((n_channels, n_samples), dtype=bool)
        markers: List[ArtifactMarker] = []
        
        if not self.config.enable_detection:
            return artifact_mask, markers
        
        # 1. Amplitude threshold detection
        amp_artifacts = np.abs(data) > self.config.amplitude_threshold_uv
        artifact_mask |= amp_artifacts
        
        if return_details and np.any(amp_artifacts):
            for ch in range(n_channels):
                if np.any(amp_artifacts[ch]):
                    # Find contiguous regions
                    regions = self._find_regions(amp_artifacts[ch])
                    for start, end in regions:
                        markers.append(ArtifactMarker(
                            artifact_type=ArtifactType.SATURATION,
                            channel=ch,
                            start_sample=start,
                            end_sample=end,
                            severity=float(np.max(np.abs(data[ch, start:end])) / 
                                         self.config.amplitude_threshold_uv)
                        ))
        
        # 2. Flat line detection
        for ch in range(n_channels):
            flat_mask = self._detect_flat_line(data[ch])
            artifact_mask[ch] |= flat_mask
            
            if return_details and np.any(flat_mask):
                regions = self._find_regions(flat_mask)
                for start, end in regions:
                    markers.append(ArtifactMarker(
                        artifact_type=ArtifactType.FLAT_LINE,
                        channel=ch,
                        start_sample=start,
                        end_sample=end,
                        severity=1.0
                    ))
        
        # 3. Gradient-based detection (electrode pops)
        gradient = np.abs(np.diff(data, axis=1))
        # Pad to match original size
        gradient = np.concatenate([gradient, gradient[:, -1:]], axis=1)
        gradient_artifacts = gradient > self.config.gradient_threshold_uv
        artifact_mask |= gradient_artifacts
        
        if return_details and np.any(gradient_artifacts):
            for ch in range(n_channels):
                if np.any(gradient_artifacts[ch]):
                    regions = self._find_regions(gradient_artifacts[ch])
                    for start, end in regions:
                        markers.append(ArtifactMarker(
                            artifact_type=ArtifactType.ELECTRODE_POP,
                            channel=ch,
                            start_sample=start,
                            end_sample=end,
                            severity=float(np.max(gradient[ch, start:end]) /
                                         self.config.gradient_threshold_uv)
                        ))
        
        # 4. Statistical outlier detection (z-score)
        if self._running_std is not None:
            # Use running statistics for adaptive detection
            z = np.abs((data - self._running_mean[:, np.newaxis]) / 
                      (self._running_std[:, np.newaxis] + 1e-10))
            zscore_artifacts = z > self.config.zscore_threshold
            artifact_mask |= zscore_artifacts
        
        # Update running statistics
        self._update_statistics(data)
        
        return artifact_mask, markers
    
    def _detect_flat_line(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Detect flat line (disconnected electrode) in single channel.
        
        Args:
            channel_data: 1D array of samples
            
        Returns:
            Boolean mask of flat line samples
        """
        n_samples = len(channel_data)
        mask = np.zeros(n_samples, dtype=bool)
        
        # Compute local standard deviation
        window = self.config.flat_line_duration_samples
        if n_samples < window:
            return mask
        
        # Use stride tricks for efficient windowed std
        shape = (n_samples - window + 1, window)
        strides = (channel_data.strides[0], channel_data.strides[0])
        windowed = np.lib.stride_tricks.as_strided(channel_data, shape=shape, strides=strides)
        local_std = np.std(windowed, axis=1)
        
        # Mark samples where std is below threshold
        flat_indices = np.where(local_std < self.config.flat_line_threshold_uv)[0]
        for idx in flat_indices:
            mask[idx:idx + window] = True
        
        return mask
    
    def _find_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find contiguous True regions in boolean mask.
        
        Args:
            mask: 1D boolean array
            
        Returns:
            List of (start, end) tuples
        """
        # Add False at boundaries to handle edge cases
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))
    
    def _update_statistics(self, data: np.ndarray) -> None:
        """
        Update running mean and std for adaptive detection.
        
        Uses Welford's online algorithm for numerical stability.
        """
        n_channels, n_samples = data.shape
        
        if self._running_mean is None:
            self._running_mean = np.mean(data, axis=1)
            self._running_std = np.std(data, axis=1)
            self._n_samples_seen = n_samples
        else:
            # Welford's algorithm for running statistics
            for i in range(n_samples):
                self._n_samples_seen += 1
                sample = data[:, i]
                delta = sample - self._running_mean
                self._running_mean += delta / self._n_samples_seen
                delta2 = sample - self._running_mean
                # Running variance (M2 in Welford's algorithm)
                # Simplified update for std
                alpha = 0.01  # Exponential moving average factor
                self._running_std = (1 - alpha) * self._running_std + alpha * np.abs(delta)
    
    def get_clean_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Get mask of clean (artifact-free) samples.
        
        Args:
            data: EEG data, shape (n_channels, n_samples)
            
        Returns:
            Boolean mask, True = clean sample
        """
        artifact_mask, _ = self.detect(data)
        # Sample is clean if all channels are clean
        return ~np.any(artifact_mask, axis=0)
    
    def reset(self) -> None:
        """Reset running statistics."""
        self._running_mean = None
        self._running_std = None
        self._n_samples_seen = 0


# =============================================================================
# Main Preprocessor Class
# =============================================================================

class Preprocessor:
    """
    Complete EEG preprocessing pipeline.
    
    Pipeline stages (in order):
        1. Baseline correction (optional)
        2. Spatial reference (CAR, Laplacian, etc.)
        3. Frequency filtering (bandpass, notch)
        4. Artifact detection
        5. Normalization (optional)
    
    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │                     Preprocessor                            │
        │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
        │  │ Baseline │──▶│Reference │──▶│ Filters  │──▶│ Artifact │ │
        │  │Correction│   │  (CAR)   │   │ (IIR)    │   │ Detection│ │
        │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
        │                                      │              │       │
        │                                      ▼              ▼       │
        │                               ┌──────────┐   ┌──────────┐  │
        │                               │Normalize │   │  Mask    │  │
        │                               │ (Z-score)│   │  Output  │  │
        │                               └──────────┘   └──────────┘  │
        └─────────────────────────────────────────────────────────────┘
    
    Thread Safety:
        Process calls are NOT thread-safe. Use separate instances
        for concurrent processing, or synchronize externally.
    
    Example:
        >>> config = PreprocessorConfig.for_motor_imagery(sampling_rate=250)
        >>> preprocessor = Preprocessor(config)
        >>> 
        >>> # Process in chunks (real-time)
        >>> for chunk in acquisition.get_chunks():
        >>>     processed = preprocessor.process(chunk)
        >>>     classifier.predict(processed)
    """
    
    def __init__(self, config: PreprocessorConfig) -> None:
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        
        # Create filter chain
        self._filters: List[DigitalFilter] = []
        for filter_config in config.filters:
            self._filters.append(DigitalFilter(
                filter_config,
                config.sampling_rate,
                config.n_channels
            ))
        
        # Create artifact detector
        self._artifact_detector = ArtifactDetector(config.artifact_config)
        
        # Normalization state
        self._norm_buffer: Optional[np.ndarray] = None
        self._norm_buffer_idx = 0
        norm_samples = int(config.normalize_window_seconds * config.sampling_rate)
        if config.normalize:
            self._norm_buffer = np.zeros((config.n_channels, norm_samples))
        
        # Baseline state
        self._baseline_buffer: Optional[np.ndarray] = None
        self._baseline_buffer_idx = 0
        baseline_samples = int(config.baseline_window_seconds * config.sampling_rate)
        if config.baseline_correction:
            self._baseline_buffer = np.zeros((config.n_channels, baseline_samples))
        
        # Statistics
        self._samples_processed = 0
        self._artifacts_detected = 0
        
        logger.info(
            f"Initialized Preprocessor: {len(self._filters)} filters, "
            f"reference={config.reference.name}, normalize={config.normalize}"
        )
    
    def process(
        self,
        data: np.ndarray,
        return_artifact_mask: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Process EEG data through the complete pipeline.
        
        Args:
            data: Raw EEG data, shape (n_channels, n_samples)
            return_artifact_mask: Whether to also return artifact mask
            
        Returns:
            Processed data, shape (n_channels, n_samples)
            If return_artifact_mask=True: tuple of (data, mask)
            
        Raises:
            ValueError: If data shape doesn't match configuration
        """
        # Validate input
        if data.shape[0] != self.config.n_channels:
            raise ValueError(
                f"Expected {self.config.n_channels} channels, got {data.shape[0]}"
            )
        
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        n_samples = data.shape[1]
        
        # Ensure float type for processing
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        else:
            data = data.copy()  # Don't modify input
        
        # 1. Baseline correction
        if self.config.baseline_correction and self._baseline_buffer is not None:
            data = self._apply_baseline_correction(data)
        
        # 2. Spatial reference
        data = self._apply_reference(data)
        
        # 3. Frequency filtering
        for filt in self._filters:
            data = filt.process(data)
        
        # 4. Artifact detection
        artifact_mask, _ = self._artifact_detector.detect(data)
        self._artifacts_detected += np.sum(artifact_mask)
        
        # 5. Handle artifacts (optional rejection/interpolation)
        if self.config.artifact_config.reject_artifacts:
            data = self._handle_artifacts(data, artifact_mask)
        
        # 6. Normalization
        if self.config.normalize and self._norm_buffer is not None:
            data = self._apply_normalization(data)
        
        self._samples_processed += n_samples
        
        if return_artifact_mask:
            return data, artifact_mask
        return data
    
    def _apply_baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Apply baseline correction using running window.
        
        Subtracts the mean of a sliding window to remove slow drifts.
        
        Args:
            data: Input data
            
        Returns:
            Baseline-corrected data
        """
        n_samples = data.shape[1]
        buffer_size = self._baseline_buffer.shape[1]
        
        # Add new samples to buffer
        for i in range(n_samples):
            self._baseline_buffer[:, self._baseline_buffer_idx] = data[:, i]
            self._baseline_buffer_idx = (self._baseline_buffer_idx + 1) % buffer_size
        
        # Compute baseline (mean of buffer)
        baseline = np.mean(self._baseline_buffer, axis=1, keepdims=True)
        
        # Subtract baseline
        return data - baseline
    
    def _apply_reference(self, data: np.ndarray) -> np.ndarray:
        """
        Apply spatial reference scheme.
        
        Args:
            data: Input data
            
        Returns:
            Re-referenced data
        """
        if self.config.reference == ReferenceType.NONE:
            return data
        
        elif self.config.reference == ReferenceType.COMMON_AVERAGE:
            # CAR: subtract mean of all channels
            car = np.mean(data, axis=0, keepdims=True)
            return data - car
        
        elif self.config.reference == ReferenceType.LINKED_MASTOID:
            if self.config.reference_channels is None:
                raise ValueError("Linked mastoid reference requires reference_channels")
            # Average of reference channels
            ref = np.mean(data[self.config.reference_channels, :], axis=0, keepdims=True)
            return data - ref
        
        elif self.config.reference == ReferenceType.BIPOLAR:
            # Difference between adjacent channels
            return np.diff(data, axis=0)
        
        elif self.config.reference == ReferenceType.LAPLACIAN:
            # Surface Laplacian - requires electrode positions
            # Simplified version: difference from neighbors
            warnings.warn("Laplacian reference requires electrode positions; using approximation")
            laplacian = np.zeros_like(data)
            for ch in range(1, data.shape[0] - 1):
                laplacian[ch] = data[ch] - 0.5 * (data[ch-1] + data[ch+1])
            # Edge channels: simple CAR
            laplacian[0] = data[0] - np.mean(data, axis=0)
            laplacian[-1] = data[-1] - np.mean(data, axis=0)
            return laplacian
        
        return data
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """
        Apply running z-score normalization.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data (z-scores)
        """
        n_samples = data.shape[1]
        buffer_size = self._norm_buffer.shape[1]
        
        # Add new samples to buffer
        for i in range(n_samples):
            self._norm_buffer[:, self._norm_buffer_idx] = data[:, i]
            self._norm_buffer_idx = (self._norm_buffer_idx + 1) % buffer_size
        
        # Compute statistics from buffer
        mean = np.mean(self._norm_buffer, axis=1, keepdims=True)
        std = np.std(self._norm_buffer, axis=1, keepdims=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-10)
        
        # Z-score normalize
        return (data - mean) / std
    
    def _handle_artifacts(
        self,
        data: np.ndarray,
        artifact_mask: np.ndarray
    ) -> np.ndarray:
        """
        Handle detected artifacts by interpolation or zeroing.
        
        Args:
            data: Input data
            artifact_mask: Boolean mask of artifacts
            
        Returns:
            Data with artifacts handled
        """
        # Simple approach: linear interpolation across artifacts
        for ch in range(data.shape[0]):
            if not np.any(artifact_mask[ch]):
                continue
            
            # Find clean samples
            clean_idx = np.where(~artifact_mask[ch])[0]
            artifact_idx = np.where(artifact_mask[ch])[0]
            
            if len(clean_idx) < 2:
                # Can't interpolate, zero out
                data[ch, artifact_mask[ch]] = 0
            else:
                # Linear interpolation
                data[ch, artifact_idx] = np.interp(
                    artifact_idx,
                    clean_idx,
                    data[ch, clean_idx]
                )
        
        return data
    
    def reset(self) -> None:
        """
        Reset all internal states.
        
        Call this when starting a new session or after a break
        to clear filter states and running statistics.
        """
        for filt in self._filters:
            filt.reset()
        
        self._artifact_detector.reset()
        
        if self._norm_buffer is not None:
            self._norm_buffer.fill(0)
            self._norm_buffer_idx = 0
        
        if self._baseline_buffer is not None:
            self._baseline_buffer.fill(0)
            self._baseline_buffer_idx = 0
        
        self._samples_processed = 0
        self._artifacts_detected = 0
        
        logger.info("Preprocessor state reset")
    
    @property
    def samples_processed(self) -> int:
        """Total samples processed since initialization or reset."""
        return self._samples_processed
    
    @property
    def artifact_rate(self) -> float:
        """Percentage of samples marked as artifacts."""
        if self._samples_processed == 0:
            return 0.0
        total_possible = self._samples_processed * self.config.n_channels
        return 100.0 * self._artifacts_detected / total_possible
    
    def get_filter_response(
        self,
        n_points: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute combined frequency response of all filters.
        
        Args:
            n_points: Number of frequency points
            
        Returns:
            Tuple of (frequencies_hz, magnitude_db)
        """
        # Start with unity response
        H_combined = np.ones(n_points, dtype=complex)
        
        for filt in self._filters:
            w, H = signal.sosfreqz(filt._sos, worN=n_points)
            H_combined *= H
        
        # Convert to Hz and dB
        freqs = w * self.config.sampling_rate / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(H_combined) + 1e-10)
        
        return freqs, magnitude_db
