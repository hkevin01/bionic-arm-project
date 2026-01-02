"""
EEG Feature Extraction Module
==============================

Extracts discriminative features from preprocessed EEG signals for
motor imagery classification. Implements multiple feature extraction
methods optimized for BCI applications.

Key Methods:
    1. Common Spatial Patterns (CSP): Maximizes variance ratio between classes
    2. Band Power: Power in frequency bands (mu, beta)
    3. Time-Frequency: Wavelet or STFT-based features
    4. Connectivity: Inter-channel coherence and phase relationships

Mathematical Background:
    
    Common Spatial Patterns (CSP):
        Finds spatial filters W that maximize:
        J(w) = w^T C1 w / w^T C2 w
        
        Where C1, C2 are covariance matrices of two classes.
        Solution: generalized eigenvalue problem C1 W = λ C2 W
        
    Band Power:
        P_band = (1/N) Σ |X(f)|^2 for f in [f_low, f_high]
        
        Log band power: log(P_band) provides better distribution for classifiers

Performance Targets:
    - Feature extraction latency: <10ms for 1-second window
    - CSP training: <1s for 100 trials
    - Memory efficient for embedded deployment

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from scipy import signal
from scipy.linalg import eigh, sqrtm, inv
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations and Constants
# =============================================================================

class FeatureMethod(Enum):
    """Available feature extraction methods."""
    BAND_POWER = auto()
    CSP = auto()
    LOG_VAR = auto()
    TIME_DOMAIN = auto()
    WAVELET = auto()
    COMBINED = auto()


# Standard frequency bands for motor imagery
MOTOR_IMAGERY_BANDS = {
    'mu': (8.0, 12.0),        # Mu rhythm - sensorimotor
    'low_beta': (12.0, 20.0),  # Low beta
    'high_beta': (20.0, 30.0), # High beta
    'beta': (12.0, 30.0),      # Combined beta
}


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class BandPowerConfig:
    """
    Configuration for band power feature extraction.
    
    Attributes:
        bands: Dictionary of band names to (low, high) frequency tuples
        sampling_rate: Signal sampling rate in Hz
        window_seconds: Analysis window duration
        use_log: Whether to use log band power (recommended)
        use_relative: Whether to compute relative band power
        welch_nperseg: Segment length for Welch's method (None = window size)
        welch_noverlap: Overlap for Welch's method (None = 50%)
    """
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: MOTOR_IMAGERY_BANDS.copy()
    )
    sampling_rate: int = 250
    window_seconds: float = 1.0
    use_log: bool = True
    use_relative: bool = False
    welch_nperseg: Optional[int] = None
    welch_noverlap: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        nyquist = self.sampling_rate / 2.0
        for name, (low, high) in self.bands.items():
            if low >= high:
                raise ValueError(f"Band '{name}': low ({low}) must be < high ({high})")
            if high > nyquist:
                raise ValueError(
                    f"Band '{name}' high freq ({high}) exceeds Nyquist ({nyquist})"
                )
    
    @property
    def n_features_per_channel(self) -> int:
        """Number of features extracted per channel."""
        return len(self.bands)


@dataclass
class CSPConfig:
    """
    Configuration for Common Spatial Patterns.
    
    Attributes:
        n_components: Number of CSP components to keep (pairs from each end)
        reg: Regularization parameter (0-1) for covariance shrinkage
        log: Whether to use log variance of CSP-filtered signals
        norm_trace: Whether to normalize covariance by trace
        cov_method: Covariance estimation method ('empirical', 'shrunk', 'oas')
    """
    n_components: int = 4
    reg: float = 0.0
    log: bool = True
    norm_trace: bool = True
    cov_method: str = 'empirical'
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if not 0 <= self.reg <= 1:
            raise ValueError(f"reg must be in [0, 1], got {self.reg}")
        if self.cov_method not in ('empirical', 'shrunk', 'oas'):
            raise ValueError(f"Unknown cov_method: {self.cov_method}")


@dataclass
class FeatureConfig:
    """
    Complete feature extraction configuration.
    
    Attributes:
        method: Primary feature extraction method
        sampling_rate: Signal sampling rate in Hz
        n_channels: Number of EEG channels
        window_seconds: Analysis window duration
        band_power_config: Configuration for band power (if used)
        csp_config: Configuration for CSP (if used)
        combine_methods: List of methods to combine (if method=COMBINED)
    """
    method: FeatureMethod = FeatureMethod.CSP
    sampling_rate: int = 250
    n_channels: int = 32
    window_seconds: float = 1.0
    band_power_config: Optional[BandPowerConfig] = None
    csp_config: Optional[CSPConfig] = None
    combine_methods: List[FeatureMethod] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Set up default configs if not provided."""
        if self.band_power_config is None:
            self.band_power_config = BandPowerConfig(
                sampling_rate=self.sampling_rate,
                window_seconds=self.window_seconds
            )
        if self.csp_config is None:
            self.csp_config = CSPConfig()


# =============================================================================
# Band Power Extraction
# =============================================================================

class BandPowerExtractor:
    """
    Extracts power in specified frequency bands.
    
    Uses Welch's method for robust spectral estimation with 50% overlap.
    Optionally computes log and/or relative band power.
    
    Mathematical Details:
        Welch's method divides signal into overlapping segments,
        computes periodogram of each, and averages:
        
        P_welch = (1/K) Σ_k |FFT(x_k * w)|^2
        
        Where w is a window function (Hann) and K is number of segments.
        
        Band power is integral of PSD over band:
        P_band = ∫_{f_low}^{f_high} PSD(f) df
        
        Log band power: log(P_band) is approximately Gaussian,
        better for linear classifiers.
    
    Example:
        >>> config = BandPowerConfig(bands={'mu': (8, 12), 'beta': (12, 30)})
        >>> extractor = BandPowerExtractor(config)
        >>> features = extractor.extract(eeg_data)  # Shape: (n_channels * n_bands,)
    """
    
    def __init__(self, config: BandPowerConfig) -> None:
        """
        Initialize band power extractor.
        
        Args:
            config: Band power configuration
        """
        self.config = config
        self._band_names = list(config.bands.keys())
        
        # Compute Welch parameters
        window_samples = int(config.window_seconds * config.sampling_rate)
        self._nperseg = config.welch_nperseg or min(window_samples, 256)
        self._noverlap = config.welch_noverlap or self._nperseg // 2
        
        logger.debug(
            f"BandPowerExtractor: {len(self._band_names)} bands, "
            f"nperseg={self._nperseg}, noverlap={self._noverlap}"
        )
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract band power features from EEG data.
        
        Args:
            data: EEG data, shape (n_channels, n_samples)
            
        Returns:
            Feature vector, shape (n_channels * n_bands,)
        """
        n_channels, n_samples = data.shape
        n_bands = len(self.config.bands)
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(
            data,
            fs=self.config.sampling_rate,
            nperseg=self._nperseg,
            noverlap=self._noverlap,
            axis=1
        )
        
        # Extract power in each band
        features = np.zeros((n_channels, n_bands))
        
        for band_idx, (band_name, (f_low, f_high)) in enumerate(self.config.bands.items()):
            # Find frequency indices for this band
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            
            if not np.any(band_mask):
                warnings.warn(f"No frequencies in band {band_name} ({f_low}-{f_high} Hz)")
                continue
            
            # Integrate PSD over band (trapezoidal rule)
            band_psd = psd[:, band_mask]
            band_freqs = freqs[band_mask]
            
            # Compute band power using trapezoidal integration
            band_power = np.trapz(band_psd, band_freqs, axis=1)
            features[:, band_idx] = band_power
        
        # Compute relative power if requested
        if self.config.use_relative:
            total_power = np.sum(features, axis=1, keepdims=True)
            features = features / (total_power + 1e-10)
        
        # Apply log transform if requested
        if self.config.use_log:
            features = np.log(features + 1e-10)
        
        # Flatten to 1D feature vector
        return features.flatten()
    
    def extract_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from batch of trials.
        
        Args:
            data: EEG data, shape (n_trials, n_channels, n_samples)
            
        Returns:
            Features, shape (n_trials, n_channels * n_bands)
        """
        n_trials = data.shape[0]
        n_features = data.shape[1] * len(self.config.bands)
        
        features = np.zeros((n_trials, n_features))
        for trial_idx in range(n_trials):
            features[trial_idx] = self.extract(data[trial_idx])
        
        return features
    
    @property
    def n_features(self) -> int:
        """Total number of features extracted."""
        # Assumes we know n_channels, but this is typically set at extraction time
        return len(self.config.bands)
    
    @property
    def feature_names(self) -> List[str]:
        """Get names of features (without channel prefix)."""
        return list(self.config.bands.keys())


# =============================================================================
# Common Spatial Patterns (CSP)
# =============================================================================

class CSP:
    """
    Common Spatial Patterns for motor imagery BCI.
    
    CSP finds spatial filters that maximize variance for one class
    while minimizing it for another. This is particularly effective
    for motor imagery, where different movements cause localized
    changes in power.
    
    Mathematical Details:
        Given covariance matrices C1 (class 1) and C2 (class 2),
        CSP solves the generalized eigenvalue problem:
        
            C1 W = λ (C1 + C2) W
        
        The eigenvectors W are the spatial filters.
        Features from first m and last m filters are most discriminative.
        
        For regularized CSP:
            C_reg = (1 - reg) * C + reg * (trace(C) / n) * I
        
        This helps when training data is limited.
    
    Algorithm:
        1. Compute class covariance matrices
        2. Apply regularization if specified
        3. Compute composite covariance C_c = C1 + C2
        4. Whiten using C_c: P = Λ^(-1/2) U^T
        5. Compute SVD of whitened C1: S1 = P C1 P^T
        6. Spatial filters: W = B^T P
        
    Example:
        >>> csp = CSP(CSPConfig(n_components=4))
        >>> csp.fit(X_train, y_train)  # X: (n_trials, n_channels, n_samples)
        >>> features = csp.transform(X_test)  # Shape: (n_trials, 2*n_components)
    """
    
    def __init__(self, config: CSPConfig) -> None:
        """
        Initialize CSP.
        
        Args:
            config: CSP configuration
        """
        self.config = config
        
        # Fitted parameters
        self._filters: Optional[np.ndarray] = None  # Spatial filters W
        self._patterns: Optional[np.ndarray] = None  # Patterns (for interpretation)
        self._eigenvalues: Optional[np.ndarray] = None
        self._n_channels: Optional[int] = None
        self._is_fitted = False
        
        logger.debug(
            f"CSP initialized: n_components={config.n_components}, "
            f"reg={config.reg}, log={config.log}"
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSP':
        """
        Fit CSP filters to training data.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Class labels, shape (n_trials,), must be binary (0/1 or -1/1)
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If input shapes are invalid or not binary classification
        """
        # Validate input
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (trials, channels, samples), got {X.ndim}D")
        
        n_trials, n_channels, n_samples = X.shape
        
        if len(y) != n_trials:
            raise ValueError(f"y length ({len(y)}) must match n_trials ({n_trials})")
        
        # Convert labels to binary
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")
        
        y_binary = (y == classes[1]).astype(int)
        
        # Validate n_components
        max_components = n_channels // 2
        if self.config.n_components > max_components:
            warnings.warn(
                f"n_components ({self.config.n_components}) > n_channels/2 ({max_components}). "
                f"Reducing to {max_components}"
            )
            n_components = max_components
        else:
            n_components = self.config.n_components
        
        # Compute class covariance matrices
        X_class0 = X[y_binary == 0]
        X_class1 = X[y_binary == 1]
        
        C0 = self._compute_covariance(X_class0)
        C1 = self._compute_covariance(X_class1)
        
        # Apply regularization
        if self.config.reg > 0:
            C0 = self._regularize_covariance(C0)
            C1 = self._regularize_covariance(C1)
        
        # Normalize by trace if requested
        if self.config.norm_trace:
            C0 /= np.trace(C0)
            C1 /= np.trace(C1)
        
        # Compute composite covariance
        C_composite = C0 + C1
        
        # Solve generalized eigenvalue problem
        # C0 W = λ C_composite W
        # Equivalent to: C_composite^(-1) C0 W = λ W
        try:
            eigenvalues, eigenvectors = eigh(C0, C_composite)
        except np.linalg.LinAlgError as e:
            logger.error(f"Eigenvalue decomposition failed: {e}")
            raise ValueError("CSP fitting failed. Try increasing regularization.") from e
        
        # Sort by eigenvalue (descending for class 0, ascending for class 1)
        # Take first n_components and last n_components
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Select components: first n and last n
        component_idx = np.concatenate([
            np.arange(n_components),
            np.arange(n_channels - n_components, n_channels)
        ])
        
        self._filters = eigenvectors[:, component_idx]
        self._eigenvalues = eigenvalues[component_idx]
        self._n_channels = n_channels
        
        # Compute patterns for interpretation
        # Patterns = C * Filters (filters are not interpretable directly)
        C_avg = (C0 + C1) / 2
        self._patterns = C_avg @ self._filters
        
        self._is_fitted = True
        
        logger.info(
            f"CSP fitted: {n_components * 2} components from {n_channels} channels, "
            f"eigenvalue ratio: {eigenvalues[0] / eigenvalues[-1]:.2f}"
        )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP filters and extract features.
        
        Args:
            X: EEG data, shape (n_trials, n_channels, n_samples) or
               (n_channels, n_samples) for single trial
            
        Returns:
            Features, shape (n_trials, 2*n_components) or (2*n_components,)
        """
        if not self._is_fitted:
            raise RuntimeError("CSP must be fitted before transform")
        
        # Handle single trial
        single_trial = X.ndim == 2
        if single_trial:
            X = X[np.newaxis, :, :]
        
        n_trials, n_channels, n_samples = X.shape
        
        if n_channels != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {n_channels}"
            )
        
        n_features = self._filters.shape[1]
        features = np.zeros((n_trials, n_features))
        
        for trial_idx in range(n_trials):
            # Apply spatial filters: (n_components, n_samples)
            filtered = self._filters.T @ X[trial_idx]
            
            # Compute variance of each filtered signal
            var = np.var(filtered, axis=1)
            
            # Apply log transform if requested (recommended)
            if self.config.log:
                features[trial_idx] = np.log(var + 1e-10)
            else:
                features[trial_idx] = var
        
        if single_trial:
            return features[0]
        return features
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP and transform in one step.
        
        Args:
            X: Training data
            y: Class labels
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute average covariance matrix from trials.
        
        Args:
            X: Data, shape (n_trials, n_channels, n_samples)
            
        Returns:
            Covariance matrix, shape (n_channels, n_channels)
        """
        n_trials = X.shape[0]
        
        if self.config.cov_method == 'empirical':
            # Simple empirical covariance
            covs = np.zeros((n_trials, X.shape[1], X.shape[1]))
            for trial in range(n_trials):
                covs[trial] = np.cov(X[trial])
            return np.mean(covs, axis=0)
        
        elif self.config.cov_method == 'shrunk':
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf
            covs = []
            for trial in range(n_trials):
                lw = LedoitWolf().fit(X[trial].T)
                covs.append(lw.covariance_)
            return np.mean(covs, axis=0)
        
        elif self.config.cov_method == 'oas':
            # Oracle Approximating Shrinkage
            from sklearn.covariance import OAS
            covs = []
            for trial in range(n_trials):
                oas = OAS().fit(X[trial].T)
                covs.append(oas.covariance_)
            return np.mean(covs, axis=0)
        
        else:
            raise ValueError(f"Unknown cov_method: {self.config.cov_method}")
    
    def _regularize_covariance(self, C: np.ndarray) -> np.ndarray:
        """
        Apply shrinkage regularization to covariance matrix.
        
        C_reg = (1 - reg) * C + reg * (trace(C) / n) * I
        
        Args:
            C: Covariance matrix
            
        Returns:
            Regularized covariance matrix
        """
        n = C.shape[0]
        reg = self.config.reg
        
        # Shrinkage toward scaled identity
        shrinkage_target = (np.trace(C) / n) * np.eye(n)
        
        return (1 - reg) * C + reg * shrinkage_target
    
    @property
    def filters(self) -> Optional[np.ndarray]:
        """Spatial filters (columns are filters)."""
        return self._filters
    
    @property
    def patterns(self) -> Optional[np.ndarray]:
        """Spatial patterns (interpretable, unlike filters)."""
        return self._patterns
    
    @property
    def n_features(self) -> int:
        """Number of features produced."""
        if self._filters is None:
            return 2 * self.config.n_components
        return self._filters.shape[1]
    
    def save(self, filepath: str) -> None:
        """Save fitted CSP to file."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted CSP")
        
        np.savez(
            filepath,
            filters=self._filters,
            patterns=self._patterns,
            eigenvalues=self._eigenvalues,
            n_channels=self._n_channels,
            config_n_components=self.config.n_components,
            config_reg=self.config.reg,
            config_log=self.config.log
        )
        logger.info(f"CSP saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CSP':
        """Load fitted CSP from file."""
        data = np.load(filepath)
        
        config = CSPConfig(
            n_components=int(data['config_n_components']),
            reg=float(data['config_reg']),
            log=bool(data['config_log'])
        )
        
        csp = cls(config)
        csp._filters = data['filters']
        csp._patterns = data['patterns']
        csp._eigenvalues = data['eigenvalues']
        csp._n_channels = int(data['n_channels'])
        csp._is_fitted = True
        
        logger.info(f"CSP loaded from {filepath}")
        return csp


# =============================================================================
# Main Feature Extractor Class
# =============================================================================

class FeatureExtractor:
    """
    Unified feature extraction interface.
    
    Combines multiple feature extraction methods and provides a
    consistent interface for the BCI pipeline.
    
    Supported Methods:
        - BAND_POWER: Power in frequency bands (fast, no training)
        - CSP: Common Spatial Patterns (requires training, highly effective)
        - LOG_VAR: Log variance per channel (simple baseline)
        - TIME_DOMAIN: Statistical features (mean, std, etc.)
        - COMBINED: Multiple methods concatenated
    
    Example:
        >>> config = FeatureConfig(method=FeatureMethod.CSP)
        >>> extractor = FeatureExtractor(config)
        >>> extractor.fit(X_train, y_train)
        >>> features = extractor.transform(X_test)
    """
    
    def __init__(self, config: FeatureConfig) -> None:
        """
        Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config
        
        # Initialize extractors based on method
        self._band_power: Optional[BandPowerExtractor] = None
        self._csp: Optional[CSP] = None
        
        if config.method in (FeatureMethod.BAND_POWER, FeatureMethod.COMBINED):
            self._band_power = BandPowerExtractor(config.band_power_config)
        
        if config.method in (FeatureMethod.CSP, FeatureMethod.COMBINED):
            self._csp = CSP(config.csp_config)
        
        self._is_fitted = False
        self._feature_names: List[str] = []
        
        logger.info(f"FeatureExtractor initialized: method={config.method.name}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureExtractor':
        """
        Fit feature extractor to training data.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Class labels (required for CSP)
            
        Returns:
            self (for method chaining)
        """
        # Fit CSP if used
        if self._csp is not None:
            if y is None:
                raise ValueError("CSP requires class labels (y)")
            self._csp.fit(X, y)
        
        # Build feature names
        self._build_feature_names(X.shape[1])
        
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from data.
        
        Args:
            X: EEG data, shape (n_trials, n_channels, n_samples) or
               (n_channels, n_samples) for single trial
            
        Returns:
            Features, shape (n_trials, n_features) or (n_features,)
        """
        # Handle single trial
        single_trial = X.ndim == 2
        if single_trial:
            X = X[np.newaxis, :, :]
        
        n_trials = X.shape[0]
        features_list = []
        
        # Extract features based on method
        if self.config.method == FeatureMethod.BAND_POWER:
            features = self._band_power.extract_batch(X)
            features_list.append(features)
        
        elif self.config.method == FeatureMethod.CSP:
            if not self._csp._is_fitted:
                raise RuntimeError("CSP must be fitted before transform")
            features = self._csp.transform(X)
            features_list.append(features)
        
        elif self.config.method == FeatureMethod.LOG_VAR:
            # Log variance per channel
            var = np.var(X, axis=2)
            features = np.log(var + 1e-10)
            features_list.append(features)
        
        elif self.config.method == FeatureMethod.TIME_DOMAIN:
            features = self._extract_time_domain(X)
            features_list.append(features)
        
        elif self.config.method == FeatureMethod.COMBINED:
            # Combine multiple methods
            if self._band_power is not None:
                features_list.append(self._band_power.extract_batch(X))
            if self._csp is not None and self._csp._is_fitted:
                features_list.append(self._csp.transform(X))
            
            # Add log variance as additional feature
            var = np.var(X, axis=2)
            features_list.append(np.log(var + 1e-10))
        
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=1)
        
        if single_trial:
            return all_features[0]
        return all_features
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _extract_time_domain(self, X: np.ndarray) -> np.ndarray:
        """
        Extract time-domain statistical features.
        
        Features per channel:
            - Mean
            - Standard deviation
            - Skewness
            - Kurtosis
            - Peak-to-peak amplitude
            - Zero crossings (normalized)
        
        Args:
            X: Data, shape (n_trials, n_channels, n_samples)
            
        Returns:
            Features, shape (n_trials, n_channels * 6)
        """
        from scipy.stats import skew, kurtosis
        
        n_trials, n_channels, n_samples = X.shape
        n_features_per_channel = 6
        
        features = np.zeros((n_trials, n_channels * n_features_per_channel))
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal_ch = X[trial_idx, ch_idx]
                base_idx = ch_idx * n_features_per_channel
                
                features[trial_idx, base_idx] = np.mean(signal_ch)
                features[trial_idx, base_idx + 1] = np.std(signal_ch)
                features[trial_idx, base_idx + 2] = skew(signal_ch)
                features[trial_idx, base_idx + 3] = kurtosis(signal_ch)
                features[trial_idx, base_idx + 4] = np.ptp(signal_ch)
                
                # Zero crossings
                zero_crossings = np.sum(np.diff(np.sign(signal_ch)) != 0)
                features[trial_idx, base_idx + 5] = zero_crossings / n_samples
        
        return features
    
    def _build_feature_names(self, n_channels: int) -> None:
        """Build list of feature names for interpretability."""
        self._feature_names = []
        
        if self.config.method == FeatureMethod.BAND_POWER:
            for ch in range(n_channels):
                for band in self._band_power.feature_names:
                    self._feature_names.append(f"ch{ch}_{band}_power")
        
        elif self.config.method == FeatureMethod.CSP:
            for i in range(self._csp.n_features):
                self._feature_names.append(f"csp_{i}")
        
        elif self.config.method == FeatureMethod.LOG_VAR:
            for ch in range(n_channels):
                self._feature_names.append(f"ch{ch}_log_var")
        
        elif self.config.method == FeatureMethod.TIME_DOMAIN:
            stats = ['mean', 'std', 'skew', 'kurtosis', 'ptp', 'zc']
            for ch in range(n_channels):
                for stat in stats:
                    self._feature_names.append(f"ch{ch}_{stat}")
    
    @property
    def feature_names(self) -> List[str]:
        """Get names of all features."""
        return self._feature_names
    
    @property
    def n_features(self) -> int:
        """Number of features produced."""
        return len(self._feature_names) if self._feature_names else 0
    
    def save(self, filepath: str) -> None:
        """Save fitted extractor to file."""
        import pickle
        
        state = {
            'config': self.config,
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
        }
        
        # Save CSP separately if fitted
        if self._csp is not None and self._csp._is_fitted:
            state['csp_filters'] = self._csp._filters
            state['csp_patterns'] = self._csp._patterns
            state['csp_eigenvalues'] = self._csp._eigenvalues
            state['csp_n_channels'] = self._csp._n_channels
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"FeatureExtractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """Load fitted extractor from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        extractor = cls(state['config'])
        extractor._is_fitted = state['is_fitted']
        extractor._feature_names = state['feature_names']
        
        # Restore CSP if saved
        if 'csp_filters' in state and extractor._csp is not None:
            extractor._csp._filters = state['csp_filters']
            extractor._csp._patterns = state['csp_patterns']
            extractor._csp._eigenvalues = state['csp_eigenvalues']
            extractor._csp._n_channels = state['csp_n_channels']
            extractor._csp._is_fitted = True
        
        logger.info(f"FeatureExtractor loaded from {filepath}")
        return extractor
