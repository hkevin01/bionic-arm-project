"""
Brain-Computer Interface (BCI) Module
======================================

This module provides complete BCI functionality for the bionic arm project,
including EEG signal acquisition, preprocessing, feature extraction,
and neural decoding to produce continuous motor commands.

Pipeline Overview:

    EEG Acquisition → Preprocessing → Feature Extraction → Decoding → Velocity
    
    1. Acquisition: Capture multi-channel EEG at 250+ Hz
    2. Preprocessing: Filter (0.5-40 Hz), artifact rejection, normalization
    3. Features: CSP, band power, time-domain features
    4. Decoding: Neural network + Kalman filter for smooth commands

Key Classes:
    - BCIPipeline: Complete pipeline orchestration
    - BaseAcquisition: Abstract acquisition interface
    - SimulatedAcquisition: Synthetic EEG for testing
    - Preprocessor: Signal conditioning
    - FeatureExtractor: CSP and band power extraction
    - ContinuousDecoder: Neural decoder with Kalman smoothing

Quick Start:
    >>> from bci import BCIPipeline, PipelineConfig
    >>> 
    >>> config = PipelineConfig()
    >>> pipeline = BCIPipeline(config)
    >>> pipeline.start()
    >>> 
    >>> while running:
    ...     velocity, confidence = pipeline.get_output()
    ...     send_to_arm(velocity)
    >>> 
    >>> pipeline.stop()

Performance Targets:
    - End-to-end latency: <50ms
    - Update rate: 30-60 Hz
    - Motor imagery accuracy: >80%

Author: Bionic Arm Project Team
License: MIT
"""

# Acquisition module
from .acquisition import (
    AcquisitionConfig,
    EEGSample,
    BaseAcquisition,
    SimulatedAcquisition,
    DeviceState,
)

# Preprocessing module
from .preprocessing import (
    FilterConfig,
    ArtifactConfig,
    PreprocessorConfig,
    DigitalFilter,
    ArtifactDetector,
    Preprocessor,
)

# Feature extraction module
from .features import (
    BandPowerConfig,
    CSPConfig,
    FeatureConfig,
    FeatureType,
    BandPowerExtractor,
    CSP,
    FeatureExtractor,
)

# Decoder module
from .decoder import (
    DecoderConfig,
    EEGNetConfig,
    KalmanFilter,
    ContinuousDecoder,
    ClassifierDecoder,
)

# Pipeline module
from .pipeline import (
    PipelineConfig,
    PipelineState,
    PipelineMetrics,
    BCIPipeline,
    create_default_pipeline,
    create_pipeline_from_config,
)

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Acquisition
    "AcquisitionConfig",
    "EEGSample",
    "BaseAcquisition",
    "SimulatedAcquisition",
    "DeviceState",
    # Preprocessing
    "FilterConfig",
    "ArtifactConfig",
    "PreprocessorConfig",
    "DigitalFilter",
    "ArtifactDetector",
    "Preprocessor",
    # Features
    "BandPowerConfig",
    "CSPConfig",
    "FeatureConfig",
    "FeatureType",
    "BandPowerExtractor",
    "CSP",
    "FeatureExtractor",
    # Decoder
    "DecoderConfig",
    "EEGNetConfig",
    "KalmanFilter",
    "ContinuousDecoder",
    "ClassifierDecoder",
    # Pipeline
    "PipelineConfig",
    "PipelineState",
    "PipelineMetrics",
    "BCIPipeline",
    "create_default_pipeline",
    "create_pipeline_from_config",
]
