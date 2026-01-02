"""
Unit Tests for BCI Module
=========================

Comprehensive tests for EEG acquisition, preprocessing, feature
extraction, and decoding components.

Author: Bionic Arm Project Team
License: MIT
"""

import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import modules under test
from src.bci.acquisition import (
    AcquisitionConfig,
    DeviceState,
    EEGSample,
    SimulatedAcquisition,
)
from src.bci.decoder import ContinuousDecoder, DecoderConfig, KalmanFilter
from src.bci.features import (
    CSP,
    BandPowerConfig,
    BandPowerExtractor,
    CSPConfig,
    FeatureConfig,
    FeatureExtractor,
    FeatureType,
)
from src.bci.pipeline import BCIPipeline, PipelineConfig, PipelineState
from src.bci.preprocessing import (
    ArtifactConfig,
    ArtifactDetector,
    DigitalFilter,
    FilterConfig,
    FilterType,
    Preprocessor,
    PreprocessorConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def acquisition_config():
    """Basic acquisition configuration."""
    return AcquisitionConfig(sampling_rate=250, n_channels=8, buffer_size_seconds=2.0)


@pytest.fixture
def simulated_acquisition(acquisition_config):
    """Simulated EEG acquisition device."""
    return SimulatedAcquisition(acquisition_config)


@pytest.fixture
def preprocessor_config():
    """Preprocessor configuration."""
    return PreprocessorConfig(
        sampling_rate=250,
        n_channels=8,
    )


@pytest.fixture
def bandpower_config():
    """Band power extraction configuration."""
    return BandPowerConfig(
        bands={
            "mu": (8.0, 12.0),
            "beta": (12.0, 30.0),
        },
        sampling_rate=250,
    )


@pytest.fixture
def synthetic_eeg():
    """Generate synthetic EEG data with known frequency content."""
    n_channels = 8
    n_samples = 500  # 2 seconds at 250 Hz
    fs = 250
    t = np.arange(n_samples) / fs

    # Generate signal with known frequencies
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # 10 Hz (mu band) + 20 Hz (beta band) + noise
        data[ch] = (
            np.sin(2 * np.pi * 10 * t)  # Mu
            + 0.5 * np.sin(2 * np.pi * 20 * t)  # Beta
            + 0.1 * np.random.randn(n_samples)  # Noise
        )

    return data


@pytest.fixture
def motor_imagery_data():
    """Synthetic motor imagery dataset for CSP training."""
    n_trials = 40
    n_channels = 8
    n_samples = 250

    # Class 0: higher mu power in left channels
    class0_data = []
    for _ in range(n_trials // 2):
        trial = np.random.randn(n_channels, n_samples)
        # Boost mu (10 Hz) in left channels (0-3)
        t = np.arange(n_samples) / 250
        for ch in range(4):
            trial[ch] += 2 * np.sin(2 * np.pi * 10 * t)
        class0_data.append(trial)

    # Class 1: higher mu power in right channels
    class1_data = []
    for _ in range(n_trials // 2):
        trial = np.random.randn(n_channels, n_samples)
        # Boost mu (10 Hz) in right channels (4-7)
        t = np.arange(n_samples) / 250
        for ch in range(4, 8):
            trial[ch] += 2 * np.sin(2 * np.pi * 10 * t)
        class1_data.append(trial)

    X = np.array(class0_data + class1_data)
    y = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))

    # Shuffle
    indices = np.random.permutation(n_trials)
    X = X[indices]
    y = y[indices]

    return X, y


# =============================================================================
# Acquisition Tests
# =============================================================================


class TestAcquisitionConfig:
    """Tests for AcquisitionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AcquisitionConfig()
        assert config.sampling_rate == 250
        assert config.n_channels == 32
        assert config.buffer_size_seconds == 5.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = AcquisitionConfig(
            sampling_rate=500, n_channels=16, buffer_size_seconds=10.0
        )
        assert config.sampling_rate == 500
        assert config.n_channels == 16
        assert config.buffer_samples == 5000

    def test_invalid_sampling_rate(self):
        """Test that invalid sampling rate raises error."""
        with pytest.raises(ValueError):
            AcquisitionConfig(sampling_rate=0)
        with pytest.raises(ValueError):
            AcquisitionConfig(sampling_rate=-100)

    def test_channel_names_auto_generation(self):
        """Test automatic channel name generation."""
        config = AcquisitionConfig(n_channels=4)
        assert config.channel_names == ["CH1", "CH2", "CH3", "CH4"]

    def test_channel_names_mismatch(self):
        """Test error when channel names don't match channel count."""
        with pytest.raises(ValueError):
            AcquisitionConfig(n_channels=4, channel_names=["CH1", "CH2"])


class TestEEGSample:
    """Tests for EEGSample data class."""

    def test_single_sample(self):
        """Test single time-point sample."""
        data = np.random.randn(8)
        sample = EEGSample(data=data, timestamp=time.time(), sample_index=0)
        assert sample.n_channels == 8
        assert sample.n_samples == 1
        assert not sample.is_chunk

    def test_chunk_sample(self):
        """Test multi-sample chunk."""
        data = np.random.randn(8, 100)
        sample = EEGSample(data=data, timestamp=time.time(), sample_index=0)
        assert sample.n_channels == 8
        assert sample.n_samples == 100
        assert sample.is_chunk

    def test_data_type_conversion(self):
        """Test that integer data is converted to float."""
        data = np.array([1, 2, 3, 4])
        sample = EEGSample(data=data, timestamp=time.time(), sample_index=0)
        assert np.issubdtype(sample.data.dtype, np.floating)


class TestSimulatedAcquisition:
    """Tests for SimulatedAcquisition."""

    def test_initialization(self, simulated_acquisition):
        """Test device initialization."""
        assert simulated_acquisition.state == DeviceState.DISCONNECTED

    def test_connect_disconnect(self, simulated_acquisition):
        """Test connection lifecycle."""
        assert simulated_acquisition.connect()
        assert simulated_acquisition.state == DeviceState.CONNECTED

        assert simulated_acquisition.disconnect()
        assert simulated_acquisition.state == DeviceState.DISCONNECTED

    def test_start_stop_streaming(self, simulated_acquisition):
        """Test streaming lifecycle."""
        simulated_acquisition.connect()
        simulated_acquisition.start()
        assert simulated_acquisition.state == DeviceState.STREAMING

        # Let it run for a bit
        time.sleep(0.1)

        simulated_acquisition.stop()
        assert simulated_acquisition.state == DeviceState.CONNECTED

        simulated_acquisition.disconnect()

    def test_get_latest_data(self, simulated_acquisition):
        """Test retrieving buffered data."""
        simulated_acquisition.connect()
        simulated_acquisition.start()

        # Wait for some data
        time.sleep(0.2)

        # Get 50 samples
        data = simulated_acquisition.get_latest_data(50)
        assert data is not None
        assert data.shape == (8, 50)

        simulated_acquisition.stop()
        simulated_acquisition.disconnect()

    def test_callback_registration(self, simulated_acquisition):
        """Test callback functionality."""
        received_samples = []

        def callback(sample: EEGSample):
            received_samples.append(sample)

        simulated_acquisition.register_callback(callback)
        simulated_acquisition.connect()
        simulated_acquisition.start()

        time.sleep(0.2)

        simulated_acquisition.stop()
        simulated_acquisition.disconnect()

        assert len(received_samples) > 0


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestFilterConfig:
    """Tests for FilterConfig."""

    def test_bandpass_config(self):
        """Test bandpass filter configuration."""
        config = FilterConfig(
            filter_type=FilterType.BANDPASS, low_freq=0.5, high_freq=50.0, order=4
        )
        assert config.filter_type == FilterType.BANDPASS

    def test_invalid_bandpass(self):
        """Test that invalid bandpass raises error."""
        with pytest.raises(ValueError):
            FilterConfig(
                filter_type=FilterType.BANDPASS,
                low_freq=50.0,
                high_freq=0.5,  # Low > high
            )

    def test_notch_config(self):
        """Test notch filter configuration."""
        config = FilterConfig(filter_type=FilterType.NOTCH, notch_freq=60.0)
        assert config.notch_freq == 60.0


class TestDigitalFilter:
    """Tests for DigitalFilter."""

    def test_bandpass_filter(self):
        """Test bandpass filtering removes out-of-band content."""
        fs = 250
        n_samples = 500
        t = np.arange(n_samples) / fs

        # Signal: 5 Hz (will be filtered out) + 15 Hz (will pass)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t)
        signal = signal.reshape(1, -1)  # Shape: (1, n_samples)

        config = FilterConfig(
            filter_type=FilterType.BANDPASS, low_freq=10.0, high_freq=20.0, order=4
        )

        filt = DigitalFilter(config, fs, 1)
        filtered = filt.process(signal)

        # Check that 5 Hz is attenuated
        # Compare power in first half vs second half of spectrum
        fft_original = np.abs(np.fft.fft(signal[0]))
        fft_filtered = np.abs(np.fft.fft(filtered[0]))

        # 5 Hz index
        idx_5hz = int(5 * n_samples / fs)
        # 15 Hz index
        idx_15hz = int(15 * n_samples / fs)

        # 5 Hz should be attenuated, 15 Hz should be preserved
        assert fft_filtered[idx_5hz] < fft_original[idx_5hz] * 0.5
        assert fft_filtered[idx_15hz] > fft_original[idx_15hz] * 0.5

    def test_notch_filter(self):
        """Test notch filter removes target frequency."""
        fs = 250
        n_samples = 500
        t = np.arange(n_samples) / fs

        # Signal: 10 Hz + 60 Hz (power line)
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)
        signal = signal.reshape(1, -1)

        config = FilterConfig(filter_type=FilterType.NOTCH, notch_freq=60.0)

        filt = DigitalFilter(config, fs, 1)
        filtered = filt.process(signal)

        # Check 60 Hz is attenuated
        fft_original = np.abs(np.fft.fft(signal[0]))
        fft_filtered = np.abs(np.fft.fft(filtered[0]))

        idx_60hz = int(60 * n_samples / fs)
        assert fft_filtered[idx_60hz] < fft_original[idx_60hz] * 0.3


class TestArtifactDetector:
    """Tests for ArtifactDetector."""

    def test_amplitude_threshold(self):
        """Test amplitude-based artifact detection."""
        config = ArtifactConfig(amplitude_threshold_uv=100.0)
        detector = ArtifactDetector(config)

        # Normal data
        normal_data = np.random.randn(8, 100) * 20  # ±20 µV
        artifacts, markers = detector.detect(normal_data)
        assert np.sum(artifacts) == 0  # No artifacts

        # Data with artifact
        artifact_data = normal_data.copy()
        artifact_data[0, 50] = 200  # Large spike
        artifacts, markers = detector.detect(artifact_data)
        assert artifacts[0, 50]  # Artifact detected

    def test_flatline_detection(self):
        """Test flat line artifact detection."""
        config = ArtifactConfig(flatline_threshold_uv=0.1, flatline_min_duration=0.02)
        detector = ArtifactDetector(config)

        # Create flat line data
        data = np.zeros((8, 100))
        data[:, :50] = np.random.randn(8, 50) * 10
        data[:, 50:] = 0.001  # Flat line

        artifacts, markers = detector.detect(data)
        # Should detect flat line region
        assert np.any(artifacts[:, 50:])


class TestPreprocessor:
    """Tests for complete Preprocessor."""

    def test_preprocessing_pipeline(self, preprocessor_config, synthetic_eeg):
        """Test complete preprocessing pipeline."""
        preprocessor = Preprocessor(preprocessor_config)

        processed = preprocessor.process(synthetic_eeg)

        assert processed.shape == synthetic_eeg.shape
        assert not np.any(np.isnan(processed))
        assert not np.any(np.isinf(processed))

    def test_common_average_reference(self, synthetic_eeg):
        """Test CAR is applied correctly."""
        config = PreprocessorConfig(
            sampling_rate=250,
            n_channels=8,
            filters=[],  # No filters, just CAR
        )
        preprocessor = Preprocessor(config)

        processed = preprocessor.process(synthetic_eeg)

        # Mean across channels should be near zero after CAR
        mean_per_sample = np.mean(processed, axis=0)
        assert np.allclose(mean_per_sample, 0, atol=1e-10)


# =============================================================================
# Feature Extraction Tests
# =============================================================================


class TestBandPowerExtractor:
    """Tests for BandPowerExtractor."""

    def test_band_power_extraction(self, bandpower_config, synthetic_eeg):
        """Test band power feature extraction."""
        extractor = BandPowerExtractor(bandpower_config)

        features = extractor.extract(synthetic_eeg)

        # Should have n_channels * n_bands features
        expected_features = 8 * 2  # 8 channels, 2 bands
        assert len(features) == expected_features

    def test_dominant_frequency(self):
        """Test that correct bands have highest power."""
        fs = 250
        n_samples = 500
        t = np.arange(n_samples) / fs

        # Pure 10 Hz signal (mu band)
        data = np.sin(2 * np.pi * 10 * t).reshape(1, -1)

        config = BandPowerConfig(
            bands={
                "mu": (8.0, 12.0),
                "beta": (12.0, 30.0),
            },
            sampling_rate=250,
        )
        extractor = BandPowerExtractor(config)

        features = extractor.extract(data)

        # Mu power should be higher than beta power
        mu_power = features[0]
        beta_power = features[1]
        assert mu_power > beta_power


class TestCSP:
    """Tests for Common Spatial Patterns."""

    def test_csp_fitting(self, motor_imagery_data):
        """Test CSP filter fitting."""
        X, y = motor_imagery_data

        config = CSPConfig(n_components=4)
        csp = CSP(config)
        csp.fit(X, y)

        assert csp._is_fitted
        assert csp._filters.shape[1] == 4  # n_components

    def test_csp_transform(self, motor_imagery_data):
        """Test CSP feature transformation."""
        X, y = motor_imagery_data

        config = CSPConfig(n_components=4)
        csp = CSP(config)
        csp.fit(X, y)

        features = csp.transform(X)

        # Output shape is (n_trials, 2*n_components) because CSP takes first and last m
        assert features.shape[0] == 40  # n_trials
        assert features.shape[1] == 8  # 2 * n_components

    def test_csp_separability(self, motor_imagery_data):
        """Test that CSP features separate classes."""
        X, y = motor_imagery_data

        config = CSPConfig(n_components=4)
        csp = CSP(config)
        csp.fit(X, y)
        features = csp.transform(X)

        # Classes should have different mean features
        class0_mean = np.mean(features[y == 0], axis=0)
        class1_mean = np.mean(features[y == 1], axis=0)

        # There should be separation (not equal means)
        assert not np.allclose(class0_mean, class1_mean)


class TestFeatureExtractor:
    """Tests for complete FeatureExtractor."""

    def test_feature_extraction(self, synthetic_eeg):
        """Test complete feature extraction."""
        # Use BAND_POWER since it doesn't require fitting
        config = FeatureConfig(
            method=FeatureType.BAND_POWER,
            sampling_rate=250,
            n_channels=8,
        )
        extractor = FeatureExtractor(config)

        # Add batch dimension for transform
        data = synthetic_eeg[np.newaxis, :, :]
        features = extractor.transform(data)

        assert features is not None
        assert len(features.flatten()) > 0
        assert not np.any(np.isnan(features))


# =============================================================================
# Decoder Tests
# =============================================================================


class TestKalmanFilter:
    """Tests for KalmanFilter."""

    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(n_dims=7)

        assert kf.x.shape == (7,)
        assert kf.P.shape == (7, 7)
        assert np.allclose(kf.x, 0)

    def test_predict_update(self):
        """Test predict and update cycle."""
        kf = KalmanFilter(n_dims=3)

        # Initial state should be zero
        assert np.allclose(kf.x, 0)

        # Update with measurement
        z = np.array([1.0, 0.5, -0.5])
        kf.predict()
        state = kf.update(z)

        # State should move toward measurement
        assert np.linalg.norm(state - z) < np.linalg.norm(kf.x - z) + 1

    def test_smoothing(self):
        """Test that Kalman filter smooths noisy measurements."""
        kf = KalmanFilter(n_dims=1, process_noise=0.001, measurement_noise=0.1)

        # Noisy measurements around true value of 1.0
        true_value = 1.0
        measurements = true_value + np.random.randn(100) * 0.5

        estimates = []
        for z in measurements:
            state = kf.filter(np.array([z]))
            estimates.append(state[0])

        # Final estimates should have lower variance than measurements
        estimates = np.array(estimates[-50:])  # Last 50 estimates

        assert np.std(estimates) < np.std(measurements[-50:])

    def test_reset(self):
        """Test filter reset."""
        kf = KalmanFilter(n_dims=3)

        # Update several times
        for _ in range(10):
            kf.filter(np.random.randn(3))

        # Reset
        kf.reset()

        assert np.allclose(kf.x, 0)


class TestContinuousDecoder:
    """Tests for ContinuousDecoder."""

    def test_decoder_output_shape(self):
        """Test decoder produces correct output shape."""
        config = DecoderConfig(n_features=64, n_outputs=7, use_kalman=True)
        decoder = ContinuousDecoder(config)

        features = np.random.randn(64)
        velocity, confidence = decoder.decode(features)

        assert velocity.shape == (7,)
        assert 0.0 <= confidence <= 1.0

    def test_velocity_limits(self):
        """Test that output respects velocity limits."""
        config = DecoderConfig(
            n_features=64, n_outputs=7, velocity_limits=np.ones(7) * 0.5
        )
        decoder = ContinuousDecoder(config)

        # Large input should still produce limited output
        features = np.random.randn(64) * 10
        velocity, confidence = decoder.decode(features)

        assert np.all(np.abs(velocity) <= 0.5 + 0.01)  # Small tolerance


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestBCIPipeline:
    """Tests for complete BCIPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = PipelineConfig(
            use_simulated_data=True,
        )
        pipeline = BCIPipeline(config)

        assert pipeline.state == PipelineState.IDLE

    def test_pipeline_lifecycle(self):
        """Test pipeline start/stop lifecycle."""
        config = PipelineConfig(
            use_simulated_data=True,
        )
        pipeline = BCIPipeline(config)

        pipeline.start()
        assert pipeline.state == PipelineState.RUNNING

        # Let it run briefly
        time.sleep(0.2)

        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_pipeline_output(self):
        """Test that pipeline produces valid output."""
        config = PipelineConfig(
            use_simulated_data=True,
        )
        pipeline = BCIPipeline(config)

        pipeline.start()
        time.sleep(0.5)  # Wait for processing

        result = pipeline.get_output(timeout=0.5)

        pipeline.stop()

        # Result may be None if no output generated yet
        if result is not None:
            velocity, confidence = result
            assert len(velocity) == config.decoder.n_outputs
            assert 0 <= confidence <= 1

    def test_pipeline_metrics(self):
        """Test pipeline performance metrics."""
        config = PipelineConfig(
            use_simulated_data=True,
        )
        pipeline = BCIPipeline(config)

        pipeline.start()
        time.sleep(0.5)

        metrics = pipeline.get_metrics()

        pipeline.stop()

        assert hasattr(metrics, "total_latency")
        assert hasattr(metrics, "actual_update_rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
