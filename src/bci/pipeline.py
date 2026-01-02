"""
BCI Pipeline Module
====================

Complete Brain-Computer Interface pipeline that orchestrates signal
acquisition, preprocessing, feature extraction, and decoding.

Pipeline Flow:
    
    EEG Device → Acquisition → Preprocessing → Features → Decoder → Velocity
                     ↓              ↓             ↓           ↓
              Raw Buffer    Filtered Data   Feature     Command
                                             Vector      Output

Key Features:
    - Asynchronous data flow with thread-safe queues
    - Configurable latency/accuracy tradeoffs
    - Real-time performance monitoring
    - Graceful degradation on component failure
    - State machine for session management

Performance Targets:
    - End-to-end latency: <50ms (acquisition to velocity)
    - Update rate: 30-60 Hz
    - CPU usage: <50% on single core

Thread Safety:
    The pipeline uses separate threads for acquisition and processing.
    All public methods are thread-safe.

Example:
    >>> pipeline = BCIPipeline.from_config("config.yaml")
    >>> pipeline.start()
    >>> 
    >>> while running:
    ...     velocity, confidence = pipeline.get_output()
    ...     arm_controller.send_velocity(velocity)
    ...     
    >>> pipeline.stop()

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Callable
from pathlib import Path
import json
import yaml
import numpy as np

from .acquisition import BaseAcquisition, SimulatedAcquisition, AcquisitionConfig
from .preprocessing import Preprocessor, PreprocessorConfig
from .features import FeatureExtractor, FeatureConfig, FeatureType
from .decoder import ContinuousDecoder, DecoderConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline States
# =============================================================================

class PipelineState(Enum):
    """Pipeline state machine states."""
    IDLE = auto()       # Not started
    STARTING = auto()   # Initialization in progress
    CALIBRATING = auto() # Collecting calibration data
    RUNNING = auto()     # Normal operation
    PAUSED = auto()      # Temporarily paused
    STOPPING = auto()    # Shutdown in progress
    ERROR = auto()       # Error state


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Master configuration for the BCI pipeline.
    
    Combines all component configurations with pipeline-level settings.
    
    Attributes:
        acquisition: EEG acquisition configuration
        preprocessing: Preprocessing configuration
        features: Feature extraction configuration
        decoder: Decoder configuration
        
        window_size_ms: Processing window size in milliseconds
        window_stride_ms: Window stride (overlap = window - stride)
        max_latency_ms: Maximum acceptable latency (triggers warning)
        output_queue_size: Size of output queue
        enable_monitoring: Whether to collect performance metrics
        
        calibration_duration_s: Duration of calibration period
        auto_calibrate: Whether to auto-calibrate on start
    """
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    preprocessing: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    
    window_size_ms: float = 500.0
    window_stride_ms: float = 50.0
    max_latency_ms: float = 100.0
    output_queue_size: int = 10
    enable_monitoring: bool = True
    
    calibration_duration_s: float = 30.0
    auto_calibrate: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        # Check window parameters make sense
        if self.window_stride_ms > self.window_size_ms:
            raise ValueError("window_stride_ms cannot exceed window_size_ms")
        
        # Derive window size in samples
        self.window_samples = int(
            self.window_size_ms / 1000.0 * self.acquisition.sampling_rate
        )
        self.stride_samples = int(
            self.window_stride_ms / 1000.0 * self.acquisition.sampling_rate
        )
        
        # Check feature config matches decoder
        # (features output dimension should match decoder input)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parse nested configurations
        acq_config = AcquisitionConfig(**data.get("acquisition", {}))
        pre_config = PreprocessorConfig(**data.get("preprocessing", {}))
        feat_config = FeatureConfig(**data.get("features", {}))
        dec_config = DecoderConfig(**data.get("decoder", {}))
        
        # Get pipeline-level settings
        pipeline_data = {
            k: v for k, v in data.items()
            if k not in ["acquisition", "preprocessing", "features", "decoder"]
        }
        
        return cls(
            acquisition=acq_config,
            preprocessing=pre_config,
            features=feat_config,
            decoder=dec_config,
            **pipeline_data
        )
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration
        """
        data = {
            "acquisition": {
                "sampling_rate": self.acquisition.sampling_rate,
                "n_channels": self.acquisition.n_channels,
                "buffer_size_seconds": self.acquisition.buffer_size_seconds,
            },
            "preprocessing": {
                # Serialize preprocessing config
            },
            "features": {
                "feature_type": self.features.feature_type.name,
            },
            "decoder": {
                "n_features": self.decoder.n_features,
                "n_outputs": self.decoder.n_outputs,
            },
            "window_size_ms": self.window_size_ms,
            "window_stride_ms": self.window_stride_ms,
            "max_latency_ms": self.max_latency_ms,
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# =============================================================================
# Pipeline Metrics
# =============================================================================

@dataclass
class PipelineMetrics:
    """
    Performance metrics for pipeline monitoring.
    
    Tracks latencies, throughput, and error rates for each stage.
    """
    # Latencies (ms)
    acquisition_latency: float = 0.0
    preprocessing_latency: float = 0.0
    feature_latency: float = 0.0
    decoder_latency: float = 0.0
    total_latency: float = 0.0
    
    # Throughput
    samples_processed: int = 0
    windows_processed: int = 0
    outputs_generated: int = 0
    
    # Rates
    actual_update_rate: float = 0.0
    
    # Errors
    dropped_samples: int = 0
    processing_errors: int = 0
    latency_violations: int = 0
    
    # Quality
    mean_confidence: float = 0.0
    artifact_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency": {
                "acquisition_ms": self.acquisition_latency,
                "preprocessing_ms": self.preprocessing_latency,
                "feature_ms": self.feature_latency,
                "decoder_ms": self.decoder_latency,
                "total_ms": self.total_latency,
            },
            "throughput": {
                "samples": self.samples_processed,
                "windows": self.windows_processed,
                "outputs": self.outputs_generated,
                "update_rate_hz": self.actual_update_rate,
            },
            "errors": {
                "dropped_samples": self.dropped_samples,
                "processing_errors": self.processing_errors,
                "latency_violations": self.latency_violations,
            },
            "quality": {
                "mean_confidence": self.mean_confidence,
                "artifact_rate": self.artifact_rate,
            }
        }


# =============================================================================
# Main Pipeline Class
# =============================================================================

class BCIPipeline:
    """
    Complete BCI pipeline orchestrating all processing stages.
    
    The pipeline manages data flow from EEG acquisition through
    preprocessing, feature extraction, and decoding to produce
    continuous velocity commands.
    
    Architecture:
        ┌─────────────────────────────────────────────────────────┐
        │                     BCIPipeline                          │
        │                                                          │
        │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐│
        │  │Acquisition│→│Preprocess │→│ Features  │→│ Decoder ││
        │  │  Thread   │  │          │  │           │  │         ││
        │  └──────────┘   └──────────┘   └──────────┘   └────────┘│
        │       ↓                                            ↓     │
        │  [Raw Buffer]                              [Output Queue]│
        └─────────────────────────────────────────────────────────┘
    
    Thread Model:
        - Acquisition runs in its own thread (managed by acquisition component)
        - Processing runs in a separate thread (managed by pipeline)
        - Main thread can poll for outputs
    
    Example:
        >>> config = PipelineConfig.from_yaml("config.yaml")
        >>> pipeline = BCIPipeline(config)
        >>> 
        >>> # Register callback for real-time output
        >>> pipeline.register_callback(my_velocity_handler)
        >>> 
        >>> pipeline.start()
        >>> time.sleep(10)
        >>> pipeline.stop()
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        acquisition: Optional[BaseAcquisition] = None
    ) -> None:
        """
        Initialize BCI pipeline.
        
        Args:
            config: Pipeline configuration
            acquisition: Optional custom acquisition instance.
                        If None, SimulatedAcquisition is used.
        """
        self.config = config
        
        # State
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        
        # Components
        if acquisition is not None:
            self.acquisition = acquisition
        else:
            self.acquisition = SimulatedAcquisition(config.acquisition)
        
        self.preprocessor = Preprocessor(config.preprocessing)
        self.feature_extractor = FeatureExtractor(config.features)
        self.decoder = ContinuousDecoder(config.decoder)
        
        # Processing thread
        self._process_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Output queue
        self._output_queue: queue.Queue = queue.Queue(
            maxsize=config.output_queue_size
        )
        
        # Callbacks for real-time output
        self._callbacks: List[Callable[[np.ndarray, float], None]] = []
        self._callback_lock = threading.Lock()
        
        # Metrics
        self._metrics = PipelineMetrics()
        self._metrics_lock = threading.Lock()
        
        # Window buffer for windowed processing
        self._window_buffer = np.zeros(
            (config.acquisition.n_channels, config.window_samples)
        )
        self._buffer_write_pos = 0
        self._samples_since_last_window = 0
        
        # Timing
        self._last_output_time = 0.0
        self._start_time = 0.0
        
        # Calibration data
        self._calibration_data: List[np.ndarray] = []
        self._calibration_features: List[np.ndarray] = []
        
        logger.info(
            f"BCIPipeline initialized: window={config.window_size_ms}ms, "
            f"stride={config.window_stride_ms}ms"
        )
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        with self._state_lock:
            return self._state
    
    def _set_state(self, state: PipelineState) -> None:
        """Set pipeline state (thread-safe)."""
        with self._state_lock:
            old_state = self._state
            self._state = state
            logger.info(f"Pipeline state: {old_state.name} → {state.name}")
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    def start(self) -> None:
        """
        Start the pipeline.
        
        Starts acquisition and processing threads.
        
        Raises:
            RuntimeError: If pipeline is not in IDLE state
        """
        if self.state != PipelineState.IDLE:
            raise RuntimeError(f"Cannot start from state {self.state.name}")
        
        self._set_state(PipelineState.STARTING)
        
        try:
            # Clear state
            self._stop_event.clear()
            self._reset_buffers()
            
            # Start acquisition
            self.acquisition.start()
            
            # Register for samples
            self.acquisition.register_callback(self._on_sample)
            
            # Start processing thread
            self._process_thread = threading.Thread(
                target=self._processing_loop,
                name="BCIPipeline-Processing",
                daemon=True
            )
            self._process_thread.start()
            
            self._start_time = time.perf_counter()
            
            # Auto-calibrate if enabled
            if self.config.auto_calibrate:
                self._set_state(PipelineState.CALIBRATING)
                self._run_calibration()
            
            self._set_state(PipelineState.RUNNING)
            logger.info("Pipeline started successfully")
            
        except Exception as e:
            self._set_state(PipelineState.ERROR)
            logger.error(f"Failed to start pipeline: {e}")
            raise
    
    def stop(self) -> None:
        """
        Stop the pipeline.
        
        Stops acquisition and waits for processing thread to finish.
        """
        if self.state in (PipelineState.IDLE, PipelineState.STOPPING):
            return
        
        self._set_state(PipelineState.STOPPING)
        
        # Signal stop
        self._stop_event.set()
        
        # Stop acquisition
        try:
            self.acquisition.unregister_callback(self._on_sample)
            self.acquisition.stop()
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")
        
        # Wait for processing thread
        if self._process_thread is not None and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
            if self._process_thread.is_alive():
                logger.warning("Processing thread did not stop cleanly")
        
        self._set_state(PipelineState.IDLE)
        logger.info("Pipeline stopped")
    
    def pause(self) -> None:
        """Pause pipeline processing."""
        if self.state == PipelineState.RUNNING:
            self._set_state(PipelineState.PAUSED)
    
    def resume(self) -> None:
        """Resume pipeline processing."""
        if self.state == PipelineState.PAUSED:
            self._set_state(PipelineState.RUNNING)
    
    def _reset_buffers(self) -> None:
        """Reset all internal buffers."""
        self._window_buffer.fill(0)
        self._buffer_write_pos = 0
        self._samples_since_last_window = 0
        
        # Clear output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset decoder
        self.decoder.reset()
    
    # =========================================================================
    # Sample Processing
    # =========================================================================
    
    def _on_sample(self, sample: Any) -> None:
        """
        Callback for new EEG sample.
        
        Adds sample to window buffer. Processing happens in separate thread.
        
        Args:
            sample: EEGSample from acquisition
        """
        if self.state not in (PipelineState.RUNNING, PipelineState.CALIBRATING):
            return
        
        # Add to window buffer
        data = sample.data.flatten()[:self.config.acquisition.n_channels]
        
        # Circular buffer write
        self._window_buffer[:, self._buffer_write_pos] = data
        self._buffer_write_pos = (self._buffer_write_pos + 1) % self.config.window_samples
        self._samples_since_last_window += 1
    
    def _processing_loop(self) -> None:
        """
        Main processing loop running in separate thread.
        
        Processes windows at the configured stride interval.
        """
        logger.info("Processing loop started")
        
        stride_interval = self.config.window_stride_ms / 1000.0
        next_process_time = time.perf_counter() + stride_interval
        
        while not self._stop_event.is_set():
            current_time = time.perf_counter()
            
            if current_time >= next_process_time:
                if self.state == PipelineState.RUNNING:
                    try:
                        self._process_window()
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        with self._metrics_lock:
                            self._metrics.processing_errors += 1
                
                # Schedule next processing
                next_process_time = current_time + stride_interval
            
            # Sleep for a bit to avoid busy waiting
            sleep_time = min(0.001, next_process_time - time.perf_counter())
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Processing loop stopped")
    
    def _process_window(self) -> None:
        """
        Process current window through pipeline stages.
        
        Executes: preprocess → extract features → decode → output
        """
        start_time = time.perf_counter()
        
        # Get current window (handle circular buffer)
        window = self._get_current_window()
        
        # Preprocessing
        t0 = time.perf_counter()
        processed, quality = self.preprocessor.process(window)
        preprocess_time = (time.perf_counter() - t0) * 1000
        
        # Check for artifact rejection
        if quality < 0.5:
            with self._metrics_lock:
                self._metrics.artifact_rate = (
                    0.95 * self._metrics.artifact_rate + 0.05 * 1.0
                )
            # Still process but with reduced confidence
        else:
            with self._metrics_lock:
                self._metrics.artifact_rate = (
                    0.95 * self._metrics.artifact_rate + 0.05 * 0.0
                )
        
        # Feature extraction
        t0 = time.perf_counter()
        features = self.feature_extractor.extract(processed)
        feature_time = (time.perf_counter() - t0) * 1000
        
        # Decoding
        t0 = time.perf_counter()
        velocity, confidence = self.decoder.decode(features)
        decode_time = (time.perf_counter() - t0) * 1000
        
        # Adjust confidence by signal quality
        adjusted_confidence = confidence * quality
        
        # Total latency
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Update metrics
        with self._metrics_lock:
            self._metrics.preprocessing_latency = preprocess_time
            self._metrics.feature_latency = feature_time
            self._metrics.decoder_latency = decode_time
            self._metrics.total_latency = total_time
            self._metrics.windows_processed += 1
            
            # Update rate calculation
            current_time = time.perf_counter()
            if self._last_output_time > 0:
                interval = current_time - self._last_output_time
                instantaneous_rate = 1.0 / interval if interval > 0 else 0
                self._metrics.actual_update_rate = (
                    0.9 * self._metrics.actual_update_rate + 0.1 * instantaneous_rate
                )
            self._last_output_time = current_time
            
            self._metrics.mean_confidence = (
                0.95 * self._metrics.mean_confidence + 0.05 * adjusted_confidence
            )
            
            # Check latency violation
            if total_time > self.config.max_latency_ms:
                self._metrics.latency_violations += 1
                logger.warning(f"Latency violation: {total_time:.1f}ms")
        
        # Output
        self._output(velocity, adjusted_confidence)
    
    def _get_current_window(self) -> np.ndarray:
        """
        Extract current window from circular buffer.
        
        Returns:
            Window data, shape (n_channels, window_samples)
        """
        # Create properly ordered window from circular buffer
        window = np.zeros_like(self._window_buffer)
        
        # Copy from write position to end
        end_samples = self.config.window_samples - self._buffer_write_pos
        window[:, :end_samples] = self._window_buffer[:, self._buffer_write_pos:]
        
        # Copy from start to write position
        if self._buffer_write_pos > 0:
            window[:, end_samples:] = self._window_buffer[:, :self._buffer_write_pos]
        
        return window
    
    def _output(self, velocity: np.ndarray, confidence: float) -> None:
        """
        Send velocity output to queue and callbacks.
        
        Args:
            velocity: Velocity command
            confidence: Confidence score
        """
        # Add to queue (non-blocking)
        try:
            self._output_queue.put_nowait((velocity.copy(), confidence))
            with self._metrics_lock:
                self._metrics.outputs_generated += 1
        except queue.Full:
            # Drop oldest and add new
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._output_queue.put_nowait((velocity.copy(), confidence))
            except queue.Full:
                pass
        
        # Call callbacks
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(velocity, confidence)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    # =========================================================================
    # Output Interface
    # =========================================================================
    
    def get_output(
        self,
        timeout: Optional[float] = 0.1
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get next velocity output.
        
        Args:
            timeout: Maximum time to wait (seconds). None for non-blocking.
            
        Returns:
            Tuple of (velocity, confidence) or None if no output available
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_output_nowait(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get next velocity output without waiting.
        
        Returns:
            Tuple of (velocity, confidence) or None if queue empty
        """
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def register_callback(
        self,
        callback: Callable[[np.ndarray, float], None]
    ) -> None:
        """
        Register callback for velocity outputs.
        
        Callback is called in processing thread, so should be fast.
        
        Args:
            callback: Function(velocity, confidence) → None
        """
        with self._callback_lock:
            self._callbacks.append(callback)
    
    def unregister_callback(
        self,
        callback: Callable[[np.ndarray, float], None]
    ) -> None:
        """
        Unregister velocity callback.
        
        Args:
            callback: Previously registered callback
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    # =========================================================================
    # Calibration
    # =========================================================================
    
    def _run_calibration(self) -> None:
        """Run calibration procedure."""
        logger.info(f"Starting calibration ({self.config.calibration_duration_s}s)")
        
        start_time = time.perf_counter()
        self._calibration_data.clear()
        self._calibration_features.clear()
        
        while time.perf_counter() - start_time < self.config.calibration_duration_s:
            if self._stop_event.is_set():
                return
            
            # Collect window
            window = self._get_current_window()
            processed, _ = self.preprocessor.process(window)
            self._calibration_data.append(processed.copy())
            
            time.sleep(0.1)
        
        # Fit feature extractor if CSP
        if self.feature_extractor.config.feature_type == FeatureType.CSP:
            logger.info("Fitting CSP on calibration data")
            # Would need labels for proper CSP fitting
            # This is a placeholder for the actual calibration logic
        
        logger.info(f"Calibration complete: {len(self._calibration_data)} windows")
    
    def calibrate(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> None:
        """
        Calibrate pipeline with labeled data.
        
        Args:
            data: Calibration EEG data, shape (n_trials, n_channels, n_samples)
            labels: Trial labels for supervised methods like CSP
        """
        logger.info(f"Manual calibration with {data.shape[0]} trials")
        
        # Process all trials
        processed_trials = []
        for trial in data:
            processed, _ = self.preprocessor.process(trial)
            processed_trials.append(processed)
        
        processed_data = np.array(processed_trials)
        
        # Fit CSP if applicable
        if labels is not None:
            self.feature_extractor.fit(processed_data, labels)
        
        logger.info("Calibration complete")
    
    # =========================================================================
    # Metrics & Monitoring
    # =========================================================================
    
    def get_metrics(self) -> PipelineMetrics:
        """Get copy of current metrics."""
        with self._metrics_lock:
            # Return a copy
            return PipelineMetrics(
                acquisition_latency=self._metrics.acquisition_latency,
                preprocessing_latency=self._metrics.preprocessing_latency,
                feature_latency=self._metrics.feature_latency,
                decoder_latency=self._metrics.decoder_latency,
                total_latency=self._metrics.total_latency,
                samples_processed=self._metrics.samples_processed,
                windows_processed=self._metrics.windows_processed,
                outputs_generated=self._metrics.outputs_generated,
                actual_update_rate=self._metrics.actual_update_rate,
                dropped_samples=self._metrics.dropped_samples,
                processing_errors=self._metrics.processing_errors,
                latency_violations=self._metrics.latency_violations,
                mean_confidence=self._metrics.mean_confidence,
                artifact_rate=self._metrics.artifact_rate,
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dictionary with status information
        """
        metrics = self.get_metrics()
        
        return {
            "state": self.state.name,
            "uptime_s": time.perf_counter() - self._start_time if self._start_time > 0 else 0,
            "metrics": metrics.to_dict(),
            "acquisition": {
                "connected": self.acquisition.is_running,
                "buffer_fill": self._samples_since_last_window / self.config.stride_samples,
            },
            "decoder": self.decoder.get_diagnostics(),
        }
    
    def print_status(self) -> None:
        """Print formatted status to console."""
        status = self.get_status()
        metrics = status["metrics"]
        
        print("\n" + "=" * 60)
        print(f"BCI Pipeline Status: {status['state']}")
        print(f"Uptime: {status['uptime_s']:.1f}s")
        print("-" * 60)
        print(f"Latency: {metrics['latency']['total_ms']:.1f}ms "
              f"(pre: {metrics['latency']['preprocessing_ms']:.1f}, "
              f"feat: {metrics['latency']['feature_ms']:.1f}, "
              f"dec: {metrics['latency']['decoder_ms']:.1f})")
        print(f"Update Rate: {metrics['throughput']['update_rate_hz']:.1f} Hz")
        print(f"Windows Processed: {metrics['throughput']['windows']}")
        print(f"Mean Confidence: {metrics['quality']['mean_confidence']:.2f}")
        print(f"Artifact Rate: {metrics['quality']['artifact_rate']:.1%}")
        print(f"Errors: {metrics['errors']['processing_errors']}, "
              f"Latency Violations: {metrics['errors']['latency_violations']}")
        print("=" * 60 + "\n")
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save_state(self, directory: str) -> None:
        """
        Save pipeline state for later resumption.
        
        Saves:
        - Configuration
        - Decoder weights
        - Feature extractor state (CSP patterns, etc.)
        
        Args:
            directory: Directory to save state files
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.to_yaml(str(path / "config.yaml"))
        
        # Save decoder
        self.decoder.save_weights(str(path / "decoder_weights.pt"))
        
        # Save feature extractor
        self.feature_extractor.save(str(path / "feature_extractor.npz"))
        
        logger.info(f"Pipeline state saved to {directory}")
    
    def load_state(self, directory: str) -> None:
        """
        Load pipeline state.
        
        Args:
            directory: Directory containing state files
        """
        path = Path(directory)
        
        # Load decoder weights
        decoder_path = path / "decoder_weights.pt"
        if decoder_path.exists():
            self.decoder.load_weights(str(decoder_path))
        
        # Load feature extractor
        feature_path = path / "feature_extractor.npz"
        if feature_path.exists():
            self.feature_extractor.load(str(feature_path))
        
        logger.info(f"Pipeline state loaded from {directory}")
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self) -> "BCIPipeline":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_default_pipeline() -> BCIPipeline:
    """
    Create a pipeline with default configuration.
    
    Returns:
        BCIPipeline with simulated acquisition
    """
    config = PipelineConfig()
    return BCIPipeline(config)


def create_pipeline_from_config(config_path: str) -> BCIPipeline:
    """
    Create a pipeline from configuration file.
    
    Args:
        config_path: Path to YAML configuration
        
    Returns:
        Configured BCIPipeline
    """
    config = PipelineConfig.from_yaml(config_path)
    return BCIPipeline(config)
