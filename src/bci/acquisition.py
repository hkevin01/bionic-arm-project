"""
EEG Signal Acquisition Module
=============================

Provides a unified interface for acquiring EEG signals from various hardware
devices (OpenBCI, Emotiv, LSL streams, etc.) with built-in buffering,
thread-safe data access, and graceful error handling.

Key Features:
- Abstract base class for device-agnostic implementation
- Thread-safe circular buffer for continuous data access
- Callback system for real-time data processing
- Automatic reconnection on connection loss
- Comprehensive timing and performance metrics

Mathematical Background:
    EEG signals are sampled at rate fs (typically 250-1000 Hz)
    Buffer stores N = fs * T samples for T seconds of history
    Data format: (n_channels, n_samples) with values in microvolts (µV)

Usage:
    config = AcquisitionConfig(sampling_rate=250, n_channels=32)
    device = OpenBCIAcquisition(config, port="/dev/ttyUSB0")
    device.connect()
    device.start()
    data = device.get_latest_data(n_samples=250)  # Last 1 second
    device.stop()
    device.disconnect()

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Full, Empty
from typing import Callable, Optional, List, Dict, Any
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class DeviceState(Enum):
    """
    Enumeration of possible device states.
    
    State Machine:
        DISCONNECTED → CONNECTING → CONNECTED → STREAMING → CONNECTED → DISCONNECTED
                     ↓                                    ↓
                   ERROR ←─────────────────────────────────
    """
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    STREAMING = auto()
    ERROR = auto()
    RECONNECTING = auto()


class AcquisitionError(Exception):
    """Base exception for acquisition-related errors."""
    pass


class ConnectionError(AcquisitionError):
    """Raised when device connection fails."""
    pass


class DataError(AcquisitionError):
    """Raised when data acquisition encounters an error."""
    pass


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class AcquisitionConfig:
    """
    Configuration parameters for EEG acquisition.
    
    Attributes:
        sampling_rate: Samples per second (Hz). Common values: 250, 500, 1000
        n_channels: Number of EEG channels. Common values: 8, 16, 32, 64
        buffer_size_seconds: Duration of circular buffer in seconds
        channel_names: Optional list of channel names (e.g., ['Fz', 'Cz', 'Pz'])
        reference_channel: Channel used as reference (None for common average)
        notch_freq: Power line frequency for notch filter (50 or 60 Hz)
        max_reconnect_attempts: Maximum reconnection attempts before giving up
        reconnect_delay_seconds: Delay between reconnection attempts
        
    Validation:
        - sampling_rate must be > 0 and <= 10000 Hz
        - n_channels must be > 0 and <= 256
        - buffer_size_seconds must be > 0 and <= 300 seconds
    """
    sampling_rate: int = 250
    n_channels: int = 32
    buffer_size_seconds: float = 5.0
    channel_names: Optional[List[str]] = None
    reference_channel: Optional[int] = None
    notch_freq: float = 60.0
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        # Validate sampling rate
        if not (0 < self.sampling_rate <= 10000):
            raise ValueError(
                f"sampling_rate must be between 1 and 10000 Hz, got {self.sampling_rate}"
            )
        
        # Validate channel count
        if not (0 < self.n_channels <= 256):
            raise ValueError(
                f"n_channels must be between 1 and 256, got {self.n_channels}"
            )
        
        # Validate buffer size
        if not (0 < self.buffer_size_seconds <= 300):
            raise ValueError(
                f"buffer_size_seconds must be between 0 and 300, got {self.buffer_size_seconds}"
            )
        
        # Auto-generate channel names if not provided
        if self.channel_names is None:
            self.channel_names = [f"CH{i+1}" for i in range(self.n_channels)]
        elif len(self.channel_names) != self.n_channels:
            raise ValueError(
                f"channel_names length ({len(self.channel_names)}) must match "
                f"n_channels ({self.n_channels})"
            )
        
        # Validate notch frequency
        if self.notch_freq not in (50.0, 60.0):
            logger.warning(
                f"Non-standard notch frequency: {self.notch_freq} Hz. "
                "Common values are 50 Hz (Europe) or 60 Hz (Americas)."
            )
    
    @property
    def buffer_samples(self) -> int:
        """Calculate number of samples in buffer."""
        return int(self.buffer_size_seconds * self.sampling_rate)
    
    @property
    def sample_period_ms(self) -> float:
        """Calculate time between samples in milliseconds."""
        return 1000.0 / self.sampling_rate


@dataclass
class EEGSample:
    """
    Container for EEG data samples.
    
    Attributes:
        data: EEG data array. Shape: (n_channels,) for single sample,
              or (n_channels, n_samples) for a chunk
        timestamp: Unix timestamp (seconds since epoch) of first sample
        sample_index: Global sample counter for synchronization
        quality: Optional per-channel signal quality metrics (0-1)
        
    Notes:
        - Data values are in microvolts (µV)
        - Timestamp uses monotonic clock for accurate timing
        - sample_index is cumulative and never resets during a session
    """
    data: np.ndarray
    timestamp: float
    sample_index: int
    quality: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate sample data after initialization."""
        if self.data.ndim not in (1, 2):
            raise ValueError(
                f"data must be 1D or 2D array, got {self.data.ndim}D"
            )
        
        # Ensure data is float type for numerical operations
        if not np.issubdtype(self.data.dtype, np.floating):
            self.data = self.data.astype(np.float32)
    
    @property
    def n_channels(self) -> int:
        """Number of channels in this sample."""
        return self.data.shape[0]
    
    @property
    def n_samples(self) -> int:
        """Number of time samples in this chunk."""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[1]
    
    @property
    def is_chunk(self) -> bool:
        """Whether this contains multiple samples."""
        return self.data.ndim == 2 and self.data.shape[1] > 1


@dataclass
class AcquisitionMetrics:
    """
    Performance metrics for monitoring acquisition quality.
    
    Attributes:
        samples_received: Total samples received since start
        samples_dropped: Samples lost due to buffer overflow or timing issues
        latency_ms: Current end-to-end latency in milliseconds
        jitter_ms: Standard deviation of inter-sample timing
        uptime_seconds: Time since acquisition started
        reconnect_count: Number of reconnection attempts
        
    These metrics are crucial for:
        - Real-time performance monitoring
        - Debugging timing issues
        - Ensuring data quality for BCI classification
    """
    samples_received: int = 0
    samples_dropped: int = 0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    uptime_seconds: float = 0.0
    reconnect_count: int = 0
    
    @property
    def drop_rate(self) -> float:
        """Calculate sample drop rate as percentage."""
        total = self.samples_received + self.samples_dropped
        if total == 0:
            return 0.0
        return 100.0 * self.samples_dropped / total


# =============================================================================
# Main Acquisition Class
# =============================================================================

class BaseAcquisition(ABC):
    """
    Abstract base class for EEG signal acquisition.
    
    This class provides a unified interface for acquiring EEG data from
    various hardware devices. Subclasses implement device-specific
    communication protocols.
    
    Architecture:
        ┌──────────────────────────────────────────────────────────┐
        │                    BaseAcquisition                       │
        │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
        │  │ Device   │───▶│ Acq Loop │───▶│ Circular Buffer  │   │
        │  │ Hardware │    │ (Thread) │    │ (Thread-safe)    │   │
        │  └──────────┘    └──────────┘    └──────────────────┘   │
        │        │                                   │             │
        │        ▼                                   ▼             │
        │  ┌──────────┐                       ┌──────────────┐    │
        │  │ Metrics  │                       │  Callbacks   │    │
        │  └──────────┘                       └──────────────┘    │
        └──────────────────────────────────────────────────────────┘
    
    Thread Safety:
        - All public methods are thread-safe
        - Buffer access uses read-write locks
        - State transitions are atomic
    
    Error Handling:
        - Connection errors trigger automatic reconnection
        - Data errors are logged and counted in metrics
        - Critical errors set state to ERROR
    
    Example:
        >>> config = AcquisitionConfig(sampling_rate=250, n_channels=8)
        >>> device = ConcreteAcquisition(config)
        >>> device.connect()
        >>> device.start()
        >>> 
        >>> # Register callback for real-time processing
        >>> device.register_callback(lambda sample: process(sample))
        >>> 
        >>> # Or poll for data
        >>> data = device.get_latest_data(250)  # Last second
        >>> 
        >>> device.stop()
        >>> device.disconnect()
    """
    
    def __init__(self, config: AcquisitionConfig) -> None:
        """
        Initialize acquisition with configuration.
        
        Args:
            config: Acquisition configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._state = DeviceState.DISCONNECTED
        self._state_lock = threading.Lock()
        
        # Circular buffer for continuous data access
        # Shape: (n_channels, buffer_samples)
        self._buffer = np.zeros(
            (config.n_channels, config.buffer_samples),
            dtype=np.float32
        )
        self._buffer_index = 0
        self._buffer_lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
        self._buffer_filled = False  # True once buffer wraps around
        
        # Sample counter for synchronization
        self._sample_counter = 0
        self._sample_lock = threading.Lock()
        
        # Callback system
        self._callbacks: List[Callable[[EEGSample], None]] = []
        self._callback_lock = threading.Lock()
        
        # Data queue for consumers (limited size to prevent memory issues)
        self._data_queue: Queue[EEGSample] = Queue(maxsize=1000)
        
        # Acquisition thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Metrics
        self._metrics = AcquisitionMetrics()
        self._start_time: Optional[float] = None
        self._last_sample_time: Optional[float] = None
        self._timing_samples: List[float] = []  # For jitter calculation
        
        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"{config.n_channels} channels @ {config.sampling_rate} Hz"
        )
    
    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _connect_device(self) -> bool:
        """
        Establish connection to the EEG device.
        
        This method should:
            1. Open communication channel (serial, USB, Bluetooth, etc.)
            2. Verify device identity and firmware version
            3. Configure device parameters (gain, sampling rate, etc.)
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails irrecoverably
        """
        pass
    
    @abstractmethod
    def _disconnect_device(self) -> None:
        """
        Close connection to the EEG device.
        
        This method should:
            1. Stop any ongoing streaming
            2. Close communication channel
            3. Release hardware resources
            
        Should not raise exceptions - log errors instead.
        """
        pass
    
    @abstractmethod
    def _read_samples(self) -> Optional[EEGSample]:
        """
        Read samples from the device.
        
        This method is called continuously by the acquisition loop.
        It should block until data is available or timeout occurs.
        
        Returns:
            EEGSample with data, or None if no data available
            
        Raises:
            DataError: If data reading fails
        """
        pass
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def connect(self) -> bool:
        """
        Connect to the EEG device.
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError: If connection fails after retries
            RuntimeError: If already connected
        """
        with self._state_lock:
            if self._state not in (DeviceState.DISCONNECTED, DeviceState.ERROR):
                raise RuntimeError(
                    f"Cannot connect from state {self._state.name}"
                )
            self._state = DeviceState.CONNECTING
        
        logger.info(f"Connecting to {self.__class__.__name__}...")
        
        # Attempt connection with retries
        for attempt in range(self.config.max_reconnect_attempts):
            try:
                if self._connect_device():
                    with self._state_lock:
                        self._state = DeviceState.CONNECTED
                    logger.info("Connection successful")
                    return True
                    
            except Exception as e:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.config.max_reconnect_attempts - 1:
                    time.sleep(self.config.reconnect_delay_seconds)
        
        # All attempts failed
        with self._state_lock:
            self._state = DeviceState.ERROR
        raise ConnectionError(
            f"Failed to connect after {self.config.max_reconnect_attempts} attempts"
        )
    
    def disconnect(self) -> None:
        """
        Disconnect from the EEG device.
        
        Stops acquisition if running, then closes connection.
        Safe to call multiple times.
        """
        # Stop acquisition if running
        if self.is_streaming:
            self.stop()
        
        with self._state_lock:
            if self._state == DeviceState.DISCONNECTED:
                return
        
        logger.info("Disconnecting...")
        
        try:
            self._disconnect_device()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            with self._state_lock:
                self._state = DeviceState.DISCONNECTED
            logger.info("Disconnected")
    
    def start(self) -> None:
        """
        Start data acquisition.
        
        Begins reading data from device in a background thread.
        
        Raises:
            RuntimeError: If not connected or already streaming
        """
        with self._state_lock:
            if self._state != DeviceState.CONNECTED:
                raise RuntimeError(
                    f"Cannot start from state {self._state.name}. "
                    "Must be CONNECTED."
                )
            self._state = DeviceState.STREAMING
        
        # Reset metrics
        self._metrics = AcquisitionMetrics()
        self._start_time = time.monotonic()
        self._sample_counter = 0
        self._buffer_filled = False
        self._timing_samples.clear()
        
        # Clear stop event and start thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name=f"{self.__class__.__name__}_AcqThread",
            daemon=True
        )
        self._thread.start()
        
        logger.info("Acquisition started")
    
    def stop(self) -> None:
        """
        Stop data acquisition.
        
        Signals acquisition thread to stop and waits for it to finish.
        Safe to call multiple times.
        """
        with self._state_lock:
            if self._state != DeviceState.STREAMING:
                return
        
        logger.info("Stopping acquisition...")
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Acquisition thread did not stop cleanly")
            self._thread = None
        
        with self._state_lock:
            self._state = DeviceState.CONNECTED
        
        # Update uptime
        if self._start_time is not None:
            self._metrics.uptime_seconds = time.monotonic() - self._start_time
        
        logger.info(
            f"Acquisition stopped. Received {self._metrics.samples_received} samples, "
            f"dropped {self._metrics.samples_dropped} ({self._metrics.drop_rate:.2f}%)"
        )
    
    def register_callback(
        self,
        callback: Callable[[EEGSample], None]
    ) -> None:
        """
        Register a callback function for real-time data processing.
        
        Callbacks are invoked in the acquisition thread whenever new
        data arrives. Keep callbacks fast to avoid data loss.
        
        Args:
            callback: Function that takes an EEGSample as argument
            
        Warning:
            Callbacks should complete quickly (<10ms) to avoid
            blocking the acquisition loop. For heavy processing,
            use a separate consumer thread with get_sample().
        """
        with self._callback_lock:
            self._callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")
    
    def unregister_callback(
        self,
        callback: Callable[[EEGSample], None]
    ) -> bool:
        """
        Remove a previously registered callback.
        
        Args:
            callback: The callback function to remove
            
        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._callback_lock:
            try:
                self._callbacks.remove(callback)
                logger.debug(f"Unregistered callback: {callback.__name__}")
                return True
            except ValueError:
                return False
    
    def get_latest_data(self, n_samples: int) -> np.ndarray:
        """
        Get the most recent n_samples from the circular buffer.
        
        This method provides access to recent history without blocking
        the acquisition loop. Thread-safe.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            numpy array of shape (n_channels, n_samples)
            
        Raises:
            ValueError: If n_samples exceeds buffer size
            
        Note:
            If buffer hasn't filled yet, returns zeros for missing data.
        """
        buffer_len = self._buffer.shape[1]
        
        if n_samples > buffer_len:
            raise ValueError(
                f"Requested {n_samples} samples, but buffer only holds "
                f"{buffer_len} ({self.config.buffer_size_seconds}s)"
            )
        
        with self._buffer_lock:
            end_idx = self._buffer_index
            start_idx = end_idx - n_samples
            
            if start_idx >= 0:
                # Simple case: contiguous slice
                return self._buffer[:, start_idx:end_idx].copy()
            else:
                # Wrap around: concatenate end and beginning
                return np.concatenate([
                    self._buffer[:, start_idx:],
                    self._buffer[:, :end_idx]
                ], axis=1)
    
    def get_sample(self, timeout: Optional[float] = None) -> Optional[EEGSample]:
        """
        Get next sample from the data queue.
        
        This is an alternative to callbacks for consuming data.
        Use this when processing cannot keep up with real-time
        and some buffering is acceptable.
        
        Args:
            timeout: Maximum time to wait in seconds (None = block forever)
            
        Returns:
            EEGSample if available, None if timeout
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    @property
    def state(self) -> DeviceState:
        """Current device state."""
        with self._state_lock:
            return self._state
    
    @property
    def is_connected(self) -> bool:
        """Whether device is connected (may or may not be streaming)."""
        return self.state in (DeviceState.CONNECTED, DeviceState.STREAMING)
    
    @property
    def is_streaming(self) -> bool:
        """Whether device is actively streaming data."""
        return self.state == DeviceState.STREAMING
    
    @property
    def metrics(self) -> AcquisitionMetrics:
        """Current acquisition metrics."""
        # Update uptime if streaming
        if self._start_time is not None and self.is_streaming:
            self._metrics.uptime_seconds = time.monotonic() - self._start_time
        return self._metrics
    
    @property
    def sample_count(self) -> int:
        """Total samples received since start."""
        with self._sample_lock:
            return self._sample_counter
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _acquisition_loop(self) -> None:
        """
        Main acquisition loop running in background thread.
        
        Continuously reads samples from device, updates buffer,
        and notifies callbacks. Handles errors gracefully with
        automatic reconnection when possible.
        """
        logger.debug("Acquisition loop started")
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self._stop_event.is_set():
            try:
                # Read samples from device
                sample = self._read_samples()
                
                if sample is None:
                    # No data available, brief sleep to prevent busy-wait
                    time.sleep(0.001)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Update timing metrics
                current_time = time.monotonic()
                if self._last_sample_time is not None:
                    interval = (current_time - self._last_sample_time) * 1000
                    self._timing_samples.append(interval)
                    
                    # Keep only last 1000 samples for jitter calc
                    if len(self._timing_samples) > 1000:
                        self._timing_samples.pop(0)
                    
                    # Update jitter (std dev of intervals)
                    if len(self._timing_samples) > 10:
                        self._metrics.jitter_ms = float(np.std(self._timing_samples))
                
                self._last_sample_time = current_time
                
                # Process the sample
                self._on_data_received(sample)
                
            except DataError as e:
                consecutive_errors += 1
                logger.warning(f"Data error: {e}")
                self._metrics.samples_dropped += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({consecutive_errors}), "
                        "attempting reconnection"
                    )
                    self._attempt_reconnection()
                    consecutive_errors = 0
                    
            except Exception as e:
                logger.exception(f"Unexpected error in acquisition loop: {e}")
                self._metrics.samples_dropped += 1
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Too many errors, stopping acquisition")
                    with self._state_lock:
                        self._state = DeviceState.ERROR
                    break
        
        logger.debug("Acquisition loop ended")
    
    def _on_data_received(self, sample: EEGSample) -> None:
        """
        Process received sample: update buffer, metrics, and callbacks.
        
        Args:
            sample: The received EEG sample
        """
        # Update sample index
        with self._sample_lock:
            sample.sample_index = self._sample_counter
            if sample.is_chunk:
                self._sample_counter += sample.n_samples
            else:
                self._sample_counter += 1
        
        # Update circular buffer
        with self._buffer_lock:
            if sample.is_chunk:
                # Handle chunk of multiple samples
                n_new = sample.n_samples
                space_at_end = self._buffer.shape[1] - self._buffer_index
                
                if n_new <= space_at_end:
                    # Fits without wrapping
                    self._buffer[:, self._buffer_index:self._buffer_index + n_new] = sample.data
                    self._buffer_index += n_new
                else:
                    # Need to wrap around
                    self._buffer[:, self._buffer_index:] = sample.data[:, :space_at_end]
                    remaining = n_new - space_at_end
                    self._buffer[:, :remaining] = sample.data[:, space_at_end:]
                    self._buffer_index = remaining
                    self._buffer_filled = True
                
                # Handle index wraparound
                if self._buffer_index >= self._buffer.shape[1]:
                    self._buffer_index = 0
                    self._buffer_filled = True
            else:
                # Single sample
                self._buffer[:, self._buffer_index] = sample.data
                self._buffer_index = (self._buffer_index + 1) % self._buffer.shape[1]
                if self._buffer_index == 0:
                    self._buffer_filled = True
        
        # Update metrics
        self._metrics.samples_received += sample.n_samples if sample.is_chunk else 1
        self._metrics.latency_ms = (time.monotonic() - sample.timestamp) * 1000
        
        # Add to queue for consumers (non-blocking, drop if full)
        try:
            self._data_queue.put_nowait(sample)
        except Full:
            self._metrics.samples_dropped += 1
        
        # Notify callbacks
        with self._callback_lock:
            callbacks = list(self._callbacks)  # Copy to avoid lock during callbacks
        
        for callback in callbacks:
            try:
                callback(sample)
            except Exception as e:
                logger.error(f"Callback {callback.__name__} raised exception: {e}")
    
    def _attempt_reconnection(self) -> bool:
        """
        Attempt to reconnect to the device.
        
        Returns:
            True if reconnection successful
        """
        with self._state_lock:
            self._state = DeviceState.RECONNECTING
        
        self._metrics.reconnect_count += 1
        logger.info(f"Attempting reconnection (attempt {self._metrics.reconnect_count})")
        
        try:
            self._disconnect_device()
        except Exception:
            pass
        
        time.sleep(self.config.reconnect_delay_seconds)
        
        try:
            if self._connect_device():
                with self._state_lock:
                    self._state = DeviceState.STREAMING
                logger.info("Reconnection successful")
                return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
        
        with self._state_lock:
            self._state = DeviceState.ERROR
        return False
    
    def __enter__(self) -> 'BaseAcquisition':
        """Context manager entry - connect and start."""
        self.connect()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop and disconnect."""
        self.stop()
        self.disconnect()
    
    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        try:
            if self.is_streaming:
                self.stop()
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass  # Ignore errors during cleanup


# =============================================================================
# Simulated Acquisition (for testing without hardware)
# =============================================================================

class SimulatedAcquisition(BaseAcquisition):
    """
    Simulated EEG acquisition for testing and development.
    
    Generates synthetic EEG-like signals with:
    - Alpha rhythm (8-12 Hz) with configurable amplitude
    - Beta rhythm (13-30 Hz) 
    - Pink noise (1/f spectrum)
    - Optional motor imagery patterns
    
    Useful for:
    - Testing the pipeline without hardware
    - Benchmarking processing algorithms
    - Developing and debugging BCI classifiers
    """
    
    def __init__(
        self,
        config: AcquisitionConfig,
        include_alpha: bool = True,
        include_beta: bool = True,
        noise_level: float = 10.0,
        motor_imagery_class: Optional[int] = None
    ) -> None:
        """
        Initialize simulated acquisition.
        
        Args:
            config: Acquisition configuration
            include_alpha: Whether to include alpha rhythm
            include_beta: Whether to include beta rhythm
            noise_level: Standard deviation of noise in µV
            motor_imagery_class: If set, simulates motor imagery patterns
                                 (0=rest, 1=left, 2=right, 3=both)
        """
        super().__init__(config)
        self.include_alpha = include_alpha
        self.include_beta = include_beta
        self.noise_level = noise_level
        self.motor_imagery_class = motor_imagery_class
        
        # Internal state
        self._phase = 0.0
        self._connected = False
        self._chunk_size = max(1, config.sampling_rate // 60)  # ~60 Hz update
    
    def _connect_device(self) -> bool:
        """Simulate device connection."""
        time.sleep(0.1)  # Simulate connection delay
        self._connected = True
        return True
    
    def _disconnect_device(self) -> None:
        """Simulate device disconnection."""
        self._connected = False
    
    def _read_samples(self) -> Optional[EEGSample]:
        """
        Generate synthetic EEG samples.
        
        Returns:
            EEGSample with synthetic data
        """
        if not self._connected:
            return None
        
        # Simulate sampling delay
        sleep_time = self._chunk_size / self.config.sampling_rate
        time.sleep(sleep_time * 0.9)  # Slightly faster to allow processing
        
        # Generate time vector
        t = np.arange(self._chunk_size) / self.config.sampling_rate + self._phase
        self._phase += self._chunk_size / self.config.sampling_rate
        
        # Initialize data array
        data = np.zeros((self.config.n_channels, self._chunk_size), dtype=np.float32)
        
        # Add alpha rhythm (8-12 Hz) - stronger in posterior channels
        if self.include_alpha:
            alpha_freq = 10.0  # Hz
            alpha_amp = 20.0  # µV
            for ch in range(self.config.n_channels):
                # Posterior channels have stronger alpha
                posterior_weight = 0.5 + 0.5 * (ch / self.config.n_channels)
                data[ch] += alpha_amp * posterior_weight * np.sin(2 * np.pi * alpha_freq * t)
        
        # Add beta rhythm (13-30 Hz) - stronger in motor channels
        if self.include_beta:
            beta_freq = 20.0  # Hz
            beta_amp = 10.0  # µV
            for ch in range(self.config.n_channels):
                # Central channels have stronger beta
                central_weight = 1.0 - abs(ch / self.config.n_channels - 0.5) * 2
                data[ch] += beta_amp * central_weight * np.sin(2 * np.pi * beta_freq * t)
        
        # Add motor imagery patterns if specified
        if self.motor_imagery_class is not None:
            self._add_motor_imagery_pattern(data, t)
        
        # Add pink noise
        data += self._generate_pink_noise(self.config.n_channels, self._chunk_size)
        
        # Add white noise
        data += np.random.randn(*data.shape) * self.noise_level
        
        return EEGSample(
            data=data,
            timestamp=time.time(),
            sample_index=0  # Will be set by _on_data_received
        )
    
    def _add_motor_imagery_pattern(self, data: np.ndarray, t: np.ndarray) -> None:
        """
        Add motor imagery-specific patterns to the data.
        
        Simulates event-related desynchronization (ERD) in motor cortex:
        - Left hand imagery: ERD over right motor cortex (C4)
        - Right hand imagery: ERD over left motor cortex (C3)
        - Both hands: Bilateral ERD
        
        Args:
            data: Data array to modify in-place
            t: Time vector
        """
        mu_freq = 10.0  # Mu rhythm frequency
        mu_amp = 15.0   # Baseline amplitude
        
        n_channels = data.shape[0]
        left_motor = n_channels // 3      # Approximate C3 position
        right_motor = 2 * n_channels // 3  # Approximate C4 position
        
        if self.motor_imagery_class == 0:
            # Rest - strong bilateral mu rhythm
            data[left_motor] += mu_amp * np.sin(2 * np.pi * mu_freq * t)
            data[right_motor] += mu_amp * np.sin(2 * np.pi * mu_freq * t)
            
        elif self.motor_imagery_class == 1:
            # Left hand - ERD over right motor cortex
            data[left_motor] += mu_amp * np.sin(2 * np.pi * mu_freq * t)
            data[right_motor] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t)  # Suppressed
            
        elif self.motor_imagery_class == 2:
            # Right hand - ERD over left motor cortex
            data[left_motor] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t)  # Suppressed
            data[right_motor] += mu_amp * np.sin(2 * np.pi * mu_freq * t)
            
        elif self.motor_imagery_class == 3:
            # Both hands - bilateral ERD
            data[left_motor] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t)
            data[right_motor] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t)
    
    def _generate_pink_noise(self, n_channels: int, n_samples: int) -> np.ndarray:
        """
        Generate pink (1/f) noise.
        
        Pink noise has equal energy per octave, which is more realistic
        for biological signals than white noise.
        
        Args:
            n_channels: Number of channels
            n_samples: Number of samples
            
        Returns:
            Pink noise array of shape (n_channels, n_samples)
        """
        # Generate white noise
        white = np.random.randn(n_channels, n_samples)
        
        # Apply 1/f filter in frequency domain
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1  # Avoid division by zero
        
        # 1/f spectrum
        spectrum = 1 / np.sqrt(freqs)
        spectrum[0] = 0  # No DC component
        
        # Apply filter
        pink = np.zeros_like(white)
        for ch in range(n_channels):
            white_fft = np.fft.rfft(white[ch])
            pink_fft = white_fft * spectrum
            pink[ch] = np.fft.irfft(pink_fft, n=n_samples)
        
        # Scale to reasonable amplitude
        pink *= self.noise_level * 0.5
        
        return pink.astype(np.float32)
    
    def set_motor_imagery_class(self, class_id: int) -> None:
        """
        Change the simulated motor imagery class.
        
        Args:
            class_id: 0=rest, 1=left, 2=right, 3=both
        """
        if class_id not in (0, 1, 2, 3):
            raise ValueError(f"Invalid class_id: {class_id}. Must be 0-3.")
        self.motor_imagery_class = class_id
        logger.debug(f"Motor imagery class set to {class_id}")


# Provide a simple RWLock fallback if not available
if not hasattr(threading, 'RWLock'):
    class _RWLock:
        """Simple read-write lock fallback using standard Lock."""
        def __init__(self):
            self._lock = threading.Lock()
        def __enter__(self):
            self._lock.acquire()
        def __exit__(self, *args):
            self._lock.release()
    threading.RWLock = _RWLock
