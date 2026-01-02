"""
System Monitor Module
=====================

Real-time monitoring of system health, performance metrics,
and diagnostics for the bionic arm system.

Features:
    - Component health tracking
    - Latency measurement
    - Resource utilization monitoring
    - Event logging and alerting
    - Performance statistics

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import threading
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Component health status levels."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()
    OFFLINE = auto()


class MetricType(Enum):
    """Types of monitored metrics."""
    LATENCY = auto()
    THROUGHPUT = auto()
    ERROR_RATE = auto()
    RESOURCE_USAGE = auto()
    QUALITY = auto()


@dataclass
class MonitorConfig:
    """
    Configuration for system monitoring.
    
    Attributes:
        sample_rate_hz: How often to sample metrics
        history_seconds: Duration of metric history to keep
        alert_thresholds: Thresholds for alerting (metric_name -> value)
        enable_logging: Whether to log metrics to file
        log_path: Path to metrics log file
        component_timeouts_ms: Timeout before marking component unhealthy
    """
    sample_rate_hz: float = 10.0
    history_seconds: float = 60.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latency_ms": 150.0,
        "error_rate": 0.1,
        "cpu_percent": 80.0,
        "memory_percent": 80.0,
    })
    enable_logging: bool = False
    log_path: Optional[str] = None
    component_timeouts_ms: Dict[str, float] = field(default_factory=lambda: {
        "bci": 100.0,
        "control": 50.0,
        "feedback": 200.0,
    })
    
    @property
    def sample_period(self) -> float:
        """Time between samples in seconds."""
        return 1.0 / self.sample_rate_hz
    
    @property
    def history_samples(self) -> int:
        """Number of samples to keep in history."""
        return int(self.history_seconds * self.sample_rate_hz)


@dataclass
class ComponentHealth:
    """Health information for a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_update_time: float = 0.0
    error_count: int = 0
    latency_ms: float = 0.0
    message: str = ""
    
    def update(self, status: HealthStatus, message: str = "") -> None:
        """Update component health status."""
        self.status = status
        self.last_update_time = time.time()
        self.message = message
    
    def mark_error(self, message: str = "") -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.message = message
        if self.error_count > 5:
            self.status = HealthStatus.UNHEALTHY
        elif self.error_count > 2:
            self.status = HealthStatus.DEGRADED
    
    def check_timeout(self, timeout_ms: float) -> bool:
        """Check if component has timed out."""
        if self.last_update_time == 0.0:
            return False
        elapsed_ms = (time.time() - self.last_update_time) * 1000
        if elapsed_ms > timeout_ms:
            self.status = HealthStatus.UNHEALTHY
            self.message = f"Timeout: {elapsed_ms:.1f}ms > {timeout_ms:.1f}ms"
            return True
        return False


@dataclass
class SystemMetrics:
    """
    Aggregated system performance metrics.
    
    Contains real-time measurements and statistics for
    monitoring system health and performance.
    """
    timestamp: float = 0.0
    
    # Latency metrics (milliseconds)
    bci_latency_ms: float = 0.0
    control_latency_ms: float = 0.0
    feedback_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Throughput metrics
    bci_updates_per_sec: float = 0.0
    control_updates_per_sec: float = 0.0
    
    # Quality metrics
    bci_signal_quality: float = 0.0
    decoder_confidence: float = 0.0
    
    # Error metrics
    bci_errors: int = 0
    control_errors: int = 0
    dropped_frames: int = 0
    
    # Resource usage (optional)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    
    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp,
            "bci_latency_ms": self.bci_latency_ms,
            "control_latency_ms": self.control_latency_ms,
            "feedback_latency_ms": self.feedback_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "bci_updates_per_sec": self.bci_updates_per_sec,
            "bci_signal_quality": self.bci_signal_quality,
            "decoder_confidence": self.decoder_confidence,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
        }


class MetricTracker:
    """
    Tracks a single metric over time with statistics.
    
    Maintains a rolling window of metric values and provides
    real-time statistics (mean, std, min, max, percentiles).
    """
    
    def __init__(self, name: str, max_samples: int = 600) -> None:
        """
        Initialize metric tracker.
        
        Args:
            name: Metric name for identification
            max_samples: Maximum samples to keep in history
        """
        self.name = name
        self._values: deque = deque(maxlen=max_samples)
        self._timestamps: deque = deque(maxlen=max_samples)
        self._total = 0.0
        self._count = 0
    
    def record(self, value: float, timestamp: Optional[float] = None) -> None:
        """Record a new metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        self._values.append(value)
        self._timestamps.append(timestamp)
        self._total += value
        self._count += 1
    
    def get_current(self) -> float:
        """Get most recent value."""
        if not self._values:
            return 0.0
        return self._values[-1]
    
    def get_mean(self) -> float:
        """Get mean of all recorded values."""
        if not self._values:
            return 0.0
        return statistics.mean(self._values)
    
    def get_std(self) -> float:
        """Get standard deviation."""
        if len(self._values) < 2:
            return 0.0
        return statistics.stdev(self._values)
    
    def get_min(self) -> float:
        """Get minimum value."""
        if not self._values:
            return 0.0
        return min(self._values)
    
    def get_max(self) -> float:
        """Get maximum value."""
        if not self._values:
            return 0.0
        return max(self._values)
    
    def get_percentile(self, p: float) -> float:
        """Get value at percentile p (0-100)."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]
    
    def get_rate(self, window_seconds: float = 1.0) -> float:
        """Get rate of events per second in recent window."""
        if len(self._timestamps) < 2:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        # Count samples in window
        count = sum(1 for t in self._timestamps if t > cutoff)
        
        return count / window_seconds
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive statistics."""
        return {
            "current": self.get_current(),
            "mean": self.get_mean(),
            "std": self.get_std(),
            "min": self.get_min(),
            "max": self.get_max(),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "count": len(self._values),
        }
    
    def clear(self) -> None:
        """Clear all recorded values."""
        self._values.clear()
        self._timestamps.clear()
        self._total = 0.0
        self._count = 0


class LatencyTimer:
    """
    High-precision timer for measuring latencies.
    
    Provides context manager interface for easy timing of code blocks.
    
    Example:
        with timer.measure("processing"):
            result = process_data(data)
        print(f"Processing took {timer.get('processing')}ms")
    """
    
    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}
        self._measurements: Dict[str, MetricTracker] = {}
    
    def start(self, name: str) -> None:
        """Start timing a named operation."""
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """
        Stop timing and return elapsed time in milliseconds.
        
        Args:
            name: Operation name (must match start call)
            
        Returns:
            Elapsed time in milliseconds
        """
        if name not in self._start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed_ms = (time.perf_counter() - self._start_times[name]) * 1000
        del self._start_times[name]
        
        # Record in tracker
        if name not in self._measurements:
            self._measurements[name] = MetricTracker(name)
        self._measurements[name].record(elapsed_ms)
        
        return elapsed_ms
    
    def measure(self, name: str) -> "TimerContext":
        """
        Return context manager for timing.
        
        Args:
            name: Operation name
            
        Returns:
            Context manager for with statement
        """
        return TimerContext(self, name)
    
    def get(self, name: str) -> float:
        """Get last measurement for named operation."""
        if name in self._measurements:
            return self._measurements[name].get_current()
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for named operation."""
        if name in self._measurements:
            return self._measurements[name].get_stats()
        return {}
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {name: tracker.get_stats() 
                for name, tracker in self._measurements.items()}


class TimerContext:
    """Context manager for LatencyTimer.measure()."""
    
    def __init__(self, timer: LatencyTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> "TimerContext":
        self._timer.start(self._name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = self._timer.stop(self._name)


class SystemMonitor:
    """
    Comprehensive system monitoring for the bionic arm.
    
    Tracks health of all components, measures latencies,
    and provides real-time performance metrics.
    
    Features:
        - Component health tracking with automatic timeout detection
        - Latency measurement for all pipeline stages
        - Rolling statistics for performance analysis
        - Alert callbacks when thresholds exceeded
        - Optional metric logging to file
    
    Example:
        >>> monitor = SystemMonitor(MonitorConfig())
        >>> monitor.start()
        >>> 
        >>> # Record metrics during operation
        >>> monitor.record_latency("bci", 15.5)
        >>> monitor.update_component_health("bci", HealthStatus.HEALTHY)
        >>> 
        >>> # Get current status
        >>> metrics = monitor.get_metrics()
        >>> print(f"Total latency: {metrics.total_latency_ms}ms")
        >>> 
        >>> monitor.stop()
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        """
        Initialize system monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitorConfig()
        
        # Component health tracking
        self._components: Dict[str, ComponentHealth] = {}
        
        # Metric trackers
        self._trackers: Dict[str, MetricTracker] = {
            "bci_latency": MetricTracker("bci_latency", self.config.history_samples),
            "control_latency": MetricTracker("control_latency", self.config.history_samples),
            "feedback_latency": MetricTracker("feedback_latency", self.config.history_samples),
            "total_latency": MetricTracker("total_latency", self.config.history_samples),
            "bci_quality": MetricTracker("bci_quality", self.config.history_samples),
            "confidence": MetricTracker("confidence", self.config.history_samples),
        }
        
        # High-precision timing
        self.timer = LatencyTimer()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, float], None]] = []
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Metrics snapshot
        self._current_metrics = SystemMetrics()
        
        logger.info("SystemMonitor initialized")
    
    def register_component(self, name: str) -> None:
        """Register a component for health tracking."""
        with self._lock:
            self._components[name] = ComponentHealth(name)
            logger.debug(f"Registered component: {name}")
    
    def update_component_health(
        self,
        name: str,
        status: HealthStatus,
        message: str = ""
    ) -> None:
        """
        Update health status of a component.
        
        Args:
            name: Component name
            status: New health status
            message: Optional status message
        """
        with self._lock:
            if name not in self._components:
                self._components[name] = ComponentHealth(name)
            self._components[name].update(status, message)
    
    def record_component_error(self, name: str, message: str = "") -> None:
        """Record an error for a component."""
        with self._lock:
            if name not in self._components:
                self._components[name] = ComponentHealth(name)
            self._components[name].mark_error(message)
    
    def record_latency(self, stage: str, latency_ms: float) -> None:
        """
        Record latency for a pipeline stage.
        
        Args:
            stage: Stage name (bci, control, feedback)
            latency_ms: Latency in milliseconds
        """
        tracker_name = f"{stage}_latency"
        with self._lock:
            if tracker_name in self._trackers:
                self._trackers[tracker_name].record(latency_ms)
                
                # Check threshold
                threshold = self.config.alert_thresholds.get("latency_ms", 150.0)
                if latency_ms > threshold:
                    self._trigger_alert(f"{stage}_latency", latency_ms)
    
    def record_quality(self, metric_name: str, value: float) -> None:
        """Record a quality metric (0-1 scale)."""
        with self._lock:
            if metric_name in self._trackers:
                self._trackers[metric_name].record(value)
    
    def add_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for alert notifications."""
        self._alert_callbacks.append(callback)
    
    def _trigger_alert(self, metric_name: str, value: float) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(metric_name, value)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health information for a component."""
        return self._components.get(name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health information for all components."""
        return dict(self._components)
    
    def is_system_healthy(self) -> bool:
        """Check if all components are healthy."""
        with self._lock:
            for component in self._components.values():
                if component.status in (HealthStatus.UNHEALTHY, HealthStatus.OFFLINE):
                    return False
        return True
    
    def get_metrics(self) -> SystemMetrics:
        """
        Get current system metrics snapshot.
        
        Returns:
            SystemMetrics with current values
        """
        with self._lock:
            return SystemMetrics(
                timestamp=time.time(),
                bci_latency_ms=self._trackers["bci_latency"].get_mean(),
                control_latency_ms=self._trackers["control_latency"].get_mean(),
                feedback_latency_ms=self._trackers["feedback_latency"].get_mean(),
                total_latency_ms=self._trackers["total_latency"].get_mean(),
                bci_signal_quality=self._trackers["bci_quality"].get_mean(),
                decoder_confidence=self._trackers["confidence"].get_mean(),
            )
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get detailed statistics for a metric."""
        with self._lock:
            if metric_name in self._trackers:
                return self._trackers[metric_name].get_stats()
        return {}
    
    def _check_timeouts(self) -> None:
        """Check for component timeouts."""
        with self._lock:
            for name, component in self._components.items():
                timeout = self.config.component_timeouts_ms.get(name, 100.0)
                component.check_timeout(timeout)
    
    def _update_total_latency(self) -> None:
        """Compute and record total end-to-end latency."""
        with self._lock:
            bci = self._trackers["bci_latency"].get_current()
            control = self._trackers["control_latency"].get_current()
            feedback = self._trackers["feedback_latency"].get_current()
            total = bci + control + feedback
            self._trackers["total_latency"].record(total)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring thread."""
        logger.debug("Monitor thread started")
        
        while self._running:
            try:
                self._check_timeouts()
                self._update_total_latency()
                
                # Optional: log metrics to file
                if self.config.enable_logging and self.config.log_path:
                    self._log_metrics()
                
                time.sleep(self.config.sample_period)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
        
        logger.debug("Monitor thread stopped")
    
    def _log_metrics(self) -> None:
        """Log metrics to file."""
        try:
            metrics = self.get_metrics()
            import json
            with open(self.config.log_path, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("SystemMonitor started")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("SystemMonitor stopped")
    
    def reset(self) -> None:
        """Reset all metrics and health status."""
        with self._lock:
            for tracker in self._trackers.values():
                tracker.clear()
            for component in self._components.values():
                component.status = HealthStatus.UNKNOWN
                component.error_count = 0
        logger.info("SystemMonitor reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.
        
        Returns:
            Dictionary with health and metrics summary
        """
        metrics = self.get_metrics()
        
        return {
            "system_healthy": self.is_system_healthy(),
            "components": {
                name: {
                    "status": comp.status.name,
                    "latency_ms": comp.latency_ms,
                    "errors": comp.error_count,
                    "message": comp.message,
                }
                for name, comp in self._components.items()
            },
            "metrics": metrics.to_dict(),
            "latency_stats": {
                "bci": self.get_metric_stats("bci_latency"),
                "control": self.get_metric_stats("control_latency"),
                "total": self.get_metric_stats("total_latency"),
            },
        }
