"""
Visual Feedback Module
======================

Real-time visual feedback display for monitoring bionic arm state,
BCI confidence levels, and force/position information.

Display Components:
    - Arm pose visualization (2D/3D)
    - Grip force gauge
    - BCI confidence meter
    - Command intent display
    - Status indicators

Rendering Options:
    - Matplotlib-based display
    - OpenCV-based overlay
    - Terminal-based ASCII (fallback)

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Visual display rendering modes."""
    MATPLOTLIB = auto()   # Full matplotlib window
    OPENCV = auto()       # OpenCV overlay
    TERMINAL = auto()     # ASCII terminal display
    HEADLESS = auto()     # No display (logging only)


class StatusLevel(Enum):
    """Status indicator levels."""
    OK = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class VisualFeedbackConfig:
    """
    Configuration for visual feedback display.
    
    Attributes:
        display_mode: Rendering backend to use
        update_rate_hz: Display refresh rate
        window_size: (width, height) in pixels
        show_arm_pose: Enable arm visualization
        show_force_gauge: Enable force display
        show_bci_confidence: Enable BCI meter
        show_intent: Enable command intent display
        history_seconds: Duration of history to display
        colormap: Color scheme for visualizations
    """
    display_mode: DisplayMode = DisplayMode.MATPLOTLIB
    update_rate_hz: float = 30.0
    window_size: Tuple[int, int] = (800, 600)
    show_arm_pose: bool = True
    show_force_gauge: bool = True
    show_bci_confidence: bool = True
    show_intent: bool = True
    history_seconds: float = 5.0
    colormap: str = "viridis"
    
    # Layout
    arm_display_region: Tuple[float, float, float, float] = (0.0, 0.4, 0.6, 0.6)
    gauge_region: Tuple[float, float, float, float] = (0.6, 0.4, 0.4, 0.6)
    status_region: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.4)
    
    @property
    def update_period(self) -> float:
        """Time between display updates in seconds."""
        return 1.0 / self.update_rate_hz
    
    @property
    def history_samples(self) -> int:
        """Number of samples to keep in history."""
        return int(self.history_seconds * self.update_rate_hz)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be positive")
        if self.window_size[0] <= 0 or self.window_size[1] <= 0:
            raise ValueError("window_size dimensions must be positive")
        if self.history_seconds <= 0:
            raise ValueError("history_seconds must be positive")


@dataclass
class ArmState:
    """Current state of the arm for visualization."""
    joint_angles: NDArray = field(default_factory=lambda: np.zeros(7))
    joint_velocities: NDArray = field(default_factory=lambda: np.zeros(7))
    end_effector_pos: NDArray = field(default_factory=lambda: np.zeros(3))
    end_effector_orient: NDArray = field(default_factory=lambda: np.eye(3))
    grip_aperture: float = 0.0  # 0=closed, 1=open
    timestamp_ns: int = 0
    
    def copy(self) -> 'ArmState':
        """Create a copy of the state."""
        return ArmState(
            joint_angles=self.joint_angles.copy(),
            joint_velocities=self.joint_velocities.copy(),
            end_effector_pos=self.end_effector_pos.copy(),
            end_effector_orient=self.end_effector_orient.copy(),
            grip_aperture=self.grip_aperture,
            timestamp_ns=self.timestamp_ns
        )


@dataclass
class BCIState:
    """Current BCI system state for visualization."""
    confidence: float = 0.0  # 0-1
    intent_class: int = -1
    intent_name: str = "none"
    velocity_command: NDArray = field(default_factory=lambda: np.zeros(3))
    signal_quality: float = 1.0  # 0-1
    artifacts_detected: bool = False
    timestamp_ns: int = 0


@dataclass
class ForceState:
    """Current force state for visualization."""
    finger_forces: NDArray = field(default_factory=lambda: np.zeros(5))
    total_grip_force: float = 0.0
    max_force: float = 30.0  # N
    slip_detected: bool = False
    contact_active: bool = False
    timestamp_ns: int = 0


class HistoryBuffer:
    """
    Circular buffer for storing time-series data.
    
    Thread-safe storage with fixed capacity.
    """
    
    def __init__(self, max_samples: int, n_channels: int = 1) -> None:
        self.max_samples = max_samples
        self.n_channels = n_channels
        self._data = np.zeros((max_samples, n_channels))
        self._timestamps = np.zeros(max_samples)
        self._write_idx = 0
        self._count = 0
        self._lock = threading.Lock()
    
    def push(self, values: NDArray, timestamp: float) -> None:
        """Add new values to buffer."""
        with self._lock:
            self._data[self._write_idx] = values
            self._timestamps[self._write_idx] = timestamp
            self._write_idx = (self._write_idx + 1) % self.max_samples
            self._count = min(self._count + 1, self.max_samples)
    
    def get_history(self) -> Tuple[NDArray, NDArray]:
        """Get ordered history (oldest to newest)."""
        with self._lock:
            if self._count < self.max_samples:
                return (
                    self._data[:self._count].copy(),
                    self._timestamps[:self._count].copy()
                )
            # Full buffer - reorder
            start = self._write_idx
            return (
                np.roll(self._data, -start, axis=0).copy(),
                np.roll(self._timestamps, -start).copy()
            )
    
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._data.fill(0)
            self._timestamps.fill(0)
            self._write_idx = 0
            self._count = 0


class StatusIndicator:
    """
    Status indicator with history and state tracking.
    
    Tracks system status with timestamps for logging.
    """
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._level = StatusLevel.OK
        self._message = ""
        self._last_update = time.time()
        self._history: List[Tuple[float, StatusLevel, str]] = []
        self._max_history = 100
    
    @property
    def level(self) -> StatusLevel:
        return self._level
    
    @property
    def message(self) -> str:
        return self._message
    
    def set_status(self, level: StatusLevel, message: str = "") -> None:
        """Update status."""
        now = time.time()
        if level != self._level or message != self._message:
            self._history.append((now, level, message))
            if len(self._history) > self._max_history:
                self._history.pop(0)
        
        self._level = level
        self._message = message
        self._last_update = now
    
    def get_color(self) -> Tuple[float, float, float]:
        """Get RGB color for current status level."""
        colors = {
            StatusLevel.OK: (0.2, 0.8, 0.2),       # Green
            StatusLevel.WARNING: (0.9, 0.7, 0.1),  # Yellow
            StatusLevel.ERROR: (0.9, 0.3, 0.1),    # Orange
            StatusLevel.CRITICAL: (0.9, 0.1, 0.1)  # Red
        }
        return colors.get(self._level, (0.5, 0.5, 0.5))


class ArmVisualizer:
    """
    2D visualization of arm pose.
    
    Renders simplified stick-figure representation
    of arm joint angles and end-effector position.
    """
    
    def __init__(self, link_lengths: Optional[List[float]] = None) -> None:
        # Default 7-DOF arm link lengths (mm -> normalized)
        self.link_lengths = link_lengths or [0.1, 0.28, 0.26, 0.1, 0.1, 0.05, 0.05]
        self._joint_colors = [
            (0.2, 0.4, 0.8),  # Shoulder
            (0.3, 0.5, 0.8),
            (0.4, 0.6, 0.8),  # Elbow
            (0.5, 0.7, 0.8),
            (0.6, 0.8, 0.8),  # Wrist
            (0.7, 0.8, 0.7),
            (0.8, 0.8, 0.6)   # End effector
        ]
    
    def compute_joint_positions(
        self, 
        joint_angles: NDArray
    ) -> List[Tuple[float, float]]:
        """
        Compute 2D joint positions for visualization.
        
        Uses simplified 2D projection (XY plane).
        
        Args:
            joint_angles: 7 joint angles in radians
            
        Returns:
            List of (x, y) positions for each joint
        """
        positions = [(0.0, 0.0)]  # Base
        x, y = 0.0, 0.0
        cumulative_angle = 0.0
        
        for i, (length, angle) in enumerate(zip(self.link_lengths, joint_angles)):
            # Simplified 2D: alternate joints contribute to XY
            if i % 2 == 0:  # Contributes to angle
                cumulative_angle += angle
            
            dx = length * np.cos(cumulative_angle)
            dy = length * np.sin(cumulative_angle)
            x += dx
            y += dy
            positions.append((x, y))
        
        return positions
    
    def render_to_array(
        self, 
        joint_angles: NDArray,
        width: int = 400,
        height: int = 400
    ) -> NDArray:
        """
        Render arm to image array.
        
        Args:
            joint_angles: Current joint angles
            width: Image width
            height: Image height
            
        Returns:
            RGB image array (height, width, 3)
        """
        # Create blank canvas
        image = np.ones((height, width, 3), dtype=np.float32) * 0.1
        
        # Compute positions
        positions = self.compute_joint_positions(joint_angles)
        
        # Scale and center
        scale = min(width, height) * 0.4
        cx, cy = width // 2, height // 2
        
        # Draw links
        for i in range(len(positions) - 1):
            x1 = int(cx + positions[i][0] * scale)
            y1 = int(cy - positions[i][1] * scale)  # Flip Y
            x2 = int(cx + positions[i + 1][0] * scale)
            y2 = int(cy - positions[i + 1][1] * scale)
            
            # Simple line drawing (Bresenham would be better)
            self._draw_line(image, x1, y1, x2, y2, self._joint_colors[i])
        
        # Draw joints
        for i, (x, y) in enumerate(positions):
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            radius = 8 if i == 0 or i == len(positions) - 1 else 5
            self._draw_circle(image, px, py, radius, self._joint_colors[min(i, 6)])
        
        return image
    
    def _draw_line(
        self,
        image: NDArray,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Tuple[float, float, float],
        thickness: int = 3
    ) -> None:
        """Draw a line on the image."""
        h, w = image.shape[:2]
        
        # Simple line rasterization
        n_steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for t in np.linspace(0, 1, n_steps * 2):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            for dy in range(-thickness // 2, thickness // 2 + 1):
                for dx in range(-thickness // 2, thickness // 2 + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < w and 0 <= py < h:
                        image[py, px] = color
    
    def _draw_circle(
        self,
        image: NDArray,
        cx: int, cy: int,
        radius: int,
        color: Tuple[float, float, float]
    ) -> None:
        """Draw a filled circle on the image."""
        h, w = image.shape[:2]
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        image[py, px] = color


class ForceGauge:
    """
    Gauge visualization for force display.
    
    Renders a bar or arc gauge showing current force
    relative to maximum.
    """
    
    def __init__(
        self,
        max_force: float = 30.0,
        warning_threshold: float = 0.7,
        danger_threshold: float = 0.9
    ) -> None:
        self.max_force = max_force
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
    
    def render_bar(
        self,
        force: float,
        width: int = 100,
        height: int = 300
    ) -> NDArray:
        """
        Render vertical bar gauge.
        
        Args:
            force: Current force in Newtons
            width: Gauge width
            height: Gauge height
            
        Returns:
            RGB image array
        """
        image = np.ones((height, width, 3), dtype=np.float32) * 0.15
        
        # Compute fill level
        fill_ratio = np.clip(force / self.max_force, 0, 1)
        fill_height = int(fill_ratio * (height - 20))
        
        # Determine color
        if fill_ratio >= self.danger_threshold:
            color = (0.9, 0.1, 0.1)  # Red
        elif fill_ratio >= self.warning_threshold:
            color = (0.9, 0.7, 0.1)  # Yellow
        else:
            color = (0.2, 0.7, 0.3)  # Green
        
        # Draw bar
        margin = 10
        bar_start_y = height - margin - fill_height
        bar_end_y = height - margin
        
        for y in range(bar_start_y, bar_end_y):
            for x in range(margin, width - margin):
                image[y, x] = color
        
        # Draw outline
        outline_color = (0.4, 0.4, 0.4)
        for x in range(margin, width - margin):
            image[margin, x] = outline_color
            image[height - margin - 1, x] = outline_color
        for y in range(margin, height - margin):
            image[y, margin] = outline_color
            image[y, width - margin - 1] = outline_color
        
        return image


class BCIConfidenceMeter:
    """
    Visual meter for BCI confidence display.
    
    Shows classification confidence and signal quality.
    """
    
    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
    
    def render(
        self,
        confidence: float,
        signal_quality: float,
        intent_name: str,
        width: int = 200,
        height: int = 150
    ) -> NDArray:
        """
        Render confidence meter.
        
        Args:
            confidence: Classification confidence (0-1)
            signal_quality: Signal quality (0-1)
            intent_name: Current intent label
            width: Display width
            height: Display height
            
        Returns:
            RGB image array
        """
        image = np.ones((height, width, 3), dtype=np.float32) * 0.12
        
        # Confidence arc
        arc_cx, arc_cy = width // 2, height - 30
        arc_radius = min(width, height) // 3
        
        # Draw confidence arc
        for angle_deg in range(-140, -40):
            angle_rad = np.radians(angle_deg)
            normalized_angle = (angle_deg + 140) / 100  # 0 to 1
            
            if normalized_angle <= confidence:
                if confidence >= self.confidence_threshold:
                    color = (0.2, 0.8, 0.3)  # Green
                else:
                    color = (0.9, 0.6, 0.1)  # Orange
            else:
                color = (0.25, 0.25, 0.25)  # Dark gray
            
            for r in range(arc_radius - 15, arc_radius):
                x = int(arc_cx + r * np.cos(angle_rad))
                y = int(arc_cy + r * np.sin(angle_rad))
                if 0 <= x < width and 0 <= y < height:
                    image[y, x] = color
        
        # Signal quality indicator (small bar at top)
        sq_width = int((width - 40) * signal_quality)
        sq_color = (0.2, 0.7, 0.3) if signal_quality > 0.5 else (0.8, 0.3, 0.2)
        for x in range(20, 20 + sq_width):
            for y in range(10, 20):
                if x < width:
                    image[y, x] = sq_color
        
        return image


class VisualFeedback:
    """
    Main visual feedback controller.
    
    Manages display rendering and state updates for
    arm visualization, force gauges, and BCI status.
    
    Example:
        >>> config = VisualFeedbackConfig(display_mode=DisplayMode.MATPLOTLIB)
        >>> feedback = VisualFeedback(config)
        >>> feedback.start()
        >>> 
        >>> # Update from system
        >>> feedback.update_arm_state(arm_state)
        >>> feedback.update_bci_state(bci_state)
        >>> feedback.update_force_state(force_state)
    """
    
    def __init__(
        self,
        config: Optional[VisualFeedbackConfig] = None
    ) -> None:
        self.config = config or VisualFeedbackConfig()
        
        # Visualizers
        self._arm_viz = ArmVisualizer()
        self._force_gauge = ForceGauge()
        self._bci_meter = BCIConfidenceMeter()
        
        # State
        self._arm_state = ArmState()
        self._bci_state = BCIState()
        self._force_state = ForceState()
        self._state_lock = threading.Lock()
        
        # History buffers
        n_history = self.config.history_samples
        self._confidence_history = HistoryBuffer(n_history, 1)
        self._force_history = HistoryBuffer(n_history, 1)
        self._velocity_history = HistoryBuffer(n_history, 3)
        
        # Status indicators
        self._indicators: Dict[str, StatusIndicator] = {
            "bci": StatusIndicator("BCI System"),
            "arm": StatusIndicator("Arm Control"),
            "safety": StatusIndicator("Safety"),
            "comm": StatusIndicator("Communication")
        }
        
        # Display backend
        self._display = None
        self._figure = None
        self._axes = None
        
        # Threading
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self._on_update_callbacks: List[Callable[[NDArray], None]] = []
        
        # Timing metrics
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        logger.info(f"VisualFeedback initialized: {self.config.display_mode.name}")
    
    def start(self) -> None:
        """Start visual feedback display."""
        if self._running:
            logger.warning("Visual feedback already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        # Initialize display backend
        self._init_display()
        
        # Start update thread
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="VisualFeedback"
        )
        self._update_thread.start()
        
        logger.info("Visual feedback started")
    
    def stop(self) -> None:
        """Stop visual feedback display."""
        self._running = False
        self._stop_event.set()
        
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        
        self._cleanup_display()
        logger.info("Visual feedback stopped")
    
    def _init_display(self) -> None:
        """Initialize display backend."""
        if self.config.display_mode == DisplayMode.MATPLOTLIB:
            try:
                import matplotlib
                matplotlib.use('TkAgg')  # Or 'Qt5Agg'
                import matplotlib.pyplot as plt
                
                self._figure, axes = plt.subplots(2, 2, figsize=(10, 8))
                self._axes = {
                    'arm': axes[0, 0],
                    'force': axes[0, 1],
                    'bci': axes[1, 0],
                    'status': axes[1, 1]
                }
                plt.ion()
                self._figure.show()
                logger.info("Matplotlib display initialized")
            except ImportError:
                logger.warning("Matplotlib not available, falling back to headless")
                self.config.display_mode = DisplayMode.HEADLESS
        
        elif self.config.display_mode == DisplayMode.TERMINAL:
            logger.info("Terminal display mode active")
    
    def _cleanup_display(self) -> None:
        """Cleanup display resources."""
        if self._figure is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._figure)
            except:
                pass
            self._figure = None
            self._axes = None
    
    def update_arm_state(self, state: ArmState) -> None:
        """Update arm state for visualization."""
        with self._state_lock:
            self._arm_state = state.copy()
    
    def update_bci_state(self, state: BCIState) -> None:
        """Update BCI state for visualization."""
        with self._state_lock:
            self._bci_state = state
            self._confidence_history.push(
                np.array([state.confidence]),
                time.time()
            )
            self._velocity_history.push(
                state.velocity_command,
                time.time()
            )
    
    def update_force_state(self, state: ForceState) -> None:
        """Update force state for visualization."""
        with self._state_lock:
            self._force_state = state
            self._force_history.push(
                np.array([state.total_grip_force]),
                time.time()
            )
    
    def set_status(
        self,
        indicator_name: str,
        level: StatusLevel,
        message: str = ""
    ) -> None:
        """Update a status indicator."""
        if indicator_name in self._indicators:
            self._indicators[indicator_name].set_status(level, message)
    
    def add_update_callback(
        self,
        callback: Callable[[NDArray], None]
    ) -> None:
        """Register callback for frame updates."""
        self._on_update_callbacks.append(callback)
    
    def _update_loop(self) -> None:
        """Background display update loop."""
        period = self.config.update_period
        
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            
            try:
                self._render_frame()
            except Exception as e:
                logger.error(f"Render error: {e}")
            
            # FPS tracking
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._current_fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now
            
            # Maintain frame rate
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _render_frame(self) -> None:
        """Render single frame."""
        with self._state_lock:
            arm_state = self._arm_state
            bci_state = self._bci_state
            force_state = self._force_state
        
        if self.config.display_mode == DisplayMode.MATPLOTLIB:
            self._render_matplotlib(arm_state, bci_state, force_state)
        elif self.config.display_mode == DisplayMode.TERMINAL:
            self._render_terminal(arm_state, bci_state, force_state)
        elif self.config.display_mode == DisplayMode.HEADLESS:
            pass  # No display
    
    def _render_matplotlib(
        self,
        arm_state: ArmState,
        bci_state: BCIState,
        force_state: ForceState
    ) -> None:
        """Render using matplotlib."""
        if self._axes is None:
            return
        
        try:
            # Clear axes
            for ax in self._axes.values():
                ax.clear()
            
            # Arm visualization
            ax = self._axes['arm']
            positions = self._arm_viz.compute_joint_positions(arm_state.joint_angles)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            ax.plot(xs, ys, 'o-', linewidth=3, markersize=8, color='#4A90D9')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.5, 1.5)
            ax.set_title('Arm Pose')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Force gauge
            ax = self._axes['force']
            force_data, timestamps = self._force_history.get_history()
            if len(timestamps) > 0:
                t_rel = timestamps - timestamps[0]
                ax.fill_between(t_rel, force_data[:, 0], alpha=0.4, color='#4CAF50')
                ax.plot(t_rel, force_data[:, 0], color='#2E7D32', linewidth=2)
            ax.set_ylim(0, force_state.max_force)
            ax.set_title(f'Grip Force: {force_state.total_grip_force:.1f} N')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)')
            ax.grid(True, alpha=0.3)
            
            # BCI confidence
            ax = self._axes['bci']
            conf_data, conf_ts = self._confidence_history.get_history()
            if len(conf_ts) > 0:
                t_rel = conf_ts - conf_ts[0]
                ax.fill_between(t_rel, conf_data[:, 0], alpha=0.4, color='#9C27B0')
                ax.plot(t_rel, conf_data[:, 0], color='#6A1B9A', linewidth=2)
            ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Threshold')
            ax.set_ylim(0, 1)
            ax.set_title(f'BCI Confidence | Intent: {bci_state.intent_name}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Confidence')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Status
            ax = self._axes['status']
            y_pos = 0.9
            for name, indicator in self._indicators.items():
                color = indicator.get_color()
                ax.text(0.1, y_pos, f"● {name.upper()}", color=color, fontsize=12, fontweight='bold')
                ax.text(0.4, y_pos, indicator.message or "OK", color='white', fontsize=10)
                y_pos -= 0.2
            ax.text(0.7, 0.1, f"FPS: {self._current_fps:.1f}", color='gray', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor('#1E1E1E')
            ax.set_title('System Status')
            ax.axis('off')
            
            # Update
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            
        except Exception as e:
            logger.debug(f"Matplotlib render issue: {e}")
    
    def _render_terminal(
        self,
        arm_state: ArmState,
        bci_state: BCIState,
        force_state: ForceState
    ) -> None:
        """Render ASCII display to terminal."""
        # Simple ASCII display
        force_bar_len = 20
        force_filled = int((force_state.total_grip_force / force_state.max_force) * force_bar_len)
        force_bar = '█' * force_filled + '░' * (force_bar_len - force_filled)
        
        conf_bar_len = 20
        conf_filled = int(bci_state.confidence * conf_bar_len)
        conf_bar = '█' * conf_filled + '░' * (conf_bar_len - conf_filled)
        
        output = (
            f"\033[2J\033[H"  # Clear screen
            f"╔══════════════════════════════════════════╗\n"
            f"║         BIONIC ARM VISUAL FEEDBACK       ║\n"
            f"╠══════════════════════════════════════════╣\n"
            f"║ Force:  [{force_bar}] {force_state.total_grip_force:5.1f}N ║\n"
            f"║ BCI:    [{conf_bar}] {bci_state.confidence:5.1%} ║\n"
            f"║ Intent: {bci_state.intent_name:32} ║\n"
            f"║ FPS:    {self._current_fps:5.1f}                          ║\n"
            f"╚══════════════════════════════════════════╝"
        )
        print(output, end='', flush=True)
    
    def render_composite_image(self) -> NDArray:
        """
        Render all visualizations to a single composite image.
        
        Returns:
            RGB image array (height, width, 3)
        """
        width, height = self.config.window_size
        image = np.ones((height, width, 3), dtype=np.float32) * 0.1
        
        with self._state_lock:
            arm_state = self._arm_state
            bci_state = self._bci_state
            force_state = self._force_state
        
        # Render components
        arm_img = self._arm_viz.render_to_array(
            arm_state.joint_angles, 
            width // 2, 
            height // 2
        )
        
        force_img = self._force_gauge.render_bar(
            force_state.total_grip_force,
            width // 4,
            height // 2
        )
        
        bci_img = self._bci_meter.render(
            bci_state.confidence,
            bci_state.signal_quality,
            bci_state.intent_name,
            width // 4,
            height // 4
        )
        
        # Composite
        image[0:height//2, 0:width//2] = arm_img
        image[0:height//2, width//2:width//2+width//4] = force_img
        image[height//2:height//2+height//4, 0:width//4] = bci_img
        
        return image
    
    @property
    def is_running(self) -> bool:
        """Check if display is running."""
        return self._running
    
    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._current_fps


# Convenience function
def create_visual_feedback(
    mode: str = "matplotlib",
    **kwargs
) -> VisualFeedback:
    """
    Create visual feedback with specified mode.
    
    Args:
        mode: Display mode ("matplotlib", "opencv", "terminal", "headless")
        **kwargs: Additional config options
        
    Returns:
        Configured VisualFeedback instance
    """
    mode_map = {
        "matplotlib": DisplayMode.MATPLOTLIB,
        "opencv": DisplayMode.OPENCV,
        "terminal": DisplayMode.TERMINAL,
        "headless": DisplayMode.HEADLESS
    }
    
    display_mode = mode_map.get(mode.lower(), DisplayMode.MATPLOTLIB)
    config = VisualFeedbackConfig(display_mode=display_mode, **kwargs)
    
    return VisualFeedback(config)
