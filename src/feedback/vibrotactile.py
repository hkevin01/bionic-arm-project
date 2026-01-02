"""
Vibrotactile Feedback Module
============================

Haptic feedback using vibration patterns to convey sensory information
from the prosthetic hand to the user.

Encoding Strategies:
    - Force encoding: Vibration intensity proportional to grip force
    - Slip encoding: Rapid burst patterns indicate object slipping
    - Contact encoding: Pattern onset indicates object contact
    - Texture encoding: Frequency modulation for surface texture

Hardware:
    - Vibration motors (ERM or LRA)
    - Placement on residual limb or body surface
    - Typically 1-6 tactors

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VibrationPattern(Enum):
    """Predefined vibration patterns."""
    CONSTANT = auto()      # Steady vibration
    PULSE = auto()         # Single pulse
    DOUBLE_PULSE = auto()  # Double tap
    RAMP_UP = auto()       # Increasing intensity
    RAMP_DOWN = auto()     # Decreasing intensity
    SINE = auto()          # Sinusoidal modulation
    SLIP = auto()          # Rapid pulses (slip warning)
    CONTACT = auto()       # Sharp onset (contact)


@dataclass
class VibrotactileConfig:
    """
    Configuration for vibrotactile feedback.
    
    Attributes:
        n_tactors: Number of vibration motors
        min_intensity: Minimum intensity (0-1)
        max_intensity: Maximum intensity (0-1)
        frequency_range: (min_hz, max_hz) for frequency modulation
        update_rate_hz: Feedback update rate
        force_scaling: Scaling from force (N) to intensity
    """
    n_tactors: int = 2
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    frequency_range: tuple = (50.0, 250.0)
    update_rate_hz: float = 100.0
    force_scaling: float = 0.05
    
    @property
    def update_period(self) -> float:
        return 1.0 / self.update_rate_hz


@dataclass
class VibrationCommand:
    """Command for a single tactor."""
    tactor_id: int
    intensity: float = 0.0
    frequency: float = 150.0
    duration_ms: float = 0.0  # 0 = continuous
    
    def __post_init__(self):
        self.intensity = np.clip(self.intensity, 0.0, 1.0)


class VibrotactileEncoder:
    """
    Encodes sensory information as vibration patterns.
    
    Transforms force, slip, and contact signals into appropriate
    vibration commands for each tactor.
    """
    
    def __init__(self, config: VibrotactileConfig) -> None:
        self.config = config
        self._slip_threshold = 0.1
        self._contact_threshold = 0.5
        
    def encode_force(self, forces: NDArray) -> List[VibrationCommand]:
        """
        Encode grip forces as vibration intensity.
        
        Args:
            forces: Force per finger (N)
            
        Returns:
            Vibration commands
        """
        commands = []
        total_force = np.sum(np.abs(forces))
        
        # Map force to intensity
        intensity = np.clip(
            total_force * self.config.force_scaling,
            self.config.min_intensity,
            self.config.max_intensity
        )
        
        # Distribute across tactors
        for i in range(self.config.n_tactors):
            commands.append(VibrationCommand(
                tactor_id=i,
                intensity=intensity,
                frequency=self._force_to_frequency(total_force)
            ))
        
        return commands
    
    def encode_slip(self) -> List[VibrationCommand]:
        """Generate slip warning pattern."""
        commands = []
        for i in range(self.config.n_tactors):
            commands.append(VibrationCommand(
                tactor_id=i,
                intensity=self.config.max_intensity,
                frequency=self.config.frequency_range[1],
                duration_ms=50
            ))
        return commands
    
    def encode_contact(self) -> List[VibrationCommand]:
        """Generate contact notification pattern."""
        commands = []
        for i in range(self.config.n_tactors):
            commands.append(VibrationCommand(
                tactor_id=i,
                intensity=0.7,
                frequency=150.0,
                duration_ms=100
            ))
        return commands
    
    def _force_to_frequency(self, force: float) -> float:
        """Map force to vibration frequency."""
        f_min, f_max = self.config.frequency_range
        # Logarithmic mapping
        normalized = np.clip(force * self.config.force_scaling, 0, 1)
        return f_min + (f_max - f_min) * normalized


class VibrotactileFeedback:
    """
    Main vibrotactile feedback controller.
    
    Manages vibration motors and pattern generation.
    
    Example:
        >>> config = VibrotactileConfig(n_tactors=2)
        >>> feedback = VibrotactileFeedback(config)
        >>> feedback.start()
        >>> 
        >>> # From grasp controller
        >>> feedback.set_force(grip_force)
        >>> feedback.notify_contact()
    """
    
    def __init__(
        self,
        config: Optional[VibrotactileConfig] = None,
        hardware_interface: Optional[Any] = None
    ) -> None:
        self.config = config or VibrotactileConfig()
        self.encoder = VibrotactileEncoder(self.config)
        self._hardware = hardware_interface
        
        # State
        self._running = False
        self._current_intensities = np.zeros(self.config.n_tactors)
        self._current_forces = np.zeros(5)  # Per finger
        
        # Pattern queue
        self._pattern_queue: List[VibrationCommand] = []
        self._pattern_lock = threading.Lock()
        
        # Update thread
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info(f"VibrotactileFeedback: {self.config.n_tactors} tactors")
    
    def start(self) -> None:
        """Start feedback system."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self._update_thread.start()
        logger.info("Vibrotactile feedback started")
    
    def stop(self) -> None:
        """Stop feedback system."""
        self._running = False
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        
        # Turn off all tactors
        self._current_intensities = np.zeros(self.config.n_tactors)
        self._send_to_hardware()
        logger.info("Vibrotactile feedback stopped")
    
    def set_force(self, forces: NDArray) -> None:
        """Update force feedback."""
        self._current_forces = np.asarray(forces).flatten()
    
    def notify_contact(self) -> None:
        """Trigger contact notification."""
        with self._pattern_lock:
            self._pattern_queue.extend(self.encoder.encode_contact())
    
    def notify_slip(self) -> None:
        """Trigger slip warning."""
        with self._pattern_lock:
            self._pattern_queue.extend(self.encoder.encode_slip())
    
    def play_pattern(self, pattern: VibrationPattern, intensity: float = 0.7) -> None:
        """Play a predefined pattern."""
        # Generate pattern commands
        if pattern == VibrationPattern.PULSE:
            for i in range(self.config.n_tactors):
                self._pattern_queue.append(VibrationCommand(
                    tactor_id=i, intensity=intensity, duration_ms=100
                ))
        elif pattern == VibrationPattern.DOUBLE_PULSE:
            for _ in range(2):
                for i in range(self.config.n_tactors):
                    self._pattern_queue.append(VibrationCommand(
                        tactor_id=i, intensity=intensity, duration_ms=50
                    ))
    
    def _update_loop(self) -> None:
        """Background update loop."""
        period = self.config.update_period
        
        while not self._stop_event.is_set():
            # Process queued patterns
            with self._pattern_lock:
                if self._pattern_queue:
                    cmd = self._pattern_queue.pop(0)
                    self._current_intensities[cmd.tactor_id] = cmd.intensity
            
            # Update from force encoding
            commands = self.encoder.encode_force(self._current_forces)
            for cmd in commands:
                self._current_intensities[cmd.tactor_id] = cmd.intensity
            
            # Send to hardware
            self._send_to_hardware()
            
            time.sleep(period)
    
    def _send_to_hardware(self) -> None:
        """Send current state to hardware."""
        if self._hardware is not None:
            self._hardware.set_intensities(self._current_intensities)
        # In simulation, just log
        else:
            pass  # Would log or simulate
