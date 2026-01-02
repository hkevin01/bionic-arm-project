"""
Grasping Module
===============

Grasp primitives and force control for prosthetic hand operation.

Grasp Types:
    - Power Grasp: Full hand closure for large objects
    - Precision Grasp: Thumb-finger pinch for small objects
    - Lateral Grasp: Key grip between thumb and index side
    - Hook Grasp: Fingers curled, thumb neutral (carrying bags)
    - Spherical Grasp: All fingers curved around spherical object
    - Cylindrical Grasp: Fingers wrapped around cylinder

Force Control:
    - Proportional force feedback from BCI confidence
    - Slip detection and automatic grip tightening
    - Compliance control for fragile objects
    - Force limits for safety

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]


# =============================================================================
# Enums and Configuration
# =============================================================================

class GraspType(Enum):
    """Standard grasp types based on Cutkosky taxonomy."""
    POWER = auto()          # Full hand power grasp
    PRECISION = auto()      # Thumb-index pinch
    LATERAL = auto()        # Key grip (thumb against index side)
    HOOK = auto()           # Hook grip (fingers only)
    SPHERICAL = auto()      # Spherical power grasp
    CYLINDRICAL = auto()    # Cylindrical power grasp
    TRIPOD = auto()         # Thumb, index, middle pinch
    TIP = auto()            # Fingertip precision grasp
    OPEN = auto()           # Fully open hand


class GraspPhase(Enum):
    """Phases of grasp execution."""
    IDLE = auto()
    PRESHAPE = auto()       # Moving to preshape position
    APPROACH = auto()       # Approaching object
    CLOSE = auto()          # Closing fingers
    HOLD = auto()           # Maintaining grasp force
    RELEASE = auto()        # Opening hand


@dataclass
class GraspConfig:
    """
    Configuration for grasp controller.
    
    Attributes:
        n_fingers: Number of controllable fingers (including thumb)
        force_min: Minimum grip force (N)
        force_max: Maximum grip force (N)
        force_default: Default grip force (N)
        close_velocity: Finger closing velocity (rad/s)
        slip_threshold: Slip detection sensitivity
        compliance: Compliance level (0=stiff, 1=compliant)
    """
    n_fingers: int = 5
    force_min: float = 0.5
    force_max: float = 30.0
    force_default: float = 5.0
    close_velocity: float = 1.0
    slip_threshold: float = 0.1
    compliance: float = 0.3
    
    # Finger joint configuration
    # [thumb_mcp, thumb_ip, index_mcp, index_pip, ...]
    joints_per_finger: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    
    @property
    def total_joints(self) -> int:
        """Total number of finger joints."""
        return sum(self.joints_per_finger)


@dataclass
class GraspPrimitive:
    """
    Definition of a grasp primitive.
    
    Specifies the target finger positions and forces for a grasp type.
    
    Attributes:
        grasp_type: Type of grasp
        preshape: Finger positions for preshape (before contact)
        final_shape: Target finger positions (at contact)
        force_distribution: Force ratio for each finger (sums to 1)
        approach_direction: Direction to approach object (unit vector)
    """
    grasp_type: GraspType
    preshape: FloatArray
    final_shape: FloatArray
    force_distribution: FloatArray
    approach_direction: FloatArray = field(default_factory=lambda: np.array([0, 0, -1]))
    
    def __post_init__(self) -> None:
        """Convert to numpy arrays and normalize."""
        self.preshape = np.asarray(self.preshape)
        self.final_shape = np.asarray(self.final_shape)
        self.force_distribution = np.asarray(self.force_distribution)
        self.approach_direction = np.asarray(self.approach_direction)
        
        # Normalize force distribution
        total = np.sum(self.force_distribution)
        if total > 0:
            self.force_distribution = self.force_distribution / total
    
    @classmethod
    def power_grasp(cls, n_joints: int = 10) -> "GraspPrimitive":
        """Create a power grasp primitive."""
        return cls(
            grasp_type=GraspType.POWER,
            preshape=np.ones(n_joints) * 0.3,
            final_shape=np.ones(n_joints) * 1.5,
            force_distribution=np.ones(5) / 5,
            approach_direction=np.array([0, 0, -1])
        )
    
    @classmethod
    def precision_grasp(cls, n_joints: int = 10) -> "GraspPrimitive":
        """Create a precision (pinch) grasp primitive."""
        preshape = np.zeros(n_joints)
        final_shape = np.zeros(n_joints)
        
        # Thumb and index finger only
        preshape[0:2] = 0.5   # Thumb
        preshape[2:4] = 0.5   # Index
        final_shape[0:2] = 1.0
        final_shape[2:4] = 1.0
        
        force_dist = np.array([0.5, 0.5, 0, 0, 0])
        
        return cls(
            grasp_type=GraspType.PRECISION,
            preshape=preshape,
            final_shape=final_shape,
            force_distribution=force_dist,
            approach_direction=np.array([0, 0, -1])
        )
    
    @classmethod
    def lateral_grasp(cls, n_joints: int = 10) -> "GraspPrimitive":
        """Create a lateral (key) grasp primitive."""
        preshape = np.zeros(n_joints)
        final_shape = np.zeros(n_joints)
        
        # Thumb adducted, index slightly flexed
        preshape[0:2] = [0.3, 0.0]  # Thumb
        preshape[2:4] = [0.5, 0.3]  # Index
        final_shape[0:2] = [0.8, 0.0]
        final_shape[2:4] = [0.6, 0.4]
        
        force_dist = np.array([0.6, 0.4, 0, 0, 0])
        
        return cls(
            grasp_type=GraspType.LATERAL,
            preshape=preshape,
            final_shape=final_shape,
            force_distribution=force_dist,
            approach_direction=np.array([0, -1, 0])
        )
    
    @classmethod
    def open_hand(cls, n_joints: int = 10) -> "GraspPrimitive":
        """Create an open hand primitive."""
        return cls(
            grasp_type=GraspType.OPEN,
            preshape=np.zeros(n_joints),
            final_shape=np.zeros(n_joints),
            force_distribution=np.zeros(5),
            approach_direction=np.array([0, 0, 0])
        )


# =============================================================================
# Force Controller
# =============================================================================

class ForceController:
    """
    Force controller for grip force regulation.
    
    Implements force feedback control with:
    - Proportional-Integral control
    - Slip detection and compensation
    - Force limits and ramping
    
    Control Law:
        τ = K_p (F_des - F_act) + K_i ∫(F_des - F_act)dt + τ_ff
        
        where τ_ff is feedforward torque based on desired force.
    """
    
    def __init__(
        self,
        config: GraspConfig,
        kp: float = 1.0,
        ki: float = 0.1
    ) -> None:
        """
        Initialize force controller.
        
        Args:
            config: Grasp configuration
            kp: Proportional gain
            ki: Integral gain
        """
        self.config = config
        self.kp = kp
        self.ki = ki
        
        # State
        self.target_force = 0.0
        self._integral_error = 0.0
        self._last_force = 0.0
        self._slip_detected = False
        
        # Force ramping
        self._ramp_rate = 10.0  # N/s
        self._ramped_target = 0.0
        
        logger.debug("ForceController initialized")
    
    def set_target_force(self, force: float) -> None:
        """
        Set target grip force.
        
        Args:
            force: Target force in Newtons
        """
        self.target_force = np.clip(
            force,
            self.config.force_min,
            self.config.force_max
        )
    
    def update(
        self,
        measured_force: float,
        dt: float
    ) -> float:
        """
        Update force controller.
        
        Args:
            measured_force: Measured grip force
            dt: Time step
            
        Returns:
            Motor torque command
        """
        # Ramp target force
        error_to_target = self.target_force - self._ramped_target
        ramp_step = self._ramp_rate * dt
        
        if abs(error_to_target) < ramp_step:
            self._ramped_target = self.target_force
        else:
            self._ramped_target += np.sign(error_to_target) * ramp_step
        
        # Compute force error
        force_error = self._ramped_target - measured_force
        
        # Slip detection (sudden force drop)
        force_rate = (measured_force - self._last_force) / dt if dt > 0 else 0
        self._last_force = measured_force
        
        if force_rate < -self.config.slip_threshold * self.config.force_max:
            self._slip_detected = True
            # Increase target force to compensate
            self._ramped_target = min(
                self._ramped_target * 1.2,
                self.config.force_max
            )
        else:
            self._slip_detected = False
        
        # PI control
        self._integral_error += force_error * dt
        self._integral_error = np.clip(self._integral_error, -5.0, 5.0)
        
        torque = self.kp * force_error + self.ki * self._integral_error
        
        # Feedforward based on target force
        torque += 0.1 * self._ramped_target
        
        return float(torque)
    
    def reset(self) -> None:
        """Reset controller state."""
        self._integral_error = 0.0
        self._ramped_target = 0.0
        self._last_force = 0.0
        self._slip_detected = False
    
    @property
    def slip_detected(self) -> bool:
        """Whether slip was detected in last update."""
        return self._slip_detected


# =============================================================================
# Grasp Controller
# =============================================================================

class GraspController:
    """
    High-level grasp controller managing grasp execution.
    
    Coordinates:
    - Grasp selection based on BCI intent
    - Preshaping and approach
    - Force-controlled closure
    - Grasp maintenance and release
    
    State Machine:
        IDLE → PRESHAPE → APPROACH → CLOSE → HOLD ↔ RELEASE → IDLE
    
    Example:
        >>> config = GraspConfig()
        >>> controller = GraspController(config)
        >>> 
        >>> # Initiate power grasp
        >>> controller.initiate_grasp(GraspType.POWER)
        >>> 
        >>> # Update loop
        >>> while running:
        ...     finger_commands = controller.update(
        ...         current_positions=current_pos,
        ...         current_forces=current_forces,
        ...         dt=0.01
        ...     )
        ...     send_to_hand(finger_commands)
    """
    
    # Default grasp primitives
    DEFAULT_PRIMITIVES = {
        GraspType.POWER: GraspPrimitive.power_grasp,
        GraspType.PRECISION: GraspPrimitive.precision_grasp,
        GraspType.LATERAL: GraspPrimitive.lateral_grasp,
        GraspType.OPEN: GraspPrimitive.open_hand,
    }
    
    def __init__(
        self,
        config: Optional[GraspConfig] = None,
        primitives: Optional[Dict[GraspType, GraspPrimitive]] = None
    ) -> None:
        """
        Initialize grasp controller.
        
        Args:
            config: Grasp configuration
            primitives: Custom grasp primitives
        """
        self.config = config or GraspConfig()
        
        # Initialize primitives
        if primitives is not None:
            self.primitives = primitives
        else:
            self.primitives = {
                grasp_type: factory(self.config.total_joints)
                for grasp_type, factory in self.DEFAULT_PRIMITIVES.items()
            }
        
        # Force controller
        self.force_controller = ForceController(self.config)
        
        # State
        self._phase = GraspPhase.IDLE
        self._current_grasp: Optional[GraspPrimitive] = None
        self._target_positions = np.zeros(self.config.total_joints)
        self._target_force = 0.0
        self._contact_detected = False
        
        # Phase timing
        self._phase_start_time = 0.0
        self._phase_duration = 0.0
        
        # Callbacks
        self._on_contact: Optional[Callable[[], None]] = None
        self._on_grasp_complete: Optional[Callable[[], None]] = None
        
        logger.info(
            f"GraspController initialized: {self.config.n_fingers} fingers, "
            f"{self.config.total_joints} joints"
        )
    
    @property
    def phase(self) -> GraspPhase:
        """Current grasp phase."""
        return self._phase
    
    @property
    def is_grasping(self) -> bool:
        """Whether currently in an active grasp."""
        return self._phase in (GraspPhase.CLOSE, GraspPhase.HOLD)
    
    def initiate_grasp(
        self,
        grasp_type: GraspType,
        force: Optional[float] = None
    ) -> bool:
        """
        Initiate a grasp.
        
        Args:
            grasp_type: Type of grasp to execute
            force: Target grip force (uses default if None)
            
        Returns:
            True if grasp was initiated, False if not possible
        """
        if self._phase not in (GraspPhase.IDLE, GraspPhase.RELEASE):
            logger.warning(f"Cannot initiate grasp in phase {self._phase.name}")
            return False
        
        if grasp_type not in self.primitives:
            logger.error(f"Unknown grasp type: {grasp_type}")
            return False
        
        self._current_grasp = self.primitives[grasp_type]
        self._target_force = force or self.config.force_default
        self._contact_detected = False
        
        self._transition_to(GraspPhase.PRESHAPE)
        
        logger.info(f"Initiating {grasp_type.name} grasp, force={self._target_force}N")
        return True
    
    def release(self) -> None:
        """Release current grasp."""
        if self._phase in (GraspPhase.CLOSE, GraspPhase.HOLD):
            self._transition_to(GraspPhase.RELEASE)
            logger.info("Releasing grasp")
    
    def update(
        self,
        current_positions: FloatArray,
        current_forces: Optional[FloatArray] = None,
        dt: float = 0.01,
        time: Optional[float] = None
    ) -> FloatArray:
        """
        Update grasp controller and get motor commands.
        
        Args:
            current_positions: Current finger joint positions
            current_forces: Current finger forces (for force control)
            dt: Time step
            time: Current time (for timing, uses internal if None)
            
        Returns:
            Target joint positions or velocities
        """
        current_positions = np.asarray(current_positions)
        
        if current_forces is not None:
            current_forces = np.asarray(current_forces)
            total_force = np.sum(np.abs(current_forces))
        else:
            total_force = 0.0
        
        # Phase-specific logic
        if self._phase == GraspPhase.IDLE:
            # Stay at current position or default open
            return current_positions
        
        elif self._phase == GraspPhase.PRESHAPE:
            return self._execute_preshape(current_positions, dt)
        
        elif self._phase == GraspPhase.APPROACH:
            return self._execute_approach(current_positions, dt)
        
        elif self._phase == GraspPhase.CLOSE:
            return self._execute_close(current_positions, total_force, dt)
        
        elif self._phase == GraspPhase.HOLD:
            return self._execute_hold(current_positions, total_force, dt)
        
        elif self._phase == GraspPhase.RELEASE:
            return self._execute_release(current_positions, dt)
        
        return current_positions
    
    def _transition_to(self, phase: GraspPhase) -> None:
        """Transition to a new phase."""
        old_phase = self._phase
        self._phase = phase
        self._phase_start_time = 0.0
        
        # Phase-specific initialization
        if phase == GraspPhase.PRESHAPE:
            self._target_positions = self._current_grasp.preshape.copy()
            self._phase_duration = 0.5
            
        elif phase == GraspPhase.CLOSE:
            self._target_positions = self._current_grasp.final_shape.copy()
            self.force_controller.set_target_force(self._target_force)
            
        elif phase == GraspPhase.RELEASE:
            self._target_positions = np.zeros(self.config.total_joints)
            self.force_controller.reset()
            self._phase_duration = 0.3
        
        logger.debug(f"Grasp phase: {old_phase.name} → {phase.name}")
    
    def _execute_preshape(
        self,
        current: FloatArray,
        dt: float
    ) -> FloatArray:
        """Execute preshape phase."""
        self._phase_start_time += dt
        
        # Move toward preshape position
        error = self._target_positions - current
        velocity = np.clip(error / 0.5, -self.config.close_velocity, self.config.close_velocity)
        command = current + velocity * dt
        
        # Check if preshape complete
        if np.max(np.abs(error)) < 0.05:
            self._transition_to(GraspPhase.CLOSE)
        
        return command
    
    def _execute_approach(
        self,
        current: FloatArray,
        dt: float
    ) -> FloatArray:
        """Execute approach phase (for integrated arm+hand control)."""
        # This phase is typically handled by arm controller
        # Transition to close when approach is complete
        self._phase_start_time += dt
        return current
    
    def _execute_close(
        self,
        current: FloatArray,
        force: float,
        dt: float
    ) -> FloatArray:
        """Execute closing phase."""
        self._phase_start_time += dt
        
        # Check for contact (force threshold)
        contact_threshold = 0.5  # N
        if force > contact_threshold:
            self._contact_detected = True
            if self._on_contact:
                self._on_contact()
        
        # If contact, transition to hold
        if self._contact_detected and force >= self._target_force * 0.8:
            self._transition_to(GraspPhase.HOLD)
            if self._on_grasp_complete:
                self._on_grasp_complete()
            return current
        
        # Close fingers at configured velocity
        error = self._target_positions - current
        velocity = np.clip(error * 2.0, -self.config.close_velocity, self.config.close_velocity)
        command = current + velocity * dt
        
        return command
    
    def _execute_hold(
        self,
        current: FloatArray,
        force: float,
        dt: float
    ) -> FloatArray:
        """Execute hold phase - maintain grasp force."""
        # Update force controller
        torque_adjustment = self.force_controller.update(force, dt)
        
        # Adjust finger positions based on force error
        adjustment = torque_adjustment * 0.01
        
        command = current + adjustment * self._current_grasp.force_distribution.repeat(2)[:len(current)]
        
        # Clamp to joint limits
        command = np.clip(command, 0, np.pi)
        
        return command
    
    def _execute_release(
        self,
        current: FloatArray,
        dt: float
    ) -> FloatArray:
        """Execute release phase."""
        self._phase_start_time += dt
        
        # Move to open position
        error = self._target_positions - current
        velocity = np.clip(error / 0.3, -self.config.close_velocity * 1.5, self.config.close_velocity * 1.5)
        command = current + velocity * dt
        
        # Check if release complete
        if self._phase_start_time > self._phase_duration or np.max(np.abs(error)) < 0.05:
            self._transition_to(GraspPhase.IDLE)
            self._current_grasp = None
        
        return command
    
    def set_force(self, force: float) -> None:
        """
        Adjust target grip force during grasp.
        
        Args:
            force: New target force in Newtons
        """
        self._target_force = np.clip(
            force,
            self.config.force_min,
            self.config.force_max
        )
        self.force_controller.set_target_force(self._target_force)
    
    def scale_force(self, scale: float) -> None:
        """
        Scale current grip force.
        
        Args:
            scale: Scale factor (0-2 typical)
        """
        new_force = self._target_force * scale
        self.set_force(new_force)
    
    def on_contact(self, callback: Callable[[], None]) -> None:
        """Register callback for contact detection."""
        self._on_contact = callback
    
    def on_grasp_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for grasp completion."""
        self._on_grasp_complete = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status for monitoring."""
        return {
            "phase": self._phase.name,
            "grasp_type": self._current_grasp.grasp_type.name if self._current_grasp else None,
            "target_force": self._target_force,
            "contact_detected": self._contact_detected,
            "slip_detected": self.force_controller.slip_detected,
            "phase_time": self._phase_start_time,
        }
    
    def add_primitive(
        self,
        grasp_type: GraspType,
        primitive: GraspPrimitive
    ) -> None:
        """
        Add or update a grasp primitive.
        
        Args:
            grasp_type: Type identifier
            primitive: Grasp primitive definition
        """
        self.primitives[grasp_type] = primitive
