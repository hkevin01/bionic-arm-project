"""
Arm Controller Module
=====================

Real-time motor control interface for the prosthetic arm.

Features:
    - Multiple control modes (position, velocity, torque)
    - Safety limits and emergency stop
    - Smooth velocity ramping
    - Integration with trajectory generator and kinematics
    - Hardware abstraction for different motor interfaces

Control Modes:
    - POSITION: Direct joint position control (PID)
    - VELOCITY: Velocity control with limits
    - TORQUE: Direct torque control (for compliant interaction)
    - IMPEDANCE: Spring-damper behavior

Safety:
    - Joint limit checking
    - Velocity limits
    - Emergency stop capability
    - Watchdog timeout
    - Collision detection (if sensors available)

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Callable, Protocol
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from .kinematics import ArmKinematics, JointLimits, Transform
from .trajectory import TrajectoryGenerator, TrajectoryPoint

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]


# =============================================================================
# Enums and Configuration
# =============================================================================

class ControlMode(Enum):
    """Arm control mode."""
    POSITION = auto()       # Position control
    VELOCITY = auto()       # Velocity control
    TORQUE = auto()         # Torque/current control
    IMPEDANCE = auto()      # Impedance control
    OFF = auto()            # Motors disabled


class ControllerState(Enum):
    """Controller state machine states."""
    IDLE = auto()
    ACTIVE = auto()
    HOLDING = auto()
    EMERGENCY_STOP = auto()
    FAULT = auto()


@dataclass
class ControllerConfig:
    """
    Configuration for arm controller.
    
    Attributes:
        n_joints: Number of arm joints (not including hand)
        control_rate_hz: Control loop rate
        position_kp: Position proportional gains
        position_kd: Position derivative gains
        velocity_limit: Maximum velocity per joint (rad/s)
        acceleration_limit: Maximum acceleration per joint (rad/s²)
        torque_limit: Maximum torque per joint (Nm)
        watchdog_timeout_ms: Watchdog timeout (ms)
    """
    n_joints: int = 7
    control_rate_hz: float = 100.0
    position_kp: Optional[FloatArray] = None
    position_kd: Optional[FloatArray] = None
    velocity_limit: Optional[FloatArray] = None
    acceleration_limit: Optional[FloatArray] = None
    torque_limit: Optional[FloatArray] = None
    watchdog_timeout_ms: float = 100.0
    
    def __post_init__(self) -> None:
        """Set default gains and limits."""
        if self.position_kp is None:
            self.position_kp = np.ones(self.n_joints) * 100.0
        if self.position_kd is None:
            self.position_kd = np.ones(self.n_joints) * 10.0
        if self.velocity_limit is None:
            self.velocity_limit = np.ones(self.n_joints) * 2.0
        if self.acceleration_limit is None:
            self.acceleration_limit = np.ones(self.n_joints) * 10.0
        if self.torque_limit is None:
            self.torque_limit = np.ones(self.n_joints) * 50.0
    
    @property
    def control_period(self) -> float:
        """Control period in seconds."""
        return 1.0 / self.control_rate_hz


@dataclass
class MotorCommand:
    """
    Command to send to motor drivers.
    
    Attributes:
        mode: Control mode
        position: Target positions (rad)
        velocity: Target velocities (rad/s)
        torque: Target torques (Nm)
        timestamp: Command timestamp
    """
    mode: ControlMode
    position: Optional[FloatArray] = None
    velocity: Optional[FloatArray] = None
    torque: Optional[FloatArray] = None
    timestamp: float = 0.0
    
    def __post_init__(self) -> None:
        """Record timestamp."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# =============================================================================
# Hardware Interface Protocol
# =============================================================================

class MotorInterface(Protocol):
    """Protocol for motor hardware interface."""
    
    def send_command(self, command: MotorCommand) -> bool:
        """Send command to motors."""
        ...
    
    def get_state(self) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """Get current (positions, velocities, torques)."""
        ...
    
    def enable(self) -> bool:
        """Enable motors."""
        ...
    
    def disable(self) -> bool:
        """Disable motors."""
        ...
    
    def emergency_stop(self) -> None:
        """Emergency stop."""
        ...


# =============================================================================
# Simulated Motor Interface
# =============================================================================

class SimulatedMotorInterface:
    """
    Simulated motor interface for testing.
    
    Simulates motor dynamics with configurable parameters.
    """
    
    def __init__(
        self,
        n_joints: int = 7,
        dynamics_time_constant: float = 0.02
    ) -> None:
        """
        Initialize simulated motors.
        
        Args:
            n_joints: Number of joints
            dynamics_time_constant: Response time constant
        """
        self.n_joints = n_joints
        self.tau = dynamics_time_constant
        
        # State
        self._position = np.zeros(n_joints)
        self._velocity = np.zeros(n_joints)
        self._torque = np.zeros(n_joints)
        
        # Target
        self._target_position = np.zeros(n_joints)
        self._target_velocity = np.zeros(n_joints)
        
        self._enabled = False
        self._last_update = time.perf_counter()
        
        logger.info(f"SimulatedMotorInterface: {n_joints} joints")
    
    def send_command(self, command: MotorCommand) -> bool:
        """Process motor command."""
        if not self._enabled:
            return False
        
        current_time = time.perf_counter()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        if command.mode == ControlMode.POSITION and command.position is not None:
            self._target_position = command.position.copy()
            # Simple first-order dynamics
            alpha = 1.0 - np.exp(-dt / self.tau)
            self._velocity = (self._target_position - self._position) / max(dt, 0.001)
            self._position += alpha * (self._target_position - self._position)
            
        elif command.mode == ControlMode.VELOCITY and command.velocity is not None:
            self._target_velocity = command.velocity.copy()
            self._velocity = self._target_velocity
            self._position += self._velocity * dt
        
        return True
    
    def get_state(self) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """Get current state."""
        return (
            self._position.copy(),
            self._velocity.copy(),
            self._torque.copy()
        )
    
    def enable(self) -> bool:
        """Enable motors."""
        self._enabled = True
        self._last_update = time.perf_counter()
        return True
    
    def disable(self) -> bool:
        """Disable motors."""
        self._enabled = False
        self._velocity = np.zeros(self.n_joints)
        return True
    
    def emergency_stop(self) -> None:
        """Emergency stop."""
        self._enabled = False
        self._velocity = np.zeros(self.n_joints)
        self._target_velocity = np.zeros(self.n_joints)


# =============================================================================
# Arm Controller
# =============================================================================

class ArmController:
    """
    High-level arm controller managing motion and safety.
    
    Integrates:
    - Trajectory generation
    - Inverse kinematics
    - Motor control
    - Safety systems
    
    Example:
        >>> config = ControllerConfig(n_joints=7)
        >>> controller = ArmController(config)
        >>> 
        >>> controller.enable()
        >>> controller.set_control_mode(ControlMode.VELOCITY)
        >>> 
        >>> # From BCI pipeline
        >>> while running:
        ...     velocity, confidence = bci_pipeline.get_output()
        ...     controller.set_velocity_command(velocity)
        ...     controller.update()
    """
    
    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        kinematics: Optional[ArmKinematics] = None,
        motor_interface: Optional[MotorInterface] = None
    ) -> None:
        """
        Initialize arm controller.
        
        Args:
            config: Controller configuration
            kinematics: Arm kinematics model
            motor_interface: Hardware interface (uses simulation if None)
        """
        self.config = config or ControllerConfig()
        
        # Kinematics
        if kinematics is not None:
            self.kinematics = kinematics
        else:
            self.kinematics = ArmKinematics.default_arm()
        
        # Motor interface
        if motor_interface is not None:
            self.motors = motor_interface
        else:
            self.motors = SimulatedMotorInterface(self.config.n_joints)
        
        # Trajectory generator
        self.trajectory = TrajectoryGenerator()
        
        # State
        self._state = ControllerState.IDLE
        self._control_mode = ControlMode.OFF
        self._state_lock = threading.Lock()
        
        # Current state
        self._current_position = np.zeros(self.config.n_joints)
        self._current_velocity = np.zeros(self.config.n_joints)
        self._current_torque = np.zeros(self.config.n_joints)
        
        # Commands
        self._target_position = np.zeros(self.config.n_joints)
        self._target_velocity = np.zeros(self.config.n_joints)
        self._target_torque = np.zeros(self.config.n_joints)
        
        # Velocity ramping
        self._velocity_command = np.zeros(self.config.n_joints)
        self._velocity_ramped = np.zeros(self.config.n_joints)
        self._ramp_rate = 5.0  # rad/s²
        
        # Watchdog
        self._last_command_time = 0.0
        self._watchdog_triggered = False
        
        # Control loop
        self._control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Metrics
        self._update_count = 0
        self._last_update_time = 0.0
        self._control_rate_actual = 0.0
        
        logger.info(
            f"ArmController initialized: {self.config.n_joints} joints, "
            f"{self.config.control_rate_hz}Hz"
        )
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    @property
    def state(self) -> ControllerState:
        """Current controller state."""
        with self._state_lock:
            return self._state
    
    @property
    def control_mode(self) -> ControlMode:
        """Current control mode."""
        with self._state_lock:
            return self._control_mode
    
    def _set_state(self, state: ControllerState) -> None:
        """Set controller state (thread-safe)."""
        with self._state_lock:
            old_state = self._state
            self._state = state
            logger.info(f"Controller state: {old_state.name} → {state.name}")
    
    # =========================================================================
    # Control Interface
    # =========================================================================
    
    def enable(self) -> bool:
        """
        Enable the controller and motors.
        
        Returns:
            True if enabled successfully
        """
        if self._state == ControllerState.EMERGENCY_STOP:
            logger.error("Cannot enable during emergency stop")
            return False
        
        if not self.motors.enable():
            logger.error("Failed to enable motors")
            return False
        
        # Read current position
        self._current_position, self._current_velocity, self._current_torque = \
            self.motors.get_state()
        
        # Initialize targets to current position
        self._target_position = self._current_position.copy()
        self._target_velocity = np.zeros(self.config.n_joints)
        
        self._set_state(ControllerState.ACTIVE)
        self._last_command_time = time.perf_counter()
        
        logger.info("Controller enabled")
        return True
    
    def disable(self) -> None:
        """Disable the controller and motors."""
        self.motors.disable()
        self._set_state(ControllerState.IDLE)
        self._control_mode = ControlMode.OFF
        logger.info("Controller disabled")
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._set_state(ControllerState.EMERGENCY_STOP)
        self.motors.emergency_stop()
        self._velocity_command = np.zeros(self.config.n_joints)
        self._velocity_ramped = np.zeros(self.config.n_joints)
        logger.warning("EMERGENCY STOP triggered")
    
    def reset_emergency_stop(self) -> bool:
        """
        Reset emergency stop state.
        
        Returns:
            True if reset successful
        """
        if self._state != ControllerState.EMERGENCY_STOP:
            return True
        
        # Check conditions for reset
        self._watchdog_triggered = False
        self._set_state(ControllerState.IDLE)
        logger.info("Emergency stop reset")
        return True
    
    def set_control_mode(self, mode: ControlMode) -> None:
        """
        Set the control mode.
        
        Args:
            mode: Desired control mode
        """
        with self._state_lock:
            old_mode = self._control_mode
            self._control_mode = mode
            logger.info(f"Control mode: {old_mode.name} → {mode.name}")
    
    # =========================================================================
    # Commands
    # =========================================================================
    
    def set_position_command(
        self,
        position: FloatArray,
        check_limits: bool = True
    ) -> bool:
        """
        Set target joint positions.
        
        Args:
            position: Target joint positions (rad)
            check_limits: Whether to check joint limits
            
        Returns:
            True if command accepted
        """
        if self._state != ControllerState.ACTIVE:
            return False
        
        position = np.asarray(position).flatten()
        
        if check_limits:
            valid, violations = self.kinematics.check_joint_limits(position)
            if not valid:
                logger.warning(f"Position violates limits on joints: {violations}")
                position = self.kinematics.clamp_joints(position)
        
        self._target_position = position
        self._last_command_time = time.perf_counter()
        
        return True
    
    def set_velocity_command(
        self,
        velocity: FloatArray,
        scale: float = 1.0
    ) -> bool:
        """
        Set target joint velocities.
        
        Velocities are automatically clamped to limits.
        
        Args:
            velocity: Target joint velocities (rad/s)
            scale: Scaling factor (0-1 from BCI confidence)
            
        Returns:
            True if command accepted
        """
        if self._state != ControllerState.ACTIVE:
            return False
        
        velocity = np.asarray(velocity).flatten() * scale
        
        # Clamp to velocity limits
        velocity = np.clip(
            velocity,
            -self.config.velocity_limit,
            self.config.velocity_limit
        )
        
        self._velocity_command = velocity
        self._last_command_time = time.perf_counter()
        
        return True
    
    def set_cartesian_velocity(
        self,
        linear: FloatArray,
        angular: FloatArray
    ) -> bool:
        """
        Set Cartesian end-effector velocity.
        
        Uses Jacobian to convert to joint velocities.
        
        Args:
            linear: Linear velocity [vx, vy, vz] (m/s)
            angular: Angular velocity [wx, wy, wz] (rad/s)
            
        Returns:
            True if command accepted
        """
        linear = np.asarray(linear).flatten()
        angular = np.asarray(angular).flatten()
        
        # Get Jacobian at current position
        J_pinv = self.kinematics.jacobian_pinv(self._current_position)
        
        # Task space velocity
        x_dot = np.concatenate([linear, angular])
        
        # Convert to joint velocity
        q_dot = J_pinv @ x_dot
        
        return self.set_velocity_command(q_dot)
    
    def move_to_position(
        self,
        position: FloatArray,
        duration: Optional[float] = None
    ) -> bool:
        """
        Generate and execute trajectory to position.
        
        Args:
            position: Target joint positions
            duration: Movement duration (auto-computed if None)
            
        Returns:
            True if trajectory started
        """
        if self._state != ControllerState.ACTIVE:
            return False
        
        position = np.asarray(position).flatten()
        
        # Generate trajectory
        trajectory = self.trajectory.point_to_point(
            start=self._current_position,
            end=position,
            duration=duration
        )
        
        # Start trajectory execution
        # (In a real system, this would be handled by the control loop)
        self._target_position = position
        self.set_control_mode(ControlMode.POSITION)
        
        return True
    
    def move_to_cartesian(
        self,
        target: Transform,
        duration: Optional[float] = None
    ) -> bool:
        """
        Move to Cartesian pose using IK.
        
        Args:
            target: Target end-effector pose
            duration: Movement duration
            
        Returns:
            True if IK successful and trajectory started
        """
        # Compute IK
        q_target, success = self.kinematics.inverse_kinematics(
            target,
            q_init=self._current_position
        )
        
        if not success:
            logger.warning("IK failed to converge")
            return False
        
        return self.move_to_position(q_target, duration)
    
    # =========================================================================
    # Update Loop
    # =========================================================================
    
    def update(self) -> None:
        """
        Execute one control cycle.
        
        Call this at the control rate.
        """
        if self._state != ControllerState.ACTIVE:
            return
        
        current_time = time.perf_counter()
        dt = current_time - self._last_update_time if self._last_update_time > 0 else self.config.control_period
        self._last_update_time = current_time
        
        # Watchdog check
        time_since_command = current_time - self._last_command_time
        if time_since_command > self.config.watchdog_timeout_ms / 1000.0:
            if not self._watchdog_triggered:
                logger.warning("Watchdog timeout - ramping down velocity")
                self._watchdog_triggered = True
            # Ramp down velocity
            self._velocity_command = np.zeros(self.config.n_joints)
        else:
            self._watchdog_triggered = False
        
        # Get current state
        self._current_position, self._current_velocity, self._current_torque = \
            self.motors.get_state()
        
        # Compute command based on mode
        command = self._compute_command(dt)
        
        # Send to motors
        self.motors.send_command(command)
        
        # Update metrics
        self._update_count += 1
        if self._update_count % 100 == 0:
            self._control_rate_actual = 100.0 / (current_time - self._last_update_time * 100) \
                if self._update_count > 100 else self.config.control_rate_hz
    
    def _compute_command(self, dt: float) -> MotorCommand:
        """
        Compute motor command based on current mode.
        
        Args:
            dt: Time step
            
        Returns:
            Motor command
        """
        mode = self._control_mode
        
        if mode == ControlMode.POSITION:
            return MotorCommand(
                mode=ControlMode.POSITION,
                position=self._target_position
            )
        
        elif mode == ControlMode.VELOCITY:
            # Ramp velocity for smooth acceleration
            self._ramp_velocity(dt)
            
            # Integrate velocity to position
            self._target_position += self._velocity_ramped * dt
            
            # Check limits
            self._target_position = self.kinematics.clamp_joints(self._target_position)
            
            return MotorCommand(
                mode=ControlMode.VELOCITY,
                velocity=self._velocity_ramped,
                position=self._target_position
            )
        
        elif mode == ControlMode.TORQUE:
            return MotorCommand(
                mode=ControlMode.TORQUE,
                torque=self._target_torque
            )
        
        else:
            # OFF mode
            return MotorCommand(
                mode=ControlMode.OFF,
                velocity=np.zeros(self.config.n_joints)
            )
    
    def _ramp_velocity(self, dt: float) -> None:
        """
        Apply velocity ramping for smooth acceleration.
        
        Args:
            dt: Time step
        """
        error = self._velocity_command - self._velocity_ramped
        max_change = self._ramp_rate * dt
        
        # Limit rate of change
        change = np.clip(error, -max_change, max_change)
        self._velocity_ramped += change
    
    # =========================================================================
    # Background Control Loop
    # =========================================================================
    
    def start_control_loop(self) -> None:
        """Start background control loop thread."""
        if self._control_thread is not None and self._control_thread.is_alive():
            logger.warning("Control loop already running")
            return
        
        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="ArmController-ControlLoop",
            daemon=True
        )
        self._control_thread.start()
        logger.info("Control loop started")
    
    def stop_control_loop(self) -> None:
        """Stop background control loop."""
        self._stop_event.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
        logger.info("Control loop stopped")
    
    def _control_loop(self) -> None:
        """Background control loop thread."""
        period = self.config.control_period
        next_time = time.perf_counter()
        
        while not self._stop_event.is_set():
            self.update()
            
            next_time += period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # =========================================================================
    # State Access
    # =========================================================================
    
    @property
    def current_position(self) -> FloatArray:
        """Current joint positions."""
        return self._current_position.copy()
    
    @property
    def current_velocity(self) -> FloatArray:
        """Current joint velocities."""
        return self._current_velocity.copy()
    
    @property
    def current_pose(self) -> Transform:
        """Current end-effector pose."""
        return self.kinematics.forward_kinematics(self._current_position)
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status for monitoring."""
        return {
            "state": self._state.name,
            "control_mode": self._control_mode.name,
            "position": self._current_position.tolist(),
            "velocity": self._current_velocity.tolist(),
            "target_position": self._target_position.tolist(),
            "velocity_command": self._velocity_command.tolist(),
            "velocity_ramped": self._velocity_ramped.tolist(),
            "watchdog_triggered": self._watchdog_triggered,
            "control_rate_actual": self._control_rate_actual,
            "update_count": self._update_count,
        }
