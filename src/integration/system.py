"""
System Orchestrator Module
==========================

Main system integration connecting BCI signals to arm control
with sensory feedback in a closed-loop architecture.

Pipeline:
    EEG Acquisition → Preprocessing → Feature Extraction → Decoding
    → Motion Planning → Control → Motor Commands
    → Force Sensing → Haptic Feedback → User

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
import json
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """Operating modes for the bionic arm system."""
    IDLE = auto()           # System initialized but not active
    CALIBRATION = auto()    # BCI calibration mode
    SIMULATION = auto()     # Simulation-only mode
    ASSISTED = auto()       # Shared control with user
    AUTONOMOUS = auto()     # Full BCI control
    EMERGENCY_STOP = auto() # Safety stop engaged


class SystemState(Enum):
    """System health states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class LatencyMetrics:
    """End-to-end latency measurements."""
    bci_acquisition_ms: float = 0.0
    preprocessing_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    decoding_ms: float = 0.0
    control_ms: float = 0.0
    feedback_ms: float = 0.0
    total_ms: float = 0.0
    
    def update_total(self) -> None:
        """Compute total latency."""
        self.total_ms = (
            self.bci_acquisition_ms +
            self.preprocessing_ms +
            self.feature_extraction_ms +
            self.decoding_ms +
            self.control_ms +
            self.feedback_ms
        )


@dataclass 
class SystemConfig:
    """
    Configuration for the bionic arm system.
    
    Attributes:
        name: System identifier
        mode: Initial operating mode
        use_simulation: Use simulated arm (vs real hardware)
        use_simulated_bci: Use simulated EEG (vs real signals)
        control_rate_hz: Main control loop rate
        bci_rate_hz: BCI processing rate
        feedback_rate_hz: Feedback update rate
        max_latency_ms: Maximum allowed end-to-end latency
        enable_safety: Enable safety monitoring
        log_level: Logging verbosity
    """
    name: str = "BionicArmSystem"
    mode: SystemMode = SystemMode.SIMULATION
    use_simulation: bool = True
    use_simulated_bci: bool = True
    control_rate_hz: float = 100.0
    bci_rate_hz: float = 50.0
    feedback_rate_hz: float = 30.0
    max_latency_ms: float = 150.0
    enable_safety: bool = True
    log_level: str = "INFO"
    
    # Component configs (optional paths)
    bci_config_path: Optional[str] = None
    control_config_path: Optional[str] = None
    feedback_config_path: Optional[str] = None
    
    # Data paths
    model_path: Optional[str] = None
    calibration_path: Optional[str] = None
    
    @property
    def control_period(self) -> float:
        return 1.0 / self.control_rate_hz
    
    @property
    def bci_period(self) -> float:
        return 1.0 / self.bci_rate_hz
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of issues."""
        issues = []
        if self.control_rate_hz <= 0:
            issues.append("control_rate_hz must be positive")
        if self.bci_rate_hz <= 0:
            issues.append("bci_rate_hz must be positive")
        if self.max_latency_ms <= 0:
            issues.append("max_latency_ms must be positive")
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mode": self.mode.name,
            "use_simulation": self.use_simulation,
            "use_simulated_bci": self.use_simulated_bci,
            "control_rate_hz": self.control_rate_hz,
            "bci_rate_hz": self.bci_rate_hz,
            "max_latency_ms": self.max_latency_ms,
            "enable_safety": self.enable_safety
        }
    
    @classmethod
    def from_file(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'mode' in data:
            data['mode'] = SystemMode[data['mode']]
        
        return cls(**data)
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ComponentManager:
    """
    Manages system component lifecycle.
    
    Handles initialization, health checking, and shutdown
    of all subsystem components.
    """
    
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._component_status: Dict[str, bool] = {}
        self._initialization_order: List[str] = []
    
    def register(self, name: str, component: Any, order: int = 100) -> None:
        """Register a component."""
        self._components[name] = component
        self._component_status[name] = False
        
        # Insert in order
        insert_pos = 0
        for i, existing in enumerate(self._initialization_order):
            if order < i:
                break
            insert_pos = i + 1
        self._initialization_order.insert(insert_pos, name)
    
    def get(self, name: str) -> Optional[Any]:
        """Get component by name."""
        return self._components.get(name)
    
    def initialize_all(self) -> bool:
        """Initialize all components in order."""
        for name in self._initialization_order:
            component = self._components[name]
            try:
                if hasattr(component, 'initialize'):
                    success = component.initialize()
                    if not success:
                        logger.error(f"Failed to initialize {name}")
                        return False
                elif hasattr(component, 'start'):
                    component.start()
                
                self._component_status[name] = True
                logger.info(f"Initialized: {name}")
                
            except Exception as e:
                logger.error(f"Error initializing {name}: {e}")
                return False
        
        return True
    
    def shutdown_all(self) -> None:
        """Shutdown all components in reverse order."""
        for name in reversed(self._initialization_order):
            component = self._components[name]
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                elif hasattr(component, 'cleanup'):
                    component.cleanup()
                elif hasattr(component, 'shutdown'):
                    component.shutdown()
                
                self._component_status[name] = False
                logger.info(f"Shutdown: {name}")
                
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {}
        for name, component in self._components.items():
            if hasattr(component, 'is_running'):
                health[name] = component.is_running
            elif hasattr(component, 'is_initialized'):
                health[name] = component.is_initialized
            else:
                health[name] = self._component_status.get(name, False)
        return health


class SafetyMonitor:
    """
    Safety monitoring for the bionic arm system.
    
    Monitors for:
        - Joint limit violations
        - Excessive velocities
        - Force limits exceeded
        - Communication timeouts
        - BCI signal quality
    """
    
    def __init__(
        self,
        max_joint_velocity: float = 2.0,  # rad/s
        max_joint_torque: float = 50.0,   # Nm
        max_force: float = 30.0,          # N
        comm_timeout_ms: float = 100.0
    ) -> None:
        self.max_joint_velocity = max_joint_velocity
        self.max_joint_torque = max_joint_torque
        self.max_force = max_force
        self.comm_timeout_ms = comm_timeout_ms
        
        self._emergency_stop = False
        self._violations: List[Dict[str, Any]] = []
        self._last_update = time.time()
        self._callbacks: List[Callable[[], None]] = []
    
    def check(
        self,
        joint_velocities: NDArray,
        joint_torques: NDArray,
        forces: NDArray
    ) -> bool:
        """
        Check safety constraints.
        
        Args:
            joint_velocities: Current joint velocities
            joint_torques: Current joint torques  
            forces: Current contact forces
            
        Returns:
            True if safe, False if violation
        """
        self._last_update = time.time()
        
        # Check velocities
        if np.any(np.abs(joint_velocities) > self.max_joint_velocity):
            self._record_violation("velocity", np.max(np.abs(joint_velocities)))
            return False
        
        # Check torques
        if np.any(np.abs(joint_torques) > self.max_joint_torque):
            self._record_violation("torque", np.max(np.abs(joint_torques)))
            return False
        
        # Check forces
        if np.any(np.abs(forces) > self.max_force):
            self._record_violation("force", np.max(np.abs(forces)))
            return False
        
        return True
    
    def check_communication(self) -> bool:
        """Check for communication timeout."""
        elapsed_ms = (time.time() - self._last_update) * 1000
        if elapsed_ms > self.comm_timeout_ms:
            self._record_violation("timeout", elapsed_ms)
            return False
        return True
    
    def _record_violation(self, violation_type: str, value: float) -> None:
        """Record a safety violation."""
        violation = {
            "type": violation_type,
            "value": value,
            "timestamp": time.time()
        }
        self._violations.append(violation)
        logger.warning(f"Safety violation: {violation_type} = {value}")
        
        # Keep only recent violations
        if len(self._violations) > 1000:
            self._violations = self._violations[-500:]
    
    def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        for callback in self._callbacks:
            try:
                callback()
            except:
                pass
    
    def reset(self) -> None:
        """Reset emergency stop."""
        self._emergency_stop = False
        self._violations = []
        logger.info("Safety monitor reset")
    
    def add_estop_callback(self, callback: Callable[[], None]) -> None:
        """Add callback for emergency stop events."""
        self._callbacks.append(callback)
    
    @property
    def is_safe(self) -> bool:
        return not self._emergency_stop
    
    @property
    def violations(self) -> List[Dict[str, Any]]:
        return self._violations.copy()


class BionicArmSystem:
    """
    Main bionic arm system orchestrator.
    
    Coordinates all components in a closed-loop control architecture:
    
    BCI Pipeline → Control System → Arm Hardware/Simulation
          ↑                              ↓
          ←←←←← Sensory Feedback ←←←←←←←←
    
    Example:
        >>> config = SystemConfig(mode=SystemMode.SIMULATION)
        >>> system = BionicArmSystem(config)
        >>> 
        >>> # Initialize
        >>> system.initialize()
        >>> 
        >>> # Run
        >>> system.start()
        >>> 
        >>> # ... system running ...
        >>> 
        >>> # Shutdown
        >>> system.stop()
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None
    ) -> None:
        self.config = config or SystemConfig()
        
        # Validate config
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")
        
        # State
        self._state = SystemState.UNINITIALIZED
        self._mode = self.config.mode
        self._running = False
        
        # Component manager
        self._components = ComponentManager()
        
        # Safety
        self._safety = SafetyMonitor() if self.config.enable_safety else None
        
        # Latency tracking
        self._latency = LatencyMetrics()
        
        # Main loop
        self._control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Data flow
        self._current_bci_output: Optional[NDArray] = None
        self._current_arm_state: Optional[Any] = None
        self._current_forces: Optional[NDArray] = None
        
        # Callbacks
        self._state_callbacks: List[Callable[[SystemState], None]] = []
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        logger.info(f"BionicArmSystem created: {self.config.name}")
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if successful
        """
        self._state = SystemState.INITIALIZING
        self._notify_state_change()
        
        try:
            # Create and register components
            self._setup_components()
            
            # Initialize all
            success = self._components.initialize_all()
            
            if success:
                self._state = SystemState.READY
                logger.info("System initialized successfully")
            else:
                self._state = SystemState.ERROR
                logger.error("System initialization failed")
            
            self._notify_state_change()
            return success
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self._state = SystemState.ERROR
            self._notify_state_change()
            return False
    
    def _setup_components(self) -> None:
        """Create and register system components."""
        # Import components (lazy to avoid circular imports)
        from ..bci import BCIPipeline, PipelineConfig
        from ..control import ArmController
        from ..feedback import VibrotactileFeedback, VisualFeedback
        from ..simulation import ArmSimulator, SimulatorConfig, PhysicsBackend
        
        # BCI Pipeline
        bci_config = PipelineConfig(
            acquisition_type="simulated" if self.config.use_simulated_bci else "lsl"
        )
        bci_pipeline = BCIPipeline(bci_config)
        self._components.register("bci", bci_pipeline, order=1)
        
        # Arm (simulation or hardware)
        if self.config.use_simulation:
            sim_config = SimulatorConfig(
                backend=PhysicsBackend.SIMPLE,
                use_realtime=True
            )
            arm = ArmSimulator(sim_config)
        else:
            # Placeholder for real hardware
            arm = ArmSimulator(SimulatorConfig(backend=PhysicsBackend.SIMPLE))
        self._components.register("arm", arm, order=2)
        
        # Controller
        controller = ArmController()
        self._components.register("controller", controller, order=3)
        
        # Feedback
        vibro = VibrotactileFeedback()
        visual = VisualFeedback()
        self._components.register("vibrotactile", vibro, order=4)
        self._components.register("visual", visual, order=5)
        
        # Safety monitor callbacks
        if self._safety:
            self._safety.add_estop_callback(self._on_emergency_stop)
    
    def start(self) -> None:
        """Start the control loop."""
        if self._state != SystemState.READY:
            logger.error(f"Cannot start: state is {self._state.name}")
            return
        
        self._running = True
        self._stop_event.clear()
        self._state = SystemState.RUNNING
        self._notify_state_change()
        
        # Start control thread
        self._control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="ControlLoop"
        )
        self._control_thread.start()
        
        logger.info("System started")
    
    def stop(self) -> None:
        """Stop the control loop."""
        self._running = False
        self._stop_event.set()
        
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
        
        self._state = SystemState.READY
        self._notify_state_change()
        logger.info("System stopped")
    
    def shutdown(self) -> None:
        """Shutdown the entire system."""
        self.stop()
        self._components.shutdown_all()
        self._state = SystemState.SHUTDOWN
        self._notify_state_change()
        logger.info("System shutdown complete")
    
    def _control_loop(self) -> None:
        """Main control loop."""
        period = self.config.control_period
        
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            
            try:
                self._control_step()
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                if self.config.enable_safety:
                    self._mode = SystemMode.EMERGENCY_STOP
            
            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > period * 1.5:
                logger.warning(f"Control loop overrun: {elapsed*1000:.1f}ms")
    
    def _control_step(self) -> None:
        """Single control loop iteration."""
        t0 = time.perf_counter()
        
        # Skip if in emergency stop
        if self._mode == SystemMode.EMERGENCY_STOP:
            self._send_zero_command()
            return
        
        # 1. Get BCI output
        bci = self._components.get("bci")
        if bci and hasattr(bci, 'get_output'):
            self._current_bci_output = bci.get_output()
        t1 = time.perf_counter()
        self._latency.bci_acquisition_ms = (t1 - t0) * 1000
        
        # 2. Get arm state
        arm = self._components.get("arm")
        if arm:
            self._current_arm_state = arm.get_state()
        t2 = time.perf_counter()
        
        # 3. Compute control command
        controller = self._components.get("controller")
        if controller and self._current_bci_output is not None:
            # Map BCI output to velocity command
            velocity_cmd = self._bci_to_velocity(self._current_bci_output)
            controller.set_velocity_command(velocity_cmd)
        t3 = time.perf_counter()
        self._latency.control_ms = (t3 - t2) * 1000
        
        # 4. Apply command to arm
        if arm and controller:
            joint_cmd = controller.get_joint_velocities()
            arm.set_velocity_target(joint_cmd)
            arm.step(1)
        
        # 5. Safety check
        if self._safety and self._current_arm_state:
            is_safe = self._safety.check(
                self._current_arm_state.joint_velocities,
                self._current_arm_state.joint_torques,
                np.zeros(5)  # Force sensing placeholder
            )
            if not is_safe:
                self._mode = SystemMode.EMERGENCY_STOP
                self._safety.trigger_emergency_stop()
        
        # 6. Update feedback
        t4 = time.perf_counter()
        self._update_feedback()
        t5 = time.perf_counter()
        self._latency.feedback_ms = (t5 - t4) * 1000
        
        # Update total latency
        self._latency.total_ms = (t5 - t0) * 1000
        
        # Warn if exceeding latency budget
        if self._latency.total_ms > self.config.max_latency_ms:
            logger.warning(f"Latency exceeded: {self._latency.total_ms:.1f}ms")
    
    def _bci_to_velocity(self, bci_output: NDArray) -> NDArray:
        """Convert BCI decoder output to Cartesian velocity."""
        # Simple linear mapping
        # BCI output: [vx, vy, vz] in normalized units
        # Scale to reasonable velocity
        max_velocity = 0.1  # m/s
        
        if len(bci_output) >= 3:
            velocity = bci_output[:3] * max_velocity
        else:
            velocity = np.zeros(3)
        
        return velocity
    
    def _update_feedback(self) -> None:
        """Update feedback systems."""
        vibro = self._components.get("vibrotactile")
        visual = self._components.get("visual")
        
        if self._current_arm_state:
            # Update visual feedback
            if visual and hasattr(visual, 'update_arm_state'):
                from ..feedback.visual import ArmState as VisArmState
                vis_state = VisArmState(
                    joint_angles=self._current_arm_state.joint_positions
                )
                visual.update_arm_state(vis_state)
        
        if self._current_forces is not None and vibro:
            vibro.set_force(self._current_forces)
    
    def _send_zero_command(self) -> None:
        """Send zero velocity command (safe stop)."""
        arm = self._components.get("arm")
        if arm:
            arm.set_velocity_target(np.zeros(7))
    
    def _on_emergency_stop(self) -> None:
        """Handle emergency stop event."""
        self._mode = SystemMode.EMERGENCY_STOP
        self._send_zero_command()
        logger.critical("Emergency stop activated")
    
    def _notify_state_change(self) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(self._state)
            except:
                pass
    
    def set_mode(self, mode: SystemMode) -> bool:
        """
        Change operating mode.
        
        Args:
            mode: New mode
            
        Returns:
            True if mode change successful
        """
        # Validate transition
        if self._mode == SystemMode.EMERGENCY_STOP and mode != SystemMode.IDLE:
            logger.warning("Must go to IDLE before leaving EMERGENCY_STOP")
            return False
        
        self._mode = mode
        logger.info(f"Mode changed to: {mode.name}")
        return True
    
    def add_state_callback(
        self,
        callback: Callable[[SystemState], None]
    ) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health report."""
        component_health = self._components.check_health()
        
        return {
            "state": self._state.name,
            "mode": self._mode.name,
            "latency_ms": self._latency.total_ms,
            "safety_ok": self._safety.is_safe if self._safety else True,
            "components": component_health
        }
    
    def get_latency(self) -> LatencyMetrics:
        """Get current latency metrics."""
        return self._latency
    
    @property
    def state(self) -> SystemState:
        return self._state
    
    @property
    def mode(self) -> SystemMode:
        return self._mode
    
    @property
    def is_running(self) -> bool:
        return self._running


# Convenience function
def create_simulation_system() -> BionicArmSystem:
    """Create a system configured for simulation."""
    config = SystemConfig(
        name="SimulatedBionicArm",
        mode=SystemMode.SIMULATION,
        use_simulation=True,
        use_simulated_bci=True,
        control_rate_hz=100.0
    )
    return BionicArmSystem(config)


def create_assisted_system() -> BionicArmSystem:
    """Create a system configured for assisted control."""
    config = SystemConfig(
        name="AssistedBionicArm",
        mode=SystemMode.ASSISTED,
        use_simulation=False,
        use_simulated_bci=False,
        enable_safety=True
    )
    return BionicArmSystem(config)
