"""
Arm Simulator Module
====================

Physics-based simulation of the 7-DOF bionic arm using PyBullet
or MuJoCo as the physics backend.

Features:
    - Accurate joint dynamics and limits
    - Contact and collision detection
    - Force/torque sensing
    - Visual rendering
    - Real-time and accelerated simulation

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
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PhysicsBackend(Enum):
    """Available physics simulation backends."""
    PYBULLET = auto()
    MUJOCO = auto()
    SIMPLE = auto()  # Basic kinematic simulation


class RenderMode(Enum):
    """Rendering modes."""
    GUI = auto()        # Interactive window
    OFFSCREEN = auto()  # Offscreen rendering
    HEADLESS = auto()   # No rendering


@dataclass
class JointState:
    """State of a single joint."""
    position: float = 0.0      # radians
    velocity: float = 0.0      # rad/s
    torque: float = 0.0        # Nm
    temperature: float = 25.0  # Celsius (simulated)


@dataclass
class ArmState:
    """Complete arm state."""
    joint_positions: NDArray = field(default_factory=lambda: np.zeros(7))
    joint_velocities: NDArray = field(default_factory=lambda: np.zeros(7))
    joint_torques: NDArray = field(default_factory=lambda: np.zeros(7))
    end_effector_pos: NDArray = field(default_factory=lambda: np.zeros(3))
    end_effector_orient: NDArray = field(default_factory=lambda: np.eye(3))
    timestamp_ns: int = 0
    
    def copy(self) -> 'ArmState':
        return ArmState(
            joint_positions=self.joint_positions.copy(),
            joint_velocities=self.joint_velocities.copy(),
            joint_torques=self.joint_torques.copy(),
            end_effector_pos=self.end_effector_pos.copy(),
            end_effector_orient=self.end_effector_orient.copy(),
            timestamp_ns=self.timestamp_ns
        )


@dataclass
class ContactInfo:
    """Contact point information."""
    body_a: int
    body_b: int
    position: NDArray  # Contact point world position
    normal: NDArray    # Contact normal
    force: float       # Normal force magnitude
    link_index: int    # Link in contact (-1 for base)


@dataclass
class SimulatorConfig:
    """
    Configuration for arm simulator.
    
    Attributes:
        backend: Physics engine to use
        render_mode: Visualization mode
        time_step: Simulation time step (seconds)
        gravity: Gravity vector (m/s^2)
        arm_urdf_path: Path to arm URDF/MJCF file
        use_realtime: Sync simulation to wall clock
        joint_damping: Damping coefficients per joint
        joint_friction: Friction coefficients per joint
    """
    backend: PhysicsBackend = PhysicsBackend.PYBULLET
    render_mode: RenderMode = RenderMode.HEADLESS
    time_step: float = 0.001  # 1kHz physics
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    arm_urdf_path: Optional[str] = None
    use_realtime: bool = False
    joint_damping: Tuple[float, ...] = (0.5,) * 7
    joint_friction: Tuple[float, ...] = (0.1,) * 7
    
    # Joint limits (radians)
    joint_lower_limits: Tuple[float, ...] = (-2.9, -1.8, -2.9, -3.1, -2.9, -1.3, -3.1)
    joint_upper_limits: Tuple[float, ...] = (2.9, 1.8, 2.9, 0.0, 2.9, 2.2, 3.1)
    
    # Torque limits (Nm)
    max_joint_torques: Tuple[float, ...] = (87, 87, 87, 87, 12, 12, 12)
    
    # Link masses (kg)
    link_masses: Tuple[float, ...] = (4.0, 4.0, 3.0, 3.0, 1.5, 1.5, 0.5)
    
    def __post_init__(self) -> None:
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.time_step > 0.01:
            logger.warning("Large time step may cause instability")


class SimplePhysics:
    """
    Simple kinematic/dynamic simulation without external dependencies.
    
    Uses basic Euler integration for joint dynamics.
    Good for testing without PyBullet/MuJoCo.
    """
    
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.n_joints = 7
        
        # State
        self._positions = np.zeros(self.n_joints)
        self._velocities = np.zeros(self.n_joints)
        self._accelerations = np.zeros(self.n_joints)
        
        # Simple dynamics model
        self._inertias = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05])
        self._damping = np.array(config.joint_damping)
        self._friction = np.array(config.joint_friction)
        
        # Control
        self._target_positions = np.zeros(self.n_joints)
        self._target_velocities = np.zeros(self.n_joints)
        self._torque_commands = np.zeros(self.n_joints)
        self._control_mode = "position"  # or "velocity" or "torque"
        
        # PD gains for position control
        self._kp = np.array([100, 100, 80, 80, 40, 40, 20])
        self._kd = np.array([10, 10, 8, 8, 4, 4, 2])
        
    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        # Compute applied torques based on control mode
        if self._control_mode == "position":
            pos_error = self._target_positions - self._positions
            vel_error = -self._velocities
            torques = self._kp * pos_error + self._kd * vel_error
        elif self._control_mode == "velocity":
            vel_error = self._target_velocities - self._velocities
            torques = self._kd * vel_error * 10
        else:
            torques = self._torque_commands.copy()
        
        # Apply torque limits
        max_torques = np.array(self.config.max_joint_torques)
        torques = np.clip(torques, -max_torques, max_torques)
        
        # Dynamics: I * a = tau - d * v - f * sign(v)
        friction_torque = self._friction * np.sign(self._velocities)
        damping_torque = self._damping * self._velocities
        net_torque = torques - damping_torque - friction_torque
        
        self._accelerations = net_torque / self._inertias
        
        # Euler integration
        self._velocities += self._accelerations * dt
        self._positions += self._velocities * dt
        
        # Apply joint limits
        lower = np.array(self.config.joint_lower_limits)
        upper = np.array(self.config.joint_upper_limits)
        
        # Bounce at limits
        for i in range(self.n_joints):
            if self._positions[i] < lower[i]:
                self._positions[i] = lower[i]
                self._velocities[i] = abs(self._velocities[i]) * 0.1
            elif self._positions[i] > upper[i]:
                self._positions[i] = upper[i]
                self._velocities[i] = -abs(self._velocities[i]) * 0.1
    
    def set_position_target(self, positions: NDArray) -> None:
        self._target_positions = np.asarray(positions)
        self._control_mode = "position"
    
    def set_velocity_target(self, velocities: NDArray) -> None:
        self._target_velocities = np.asarray(velocities)
        self._control_mode = "velocity"
    
    def set_torques(self, torques: NDArray) -> None:
        self._torque_commands = np.asarray(torques)
        self._control_mode = "torque"
    
    def get_positions(self) -> NDArray:
        return self._positions.copy()
    
    def get_velocities(self) -> NDArray:
        return self._velocities.copy()
    
    def reset(self, positions: Optional[NDArray] = None) -> None:
        if positions is not None:
            self._positions = np.asarray(positions)
        else:
            self._positions = np.zeros(self.n_joints)
        self._velocities = np.zeros(self.n_joints)
        self._accelerations = np.zeros(self.n_joints)


class PyBulletBackend:
    """
    PyBullet physics simulation backend.
    
    Provides high-fidelity physics simulation with
    collision detection and force sensing.
    """
    
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self._client = None
        self._arm_id = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize PyBullet simulation."""
        try:
            import pybullet as p
            import pybullet_data
            
            # Create physics client
            if self.config.render_mode == RenderMode.GUI:
                self._client = p.connect(p.GUI)
            else:
                self._client = p.connect(p.DIRECT)
            
            # Set physics parameters
            p.setGravity(*self.config.gravity, physicsClientId=self._client)
            p.setTimeStep(self.config.time_step, physicsClientId=self._client)
            
            # Add ground plane
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self._client)
            
            # Load arm model
            if self.config.arm_urdf_path and Path(self.config.arm_urdf_path).exists():
                self._arm_id = p.loadURDF(
                    self.config.arm_urdf_path,
                    basePosition=[0, 0, 0],
                    useFixedBase=True,
                    physicsClientId=self._client
                )
            else:
                # Create simple arm procedurally
                self._arm_id = self._create_simple_arm(p)
            
            self._initialized = True
            logger.info("PyBullet initialized successfully")
            return True
            
        except ImportError:
            logger.error("PyBullet not installed")
            return False
        except Exception as e:
            logger.error(f"PyBullet init failed: {e}")
            return False
    
    def _create_simple_arm(self, p) -> int:
        """Create a simple 7-DOF arm procedurally."""
        # Link dimensions
        link_lengths = [0.1, 0.28, 0.26, 0.1, 0.1, 0.05, 0.05]
        link_radii = [0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01]
        
        # Create collision shapes
        col_shapes = []
        vis_shapes = []
        for length, radius in zip(link_lengths, link_radii):
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=length,
                physicsClientId=self._client
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=length,
                rgbaColor=[0.7, 0.7, 0.8, 1.0],
                physicsClientId=self._client
            )
            col_shapes.append(col)
            vis_shapes.append(vis)
        
        # Link positions relative to parent
        link_positions = [[0, 0, l] for l in link_lengths]
        
        # Create multi-body
        masses = list(self.config.link_masses)
        
        arm_id = p.createMultiBody(
            baseMass=0,  # Fixed base
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0.5],
            linkMasses=masses,
            linkCollisionShapeIndices=col_shapes,
            linkVisualShapeIndices=vis_shapes,
            linkPositions=link_positions,
            linkOrientations=[[0, 0, 0, 1]] * 7,
            linkInertialFramePositions=[[0, 0, 0]] * 7,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 7,
            linkParentIndices=[0, 1, 2, 3, 4, 5, 6],
            linkJointTypes=[p.JOINT_REVOLUTE] * 7,
            linkJointAxis=[[0, 0, 1], [0, 1, 0], [0, 0, 1], 
                          [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]],
            physicsClientId=self._client
        )
        
        return arm_id
    
    def step(self) -> None:
        """Advance simulation one step."""
        if not self._initialized:
            return
        import pybullet as p
        p.stepSimulation(physicsClientId=self._client)
    
    def get_joint_states(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Get joint positions, velocities, torques."""
        if not self._initialized:
            return np.zeros(7), np.zeros(7), np.zeros(7)
        
        import pybullet as p
        states = []
        for i in range(7):
            state = p.getJointState(
                self._arm_id, i,
                physicsClientId=self._client
            )
            states.append(state)
        
        positions = np.array([s[0] for s in states])
        velocities = np.array([s[1] for s in states])
        torques = np.array([s[3] for s in states])
        
        return positions, velocities, torques
    
    def set_position_target(self, positions: NDArray) -> None:
        """Set position targets for all joints."""
        if not self._initialized:
            return
        
        import pybullet as p
        for i, pos in enumerate(positions):
            p.setJointMotorControl2(
                self._arm_id, i,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=self.config.max_joint_torques[i],
                physicsClientId=self._client
            )
    
    def set_velocity_target(self, velocities: NDArray) -> None:
        """Set velocity targets for all joints."""
        if not self._initialized:
            return
        
        import pybullet as p
        for i, vel in enumerate(velocities):
            p.setJointMotorControl2(
                self._arm_id, i,
                p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=self.config.max_joint_torques[i],
                physicsClientId=self._client
            )
    
    def set_torques(self, torques: NDArray) -> None:
        """Apply torques directly to joints."""
        if not self._initialized:
            return
        
        import pybullet as p
        for i, torque in enumerate(torques):
            p.setJointMotorControl2(
                self._arm_id, i,
                p.TORQUE_CONTROL,
                force=torque,
                physicsClientId=self._client
            )
    
    def get_end_effector_state(self) -> Tuple[NDArray, NDArray]:
        """Get end-effector position and orientation."""
        if not self._initialized:
            return np.zeros(3), np.eye(3)
        
        import pybullet as p
        state = p.getLinkState(
            self._arm_id, 6,  # Last link
            physicsClientId=self._client
        )
        pos = np.array(state[0])
        quat = state[1]
        
        # Quaternion to rotation matrix
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        return pos, rot
    
    def get_contacts(self) -> List[ContactInfo]:
        """Get all contact points."""
        if not self._initialized:
            return []
        
        import pybullet as p
        contacts = []
        contact_points = p.getContactPoints(
            bodyA=self._arm_id,
            physicsClientId=self._client
        )
        
        for cp in contact_points:
            contacts.append(ContactInfo(
                body_a=cp[1],
                body_b=cp[2],
                position=np.array(cp[5]),
                normal=np.array(cp[7]),
                force=cp[9],
                link_index=cp[3]
            ))
        
        return contacts
    
    def reset(self, positions: Optional[NDArray] = None) -> None:
        """Reset arm to initial or specified positions."""
        if not self._initialized:
            return
        
        import pybullet as p
        if positions is None:
            positions = np.zeros(7)
        
        for i, pos in enumerate(positions):
            p.resetJointState(
                self._arm_id, i, pos,
                physicsClientId=self._client
            )
    
    def cleanup(self) -> None:
        """Cleanup PyBullet resources."""
        if self._client is not None:
            import pybullet as p
            p.disconnect(self._client)
            self._client = None
            self._initialized = False


class ArmSimulator:
    """
    Main arm simulator class.
    
    Provides unified interface to physics backends
    and manages simulation state.
    
    Example:
        >>> config = SimulatorConfig(
        ...     backend=PhysicsBackend.PYBULLET,
        ...     render_mode=RenderMode.GUI
        ... )
        >>> sim = ArmSimulator(config)
        >>> sim.initialize()
        >>> 
        >>> # Position control
        >>> target = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.5, 0.0])
        >>> sim.set_position_target(target)
        >>> 
        >>> # Step simulation
        >>> for _ in range(1000):
        ...     sim.step()
        ...     state = sim.get_state()
    """
    
    def __init__(
        self,
        config: Optional[SimulatorConfig] = None
    ) -> None:
        self.config = config or SimulatorConfig()
        
        # Backend selection
        self._backend = None
        self._simple_physics = None
        
        # State
        self._state = ArmState()
        self._simulation_time = 0.0
        self._step_count = 0
        self._initialized = False
        
        # Real-time sync
        self._last_step_time = time.perf_counter()
        
        # Callbacks
        self._step_callbacks: List[Callable[[ArmState], None]] = []
        self._contact_callbacks: List[Callable[[List[ContactInfo]], None]] = []
        
        # Threading for real-time mode
        self._running = False
        self._sim_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        
        logger.info(f"ArmSimulator created: {self.config.backend.name}")
    
    def initialize(self) -> bool:
        """Initialize simulation backend."""
        if self._initialized:
            return True
        
        if self.config.backend == PhysicsBackend.PYBULLET:
            self._backend = PyBulletBackend(self.config)
            success = self._backend.initialize()
            if not success:
                logger.warning("PyBullet failed, falling back to simple physics")
                self._simple_physics = SimplePhysics(self.config)
        elif self.config.backend == PhysicsBackend.SIMPLE:
            self._simple_physics = SimplePhysics(self.config)
            success = True
        else:
            logger.warning(f"Backend {self.config.backend} not implemented, using simple")
            self._simple_physics = SimplePhysics(self.config)
            success = True
        
        self._initialized = True
        return success
    
    def step(self, n_steps: int = 1) -> None:
        """
        Advance simulation.
        
        Args:
            n_steps: Number of physics steps to take
        """
        if not self._initialized:
            return
        
        for _ in range(n_steps):
            if self.config.use_realtime:
                # Wait for real-time sync
                target_time = self._last_step_time + self.config.time_step
                now = time.perf_counter()
                if now < target_time:
                    time.sleep(target_time - now)
                self._last_step_time = time.perf_counter()
            
            # Step physics
            if self._backend is not None:
                self._backend.step()
            elif self._simple_physics is not None:
                self._simple_physics.step(self.config.time_step)
            
            self._simulation_time += self.config.time_step
            self._step_count += 1
            
            # Update state
            self._update_state()
            
            # Check contacts
            contacts = self.get_contacts()
            if contacts:
                for callback in self._contact_callbacks:
                    callback(contacts)
            
            # Step callbacks
            for callback in self._step_callbacks:
                callback(self._state)
    
    def _update_state(self) -> None:
        """Update internal state from physics."""
        with self._state_lock:
            if self._backend is not None:
                pos, vel, torque = self._backend.get_joint_states()
                ee_pos, ee_rot = self._backend.get_end_effector_state()
            elif self._simple_physics is not None:
                pos = self._simple_physics.get_positions()
                vel = self._simple_physics.get_velocities()
                torque = np.zeros(7)
                ee_pos = self._compute_fk_simple(pos)
                ee_rot = np.eye(3)
            else:
                return
            
            self._state = ArmState(
                joint_positions=pos,
                joint_velocities=vel,
                joint_torques=torque,
                end_effector_pos=ee_pos,
                end_effector_orient=ee_rot,
                timestamp_ns=int(self._simulation_time * 1e9)
            )
    
    def _compute_fk_simple(self, positions: NDArray) -> NDArray:
        """Simple forward kinematics for end-effector position."""
        # Simplified 2D-like FK
        link_lengths = [0.1, 0.28, 0.26, 0.1, 0.1, 0.05, 0.05]
        x, y, z = 0.0, 0.0, 0.5  # Base height
        cumulative = 0.0
        
        for i, (length, angle) in enumerate(zip(link_lengths, positions)):
            if i % 2 == 0:
                cumulative += angle
            x += length * np.cos(cumulative)
            z += length * np.sin(cumulative) * 0.5
        
        return np.array([x, y, z])
    
    def get_state(self) -> ArmState:
        """Get current arm state."""
        with self._state_lock:
            return self._state.copy()
    
    def set_position_target(self, positions: NDArray) -> None:
        """Set joint position targets."""
        positions = np.asarray(positions)
        
        # Clamp to limits
        lower = np.array(self.config.joint_lower_limits)
        upper = np.array(self.config.joint_upper_limits)
        positions = np.clip(positions, lower, upper)
        
        if self._backend is not None:
            self._backend.set_position_target(positions)
        elif self._simple_physics is not None:
            self._simple_physics.set_position_target(positions)
    
    def set_velocity_target(self, velocities: NDArray) -> None:
        """Set joint velocity targets."""
        if self._backend is not None:
            self._backend.set_velocity_target(velocities)
        elif self._simple_physics is not None:
            self._simple_physics.set_velocity_target(velocities)
    
    def set_torques(self, torques: NDArray) -> None:
        """Apply joint torques directly."""
        torques = np.asarray(torques)
        max_t = np.array(self.config.max_joint_torques)
        torques = np.clip(torques, -max_t, max_t)
        
        if self._backend is not None:
            self._backend.set_torques(torques)
        elif self._simple_physics is not None:
            self._simple_physics.set_torques(torques)
    
    def get_contacts(self) -> List[ContactInfo]:
        """Get current contact points."""
        if self._backend is not None:
            return self._backend.get_contacts()
        return []
    
    def reset(self, positions: Optional[NDArray] = None) -> None:
        """Reset simulation to initial state."""
        self._simulation_time = 0.0
        self._step_count = 0
        
        if self._backend is not None:
            self._backend.reset(positions)
        elif self._simple_physics is not None:
            self._simple_physics.reset(positions)
        
        self._update_state()
    
    def start_realtime(self) -> None:
        """Start real-time simulation in background thread."""
        if self._running:
            return
        
        self.config.use_realtime = True
        self._running = True
        self._stop_event.clear()
        
        self._sim_thread = threading.Thread(
            target=self._realtime_loop,
            daemon=True,
            name="ArmSimulator"
        )
        self._sim_thread.start()
        logger.info("Real-time simulation started")
    
    def stop_realtime(self) -> None:
        """Stop real-time simulation."""
        self._running = False
        self._stop_event.set()
        
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)
        
        logger.info("Real-time simulation stopped")
    
    def _realtime_loop(self) -> None:
        """Background loop for real-time simulation."""
        while not self._stop_event.is_set():
            self.step(1)
    
    def add_step_callback(
        self,
        callback: Callable[[ArmState], None]
    ) -> None:
        """Register callback for each simulation step."""
        self._step_callbacks.append(callback)
    
    def add_contact_callback(
        self,
        callback: Callable[[List[ContactInfo]], None]
    ) -> None:
        """Register callback for contact events."""
        self._contact_callbacks.append(callback)
    
    def cleanup(self) -> None:
        """Cleanup simulation resources."""
        self.stop_realtime()
        
        if self._backend is not None:
            self._backend.cleanup()
        
        self._initialized = False
        logger.info("Simulator cleanup complete")
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds."""
        return self._simulation_time
    
    @property
    def step_count(self) -> int:
        """Total number of simulation steps."""
        return self._step_count
    
    @property
    def is_initialized(self) -> bool:
        """Check if simulator is initialized."""
        return self._initialized
    
    @property
    def is_running(self) -> bool:
        """Check if real-time simulation is running."""
        return self._running


# Convenience function
def create_simulator(
    backend: str = "pybullet",
    render: str = "headless",
    **kwargs
) -> ArmSimulator:
    """
    Create simulator with specified backend and render mode.
    
    Args:
        backend: "pybullet", "mujoco", or "simple"
        render: "gui", "offscreen", or "headless"
        **kwargs: Additional config options
        
    Returns:
        Configured ArmSimulator instance
    """
    backend_map = {
        "pybullet": PhysicsBackend.PYBULLET,
        "mujoco": PhysicsBackend.MUJOCO,
        "simple": PhysicsBackend.SIMPLE
    }
    render_map = {
        "gui": RenderMode.GUI,
        "offscreen": RenderMode.OFFSCREEN,
        "headless": RenderMode.HEADLESS
    }
    
    config = SimulatorConfig(
        backend=backend_map.get(backend.lower(), PhysicsBackend.PYBULLET),
        render_mode=render_map.get(render.lower(), RenderMode.HEADLESS),
        **kwargs
    )
    
    return ArmSimulator(config)
