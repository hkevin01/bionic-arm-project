"""
Simulation Environment Module
=============================

Manages the simulation world including objects, obstacles,
and interaction targets for the bionic arm.

Features:
    - Object spawning and management
    - Workspace definition
    - Target placement
    - Obstacle configuration
    - Recording and playback

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ObjectType(Enum):
    """Types of simulation objects."""
    CUBE = auto()
    SPHERE = auto()
    CYLINDER = auto()
    MESH = auto()
    PLANE = auto()


class ObjectMaterial(Enum):
    """Material properties for objects."""
    WOOD = auto()
    METAL = auto()
    PLASTIC = auto()
    RUBBER = auto()
    GLASS = auto()
    FOAM = auto()


@dataclass
class MaterialProperties:
    """Physical material properties."""
    density: float = 1000.0       # kg/m^3
    friction: float = 0.5         # Coefficient
    restitution: float = 0.3      # Bounciness
    stiffness: float = 1e6        # Contact stiffness
    damping: float = 100.0        # Contact damping
    
    @classmethod
    def from_material(cls, material: ObjectMaterial) -> 'MaterialProperties':
        """Get properties for a standard material."""
        presets = {
            ObjectMaterial.WOOD: cls(600, 0.4, 0.3, 1e6, 100),
            ObjectMaterial.METAL: cls(7800, 0.3, 0.4, 1e7, 200),
            ObjectMaterial.PLASTIC: cls(950, 0.35, 0.4, 5e5, 50),
            ObjectMaterial.RUBBER: cls(1100, 0.9, 0.7, 1e5, 20),
            ObjectMaterial.GLASS: cls(2500, 0.2, 0.3, 1e7, 300),
            ObjectMaterial.FOAM: cls(50, 0.6, 0.1, 1e4, 10),
        }
        return presets.get(material, cls())


@dataclass
class ObjectConfig:
    """
    Configuration for a simulation object.
    
    Attributes:
        name: Unique identifier
        obj_type: Object geometry type
        position: Initial position (x, y, z) in meters
        orientation: Orientation as quaternion (x, y, z, w)
        dimensions: Size depending on type
        mass: Object mass in kg (0 = static)
        color: RGBA color (0-1)
        material: Material type
        is_graspable: Can be grasped by arm
        is_target: Is a reach target
    """
    name: str = "object"
    obj_type: ObjectType = ObjectType.CUBE
    position: Tuple[float, float, float] = (0.5, 0.0, 0.1)
    orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)
    dimensions: Tuple[float, ...] = (0.05, 0.05, 0.05)
    mass: float = 0.1
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    material: ObjectMaterial = ObjectMaterial.PLASTIC
    is_graspable: bool = True
    is_target: bool = False
    
    def __post_init__(self) -> None:
        if self.mass < 0:
            raise ValueError("mass must be non-negative")
        if len(self.position) != 3:
            raise ValueError("position must have 3 elements")
        if len(self.orientation) != 4:
            raise ValueError("orientation must have 4 elements (quaternion)")


@dataclass
class SimObject:
    """
    Runtime simulation object with state.
    
    Wraps ObjectConfig with physics-related state.
    """
    config: ObjectConfig
    body_id: int = -1
    
    # Current state
    position: NDArray = field(default_factory=lambda: np.zeros(3))
    orientation: NDArray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    linear_velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    
    # Interaction state
    is_grasped: bool = False
    grasp_force: float = 0.0
    contact_points: int = 0
    
    def __post_init__(self) -> None:
        self.position = np.array(self.config.position)
        self.orientation = np.array(self.config.orientation)
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def is_static(self) -> bool:
        return self.config.mass == 0
    
    def get_pose(self) -> Tuple[NDArray, NDArray]:
        """Get current position and orientation."""
        return self.position.copy(), self.orientation.copy()
    
    def distance_to(self, point: NDArray) -> float:
        """Compute distance to a point."""
        return float(np.linalg.norm(self.position - point))


@dataclass
class WorkspaceConfig:
    """
    Workspace boundary definition.
    
    Defines the valid operating region for the arm.
    """
    # Bounds in meters (min, max)
    x_bounds: Tuple[float, float] = (-0.5, 0.8)
    y_bounds: Tuple[float, float] = (-0.5, 0.5)
    z_bounds: Tuple[float, float] = (0.0, 0.8)
    
    # Exclusion zones (e.g., around the base)
    exclusion_zones: List[Tuple[NDArray, float]] = field(default_factory=list)
    
    def contains(self, point: NDArray) -> bool:
        """Check if point is within workspace."""
        x, y, z = point
        if not (self.x_bounds[0] <= x <= self.x_bounds[1]):
            return False
        if not (self.y_bounds[0] <= y <= self.y_bounds[1]):
            return False
        if not (self.z_bounds[0] <= z <= self.z_bounds[1]):
            return False
        
        # Check exclusion zones
        for center, radius in self.exclusion_zones:
            if np.linalg.norm(point - center) < radius:
                return False
        
        return True
    
    def clamp(self, point: NDArray) -> NDArray:
        """Clamp point to workspace bounds."""
        result = point.copy()
        result[0] = np.clip(result[0], self.x_bounds[0], self.x_bounds[1])
        result[1] = np.clip(result[1], self.y_bounds[0], self.y_bounds[1])
        result[2] = np.clip(result[2], self.z_bounds[0], self.z_bounds[1])
        return result
    
    def random_point(self) -> NDArray:
        """Generate random point in workspace."""
        point = np.array([
            np.random.uniform(*self.x_bounds),
            np.random.uniform(*self.y_bounds),
            np.random.uniform(*self.z_bounds)
        ])
        
        # Retry if in exclusion zone
        max_retries = 100
        for _ in range(max_retries):
            if self.contains(point):
                return point
            point = np.array([
                np.random.uniform(*self.x_bounds),
                np.random.uniform(*self.y_bounds),
                np.random.uniform(*self.z_bounds)
            ])
        
        # Fall back to center
        return np.array([
            (self.x_bounds[0] + self.x_bounds[1]) / 2,
            (self.y_bounds[0] + self.y_bounds[1]) / 2,
            (self.z_bounds[0] + self.z_bounds[1]) / 2
        ])


class ObjectFactory:
    """
    Factory for creating simulation objects.
    
    Provides convenient methods for common object types.
    """
    
    _instance_counter: Dict[str, int] = {}
    
    @classmethod
    def _get_unique_name(cls, prefix: str) -> str:
        """Generate unique object name."""
        count = cls._instance_counter.get(prefix, 0)
        cls._instance_counter[prefix] = count + 1
        return f"{prefix}_{count}"
    
    @classmethod
    def create_cube(
        cls,
        size: float = 0.05,
        position: Tuple[float, float, float] = (0.5, 0, 0.1),
        mass: float = 0.1,
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1),
        **kwargs
    ) -> ObjectConfig:
        """Create a cube object."""
        return ObjectConfig(
            name=cls._get_unique_name("cube"),
            obj_type=ObjectType.CUBE,
            position=position,
            dimensions=(size, size, size),
            mass=mass,
            color=color,
            **kwargs
        )
    
    @classmethod
    def create_sphere(
        cls,
        radius: float = 0.03,
        position: Tuple[float, float, float] = (0.5, 0, 0.1),
        mass: float = 0.05,
        color: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1),
        **kwargs
    ) -> ObjectConfig:
        """Create a sphere object."""
        return ObjectConfig(
            name=cls._get_unique_name("sphere"),
            obj_type=ObjectType.SPHERE,
            position=position,
            dimensions=(radius,),
            mass=mass,
            color=color,
            **kwargs
        )
    
    @classmethod
    def create_cylinder(
        cls,
        radius: float = 0.02,
        height: float = 0.1,
        position: Tuple[float, float, float] = (0.5, 0, 0.1),
        mass: float = 0.08,
        color: Tuple[float, float, float, float] = (0.2, 0.2, 0.8, 1),
        **kwargs
    ) -> ObjectConfig:
        """Create a cylinder object."""
        return ObjectConfig(
            name=cls._get_unique_name("cylinder"),
            obj_type=ObjectType.CYLINDER,
            position=position,
            dimensions=(radius, height),
            mass=mass,
            color=color,
            **kwargs
        )
    
    @classmethod
    def create_target(
        cls,
        position: Tuple[float, float, float] = (0.5, 0, 0.3),
        radius: float = 0.02,
        **kwargs
    ) -> ObjectConfig:
        """Create a target marker (visual only)."""
        return ObjectConfig(
            name=cls._get_unique_name("target"),
            obj_type=ObjectType.SPHERE,
            position=position,
            dimensions=(radius,),
            mass=0,  # Static
            color=(0.9, 0.9, 0.2, 0.5),  # Semi-transparent yellow
            is_graspable=False,
            is_target=True,
            **kwargs
        )
    
    @classmethod
    def create_obstacle(
        cls,
        obj_type: ObjectType = ObjectType.CUBE,
        position: Tuple[float, float, float] = (0.3, 0.2, 0.1),
        dimensions: Tuple[float, ...] = (0.1, 0.1, 0.2),
        **kwargs
    ) -> ObjectConfig:
        """Create a static obstacle."""
        return ObjectConfig(
            name=cls._get_unique_name("obstacle"),
            obj_type=obj_type,
            position=position,
            dimensions=dimensions,
            mass=0,  # Static
            color=(0.4, 0.4, 0.4, 1),
            is_graspable=False,
            is_target=False,
            **kwargs
        )


class SimulationEnvironment:
    """
    Main simulation environment manager.
    
    Manages objects, workspace, and environment state.
    
    Example:
        >>> env = SimulationEnvironment()
        >>> 
        >>> # Add objects
        >>> cube = ObjectFactory.create_cube(position=(0.4, 0, 0.1))
        >>> env.add_object(cube)
        >>> 
        >>> # Add target
        >>> target = ObjectFactory.create_target(position=(0.5, 0.1, 0.2))
        >>> env.add_object(target)
        >>> 
        >>> # Query
        >>> graspable = env.get_graspable_objects()
        >>> nearest = env.get_nearest_object([0.45, 0, 0.15])
    """
    
    def __init__(
        self,
        workspace: Optional[WorkspaceConfig] = None,
        physics_backend: Optional[Any] = None
    ) -> None:
        self.workspace = workspace or WorkspaceConfig()
        self._physics = physics_backend
        
        # Object storage
        self._objects: Dict[str, SimObject] = {}
        self._object_counter = 0
        
        # Target management
        self._current_target: Optional[str] = None
        self._target_reached_threshold = 0.02  # 2cm
        
        # Recording
        self._recording = False
        self._recorded_states: List[Dict[str, Any]] = []
        
        logger.info("SimulationEnvironment initialized")
    
    def add_object(self, config: ObjectConfig) -> SimObject:
        """
        Add an object to the environment.
        
        Args:
            config: Object configuration
            
        Returns:
            Created SimObject
        """
        # Ensure unique name
        if config.name in self._objects:
            base_name = config.name
            config.name = f"{base_name}_{self._object_counter}"
            self._object_counter += 1
        
        # Create physics body if backend available
        body_id = -1
        if self._physics is not None:
            body_id = self._create_physics_body(config)
        
        obj = SimObject(config=config, body_id=body_id)
        self._objects[config.name] = obj
        
        # Track if this is a target
        if config.is_target and self._current_target is None:
            self._current_target = config.name
        
        logger.debug(f"Added object: {config.name}")
        return obj
    
    def _create_physics_body(self, config: ObjectConfig) -> int:
        """Create physics body in backend (PyBullet)."""
        try:
            import pybullet as p
            
            # Create collision shape
            if config.obj_type == ObjectType.CUBE:
                col_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[d/2 for d in config.dimensions[:3]]
                )
            elif config.obj_type == ObjectType.SPHERE:
                col_id = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=config.dimensions[0]
                )
            elif config.obj_type == ObjectType.CYLINDER:
                col_id = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=config.dimensions[0],
                    height=config.dimensions[1]
                )
            else:
                col_id = -1
            
            # Create visual shape
            vis_id = p.createVisualShape(
                p.GEOM_BOX if config.obj_type == ObjectType.CUBE else
                p.GEOM_SPHERE if config.obj_type == ObjectType.SPHERE else
                p.GEOM_CYLINDER,
                halfExtents=[d/2 for d in config.dimensions[:3]] if config.obj_type == ObjectType.CUBE else None,
                radius=config.dimensions[0] if config.obj_type in [ObjectType.SPHERE, ObjectType.CYLINDER] else None,
                length=config.dimensions[1] if config.obj_type == ObjectType.CYLINDER else None,
                rgbaColor=config.color
            )
            
            # Create body
            body_id = p.createMultiBody(
                baseMass=config.mass,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=config.position,
                baseOrientation=config.orientation
            )
            
            # Set material properties
            props = MaterialProperties.from_material(config.material)
            p.changeDynamics(
                body_id, -1,
                lateralFriction=props.friction,
                restitution=props.restitution
            )
            
            return body_id
            
        except ImportError:
            return -1
        except Exception as e:
            logger.error(f"Failed to create physics body: {e}")
            return -1
    
    def remove_object(self, name: str) -> bool:
        """
        Remove an object from the environment.
        
        Args:
            name: Object name
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._objects:
            return False
        
        obj = self._objects[name]
        
        # Remove from physics
        if self._physics is not None and obj.body_id >= 0:
            try:
                import pybullet as p
                p.removeBody(obj.body_id)
            except:
                pass
        
        del self._objects[name]
        
        if self._current_target == name:
            self._current_target = None
        
        logger.debug(f"Removed object: {name}")
        return True
    
    def get_object(self, name: str) -> Optional[SimObject]:
        """Get object by name."""
        return self._objects.get(name)
    
    def get_all_objects(self) -> List[SimObject]:
        """Get all objects."""
        return list(self._objects.values())
    
    def get_graspable_objects(self) -> List[SimObject]:
        """Get all graspable objects."""
        return [obj for obj in self._objects.values() 
                if obj.config.is_graspable]
    
    def get_targets(self) -> List[SimObject]:
        """Get all target objects."""
        return [obj for obj in self._objects.values()
                if obj.config.is_target]
    
    def get_nearest_object(
        self,
        point: NDArray,
        graspable_only: bool = False
    ) -> Optional[SimObject]:
        """
        Get nearest object to a point.
        
        Args:
            point: Query point
            graspable_only: Only consider graspable objects
            
        Returns:
            Nearest object or None
        """
        point = np.asarray(point)
        
        objects = self.get_graspable_objects() if graspable_only else self.get_all_objects()
        
        if not objects:
            return None
        
        nearest = min(objects, key=lambda obj: obj.distance_to(point))
        return nearest
    
    def get_current_target(self) -> Optional[SimObject]:
        """Get current target object."""
        if self._current_target:
            return self._objects.get(self._current_target)
        return None
    
    def set_current_target(self, name: str) -> bool:
        """Set current target by name."""
        if name in self._objects and self._objects[name].config.is_target:
            self._current_target = name
            return True
        return False
    
    def check_target_reached(
        self,
        end_effector_pos: NDArray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if end-effector has reached current target.
        
        Args:
            end_effector_pos: Current end-effector position
            threshold: Distance threshold (default: 2cm)
            
        Returns:
            True if target reached
        """
        target = self.get_current_target()
        if target is None:
            return False
        
        threshold = threshold or self._target_reached_threshold
        distance = target.distance_to(end_effector_pos)
        
        return distance < threshold
    
    def update_from_physics(self) -> None:
        """Update all object states from physics backend."""
        if self._physics is None:
            return
        
        try:
            import pybullet as p
            
            for obj in self._objects.values():
                if obj.body_id >= 0:
                    pos, orient = p.getBasePositionAndOrientation(obj.body_id)
                    vel, ang_vel = p.getBaseVelocity(obj.body_id)
                    
                    obj.position = np.array(pos)
                    obj.orientation = np.array(orient)
                    obj.linear_velocity = np.array(vel)
                    obj.angular_velocity = np.array(ang_vel)
                    
                    # Check contacts
                    contacts = p.getContactPoints(bodyA=obj.body_id)
                    obj.contact_points = len(contacts)
                    
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Physics update error: {e}")
        
        # Record if enabled
        if self._recording:
            self._record_state()
    
    def reset(self) -> None:
        """Reset all objects to initial positions."""
        for obj in self._objects.values():
            obj.position = np.array(obj.config.position)
            obj.orientation = np.array(obj.config.orientation)
            obj.linear_velocity = np.zeros(3)
            obj.angular_velocity = np.zeros(3)
            obj.is_grasped = False
            obj.grasp_force = 0.0
            
            # Reset in physics
            if self._physics is not None and obj.body_id >= 0:
                try:
                    import pybullet as p
                    p.resetBasePositionAndOrientation(
                        obj.body_id,
                        obj.config.position,
                        obj.config.orientation
                    )
                    p.resetBaseVelocity(obj.body_id, [0, 0, 0], [0, 0, 0])
                except:
                    pass
        
        logger.info("Environment reset")
    
    def clear(self) -> None:
        """Remove all objects."""
        for name in list(self._objects.keys()):
            self.remove_object(name)
        self._current_target = None
        logger.info("Environment cleared")
    
    # Recording functionality
    def start_recording(self) -> None:
        """Start recording object states."""
        self._recording = True
        self._recorded_states = []
        logger.info("Recording started")
    
    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return states."""
        self._recording = False
        states = self._recorded_states
        self._recorded_states = []
        logger.info(f"Recording stopped: {len(states)} frames")
        return states
    
    def _record_state(self) -> None:
        """Record current state of all objects."""
        import time
        state = {
            "timestamp": time.time(),
            "objects": {}
        }
        for name, obj in self._objects.items():
            state["objects"][name] = {
                "position": obj.position.tolist(),
                "orientation": obj.orientation.tolist(),
                "velocity": obj.linear_velocity.tolist(),
                "is_grasped": obj.is_grasped
            }
        self._recorded_states.append(state)
    
    def save_to_file(self, filepath: str) -> None:
        """Save environment configuration to file."""
        config = {
            "workspace": {
                "x_bounds": self.workspace.x_bounds,
                "y_bounds": self.workspace.y_bounds,
                "z_bounds": self.workspace.z_bounds
            },
            "objects": []
        }
        
        for obj in self._objects.values():
            cfg = obj.config
            config["objects"].append({
                "name": cfg.name,
                "type": cfg.obj_type.name,
                "position": cfg.position,
                "orientation": cfg.orientation,
                "dimensions": cfg.dimensions,
                "mass": cfg.mass,
                "color": cfg.color,
                "material": cfg.material.name,
                "is_graspable": cfg.is_graspable,
                "is_target": cfg.is_target
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Environment saved to {filepath}")
    
    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        physics_backend: Optional[Any] = None
    ) -> 'SimulationEnvironment':
        """Load environment from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        workspace = WorkspaceConfig(
            x_bounds=tuple(config["workspace"]["x_bounds"]),
            y_bounds=tuple(config["workspace"]["y_bounds"]),
            z_bounds=tuple(config["workspace"]["z_bounds"])
        )
        
        env = cls(workspace=workspace, physics_backend=physics_backend)
        
        for obj_data in config["objects"]:
            obj_config = ObjectConfig(
                name=obj_data["name"],
                obj_type=ObjectType[obj_data["type"]],
                position=tuple(obj_data["position"]),
                orientation=tuple(obj_data["orientation"]),
                dimensions=tuple(obj_data["dimensions"]),
                mass=obj_data["mass"],
                color=tuple(obj_data["color"]),
                material=ObjectMaterial[obj_data["material"]],
                is_graspable=obj_data["is_graspable"],
                is_target=obj_data["is_target"]
            )
            env.add_object(obj_config)
        
        logger.info(f"Environment loaded from {filepath}")
        return env
    
    @property
    def object_count(self) -> int:
        """Number of objects in environment."""
        return len(self._objects)


# Convenience function for common setups
def create_tabletop_environment(
    n_objects: int = 3,
    physics_backend: Optional[Any] = None
) -> SimulationEnvironment:
    """
    Create a standard tabletop grasping environment.
    
    Args:
        n_objects: Number of random graspable objects
        physics_backend: Optional physics backend
        
    Returns:
        Configured environment
    """
    workspace = WorkspaceConfig(
        x_bounds=(0.2, 0.7),
        y_bounds=(-0.3, 0.3),
        z_bounds=(0.0, 0.5)
    )
    
    env = SimulationEnvironment(
        workspace=workspace,
        physics_backend=physics_backend
    )
    
    # Add table surface (obstacle)
    table = ObjectFactory.create_obstacle(
        obj_type=ObjectType.CUBE,
        position=(0.45, 0, -0.025),
        dimensions=(0.6, 0.6, 0.05)
    )
    env.add_object(table)
    
    # Add random objects
    for i in range(n_objects):
        x = np.random.uniform(0.3, 0.6)
        y = np.random.uniform(-0.2, 0.2)
        
        # Random object type
        obj_type = np.random.choice([
            ObjectFactory.create_cube,
            ObjectFactory.create_sphere,
            ObjectFactory.create_cylinder
        ])
        
        config = obj_type(position=(x, y, 0.05))
        env.add_object(config)
    
    # Add target
    target = ObjectFactory.create_target(position=(0.5, 0, 0.2))
    env.add_object(target)
    
    return env
