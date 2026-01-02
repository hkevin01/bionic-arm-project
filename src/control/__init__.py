"""
Control Module
==============

Robotic arm control for the bionic arm project including kinematics,
trajectory planning, grasping primitives, and motor commands.

Key Components:
    - Kinematics: Forward/inverse kinematics for 7-DOF arm
    - Trajectory: Minimum-jerk trajectory generation
    - Grasping: Grasp primitives with force control
    - Controller: Real-time motor control interface

Arm Configuration:
    7-DOF anthropomorphic arm with:
    - Shoulder: 3 DOF (abduction/adduction, flexion/extension, rotation)
    - Elbow: 1 DOF (flexion/extension)
    - Wrist: 3 DOF (pronation/supination, flexion/extension, deviation)
    
    Additional DOF for hand/gripper control.

Author: Bionic Arm Project Team
License: MIT
"""

from .kinematics import (
    DHParameters,
    JointLimits,
    ArmKinematics,
    Transform,
)

from .trajectory import (
    TrajectoryConfig,
    TrajectoryPoint,
    TrajectorySegment,
    TrajectoryGenerator,
    MinimumJerkTrajectory,
)

from .grasping import (
    GraspConfig,
    GraspPrimitive,
    GraspType,
    GraspController,
)

from .controller import (
    ControllerConfig,
    ControlMode,
    MotorCommand,
    ArmController,
)

__version__ = "0.1.0"

__all__ = [
    # Kinematics
    "DHParameters",
    "JointLimits",
    "ArmKinematics",
    "Transform",
    # Trajectory
    "TrajectoryConfig",
    "TrajectoryPoint",
    "TrajectorySegment",
    "TrajectoryGenerator",
    "MinimumJerkTrajectory",
    # Grasping
    "GraspConfig",
    "GraspPrimitive",
    "GraspType",
    "GraspController",
    # Controller
    "ControllerConfig",
    "ControlMode",
    "MotorCommand",
    "ArmController",
]
