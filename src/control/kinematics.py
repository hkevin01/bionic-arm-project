"""
Kinematics Module
=================

Forward and inverse kinematics for a 7-DOF anthropomorphic robotic arm.
Uses Denavit-Hartenberg (DH) parameter convention for kinematic modeling.

Mathematical Background:

    Denavit-Hartenberg Convention:
        Each joint is described by 4 parameters:
        - d: offset along previous z-axis
        - θ: angle about previous z-axis (joint variable for revolute)
        - a: length of common normal (link length)
        - α: angle about common normal (link twist)
        
        Transformation matrix for each link:
        T_i = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)
        
             ⎡ cθ  -sθcα   sθsα   a·cθ ⎤
        T = ⎢ sθ   cθcα  -cθsα   a·sθ ⎥
             ⎢ 0    sα     cα     d    ⎥
             ⎣ 0    0      0      1    ⎦

    Forward Kinematics:
        End-effector pose = T_base · T_1 · T_2 · ... · T_n
        
    Inverse Kinematics:
        Given desired pose, solve for joint angles.
        Uses numerical methods (damped least squares) for 7-DOF.

Arm Specifications:
    - 7 revolute joints
    - Shoulder: 3 DOF (300mm from base)
    - Elbow: 1 DOF (280mm from shoulder)
    - Wrist: 3 DOF (260mm from elbow)
    - Total reach: ~800mm

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.floating]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DHParameters:
    """
    Denavit-Hartenberg parameters for a single link.
    
    Attributes:
        d: Offset along z-axis (m)
        theta: Joint angle (rad) - variable for revolute joints
        a: Link length (m)
        alpha: Link twist angle (rad)
        is_revolute: True for revolute, False for prismatic
        theta_offset: Constant offset for theta
    """
    d: float = 0.0
    theta: float = 0.0
    a: float = 0.0
    alpha: float = 0.0
    is_revolute: bool = True
    theta_offset: float = 0.0
    
    def get_theta(self, q: float) -> float:
        """Get total theta angle including offset."""
        return q + self.theta_offset if self.is_revolute else self.theta
    
    def get_d(self, q: float) -> float:
        """Get d parameter (variable for prismatic joints)."""
        return q if not self.is_revolute else self.d


@dataclass
class JointLimits:
    """
    Joint limits and constraints.
    
    Attributes:
        lower: Lower position limit (rad or m)
        upper: Upper position limit (rad or m)
        velocity: Maximum velocity (rad/s or m/s)
        acceleration: Maximum acceleration (rad/s² or m/s²)
        torque: Maximum torque (Nm) or force (N)
    """
    lower: float = -np.pi
    upper: float = np.pi
    velocity: float = 2.0
    acceleration: float = 10.0
    torque: float = 50.0
    
    def __post_init__(self) -> None:
        """Validate limits."""
        if self.lower >= self.upper:
            raise ValueError(f"lower ({self.lower}) must be < upper ({self.upper})")
        if self.velocity <= 0:
            raise ValueError("velocity must be positive")
    
    def clamp(self, value: float) -> float:
        """Clamp value to position limits."""
        return float(np.clip(value, self.lower, self.upper))
    
    def is_within(self, value: float, margin: float = 0.0) -> bool:
        """Check if value is within limits with optional margin."""
        return (self.lower + margin) <= value <= (self.upper - margin)
    
    def range(self) -> float:
        """Get joint range."""
        return self.upper - self.lower
    
    def center(self) -> float:
        """Get center of joint range."""
        return (self.lower + self.upper) / 2


@dataclass
class Transform:
    """
    Rigid body transformation (SE(3)).
    
    Represents position and orientation in 3D space using
    a 4x4 homogeneous transformation matrix.
    
    Attributes:
        matrix: 4x4 homogeneous transformation matrix
    """
    matrix: FloatArray = field(default_factory=lambda: np.eye(4))
    
    def __post_init__(self) -> None:
        """Ensure matrix is proper shape."""
        self.matrix = np.asarray(self.matrix).reshape(4, 4)
    
    @classmethod
    def from_position_rotation(
        cls,
        position: FloatArray,
        rotation: FloatArray
    ) -> "Transform":
        """
        Create transform from position and rotation matrix.
        
        Args:
            position: [x, y, z] position
            rotation: 3x3 rotation matrix
            
        Returns:
            Transform instance
        """
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position
        return cls(matrix)
    
    @classmethod
    def from_position_quaternion(
        cls,
        position: FloatArray,
        quaternion: FloatArray
    ) -> "Transform":
        """
        Create transform from position and quaternion.
        
        Args:
            position: [x, y, z] position
            quaternion: [w, x, y, z] quaternion
            
        Returns:
            Transform instance
        """
        rotation = cls._quaternion_to_rotation_matrix(quaternion)
        return cls.from_position_rotation(position, rotation)
    
    @classmethod
    def from_position_rpy(
        cls,
        position: FloatArray,
        rpy: FloatArray
    ) -> "Transform":
        """
        Create transform from position and roll-pitch-yaw angles.
        
        Args:
            position: [x, y, z] position
            rpy: [roll, pitch, yaw] in radians
            
        Returns:
            Transform instance
        """
        rotation = cls._rpy_to_rotation_matrix(rpy)
        return cls.from_position_rotation(position, rotation)
    
    @staticmethod
    def _quaternion_to_rotation_matrix(q: FloatArray) -> FloatArray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    @staticmethod
    def _rpy_to_rotation_matrix(rpy: FloatArray) -> FloatArray:
        """Convert roll-pitch-yaw to 3x3 rotation matrix (XYZ convention)."""
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
    
    @property
    def position(self) -> FloatArray:
        """Get position vector [x, y, z]."""
        return self.matrix[:3, 3].copy()
    
    @property
    def rotation(self) -> FloatArray:
        """Get 3x3 rotation matrix."""
        return self.matrix[:3, :3].copy()
    
    @property
    def quaternion(self) -> FloatArray:
        """Get quaternion [w, x, y, z] from rotation matrix."""
        R = self.rotation
        
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    @property
    def rpy(self) -> FloatArray:
        """Get roll-pitch-yaw angles from rotation matrix."""
        R = self.rotation
        
        # Handle gimbal lock
        if abs(R[2, 0]) >= 1.0 - 1e-6:
            yaw = 0.0
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
            yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        
        return np.array([roll, pitch, yaw])
    
    def inverse(self) -> "Transform":
        """Compute inverse transform."""
        R = self.rotation
        p = self.position
        
        R_inv = R.T
        p_inv = -R_inv @ p
        
        return Transform.from_position_rotation(p_inv, R_inv)
    
    def __matmul__(self, other: "Transform") -> "Transform":
        """Matrix multiplication (composition) of transforms."""
        return Transform(self.matrix @ other.matrix)
    
    def transform_point(self, point: FloatArray) -> FloatArray:
        """Transform a 3D point."""
        homogeneous = np.array([point[0], point[1], point[2], 1.0])
        result = self.matrix @ homogeneous
        return result[:3]
    
    def transform_vector(self, vector: FloatArray) -> FloatArray:
        """Transform a 3D vector (rotation only)."""
        return self.rotation @ vector
    
    def distance_to(self, other: "Transform") -> Tuple[float, float]:
        """
        Compute distance to another transform.
        
        Returns:
            Tuple of (position_distance, rotation_distance)
            Rotation distance is angle in radians.
        """
        pos_dist = float(np.linalg.norm(self.position - other.position))
        
        # Rotation distance using quaternion geodesic
        q1 = self.quaternion
        q2 = other.quaternion
        
        dot = abs(np.dot(q1, q2))
        dot = min(1.0, dot)  # Clamp for numerical stability
        rot_dist = 2.0 * np.arccos(dot)
        
        return pos_dist, rot_dist


# =============================================================================
# Arm Kinematics Class
# =============================================================================

class ArmKinematics:
    """
    Complete kinematic model for a 7-DOF anthropomorphic arm.
    
    Provides forward kinematics, inverse kinematics, and Jacobian
    computation for motion planning and control.
    
    Default Arm Configuration:
        Joint 0: Shoulder abduction/adduction
        Joint 1: Shoulder flexion/extension
        Joint 2: Shoulder rotation
        Joint 3: Elbow flexion/extension
        Joint 4: Wrist pronation/supination
        Joint 5: Wrist flexion/extension
        Joint 6: Wrist deviation
    
    Example:
        >>> arm = ArmKinematics.default_arm()
        >>> q = np.zeros(7)  # Home position
        >>> ee_pose = arm.forward_kinematics(q)
        >>> print(f"End-effector at: {ee_pose.position}")
        >>> 
        >>> # Inverse kinematics
        >>> target = Transform.from_position_rpy([0.5, 0.0, 0.3], [0, 0, 0])
        >>> q_solution, success = arm.inverse_kinematics(target, q)
    """
    
    # Standard link lengths (meters)
    DEFAULT_LINK_LENGTHS = {
        "base_to_shoulder": 0.1,
        "shoulder_offset": 0.05,
        "upper_arm": 0.28,
        "forearm": 0.26,
        "wrist": 0.08,
        "hand": 0.12
    }
    
    def __init__(
        self,
        dh_params: List[DHParameters],
        joint_limits: Optional[List[JointLimits]] = None,
        base_transform: Optional[Transform] = None,
        tool_transform: Optional[Transform] = None
    ) -> None:
        """
        Initialize arm kinematics.
        
        Args:
            dh_params: List of DH parameters for each joint
            joint_limits: Optional joint limits
            base_transform: Transform from world to base frame
            tool_transform: Transform from flange to tool center point
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)
        
        # Set default joint limits if not provided
        if joint_limits is None:
            self.joint_limits = [JointLimits() for _ in range(self.n_joints)]
        else:
            if len(joint_limits) != self.n_joints:
                raise ValueError("joint_limits length must match dh_params")
            self.joint_limits = joint_limits
        
        # Base and tool transforms
        self.base_transform = base_transform or Transform()
        self.tool_transform = tool_transform or Transform()
        
        # Cache for IK
        self._last_solution: Optional[FloatArray] = None
        
        logger.info(f"ArmKinematics initialized with {self.n_joints} joints")
    
    @classmethod
    def default_arm(cls) -> "ArmKinematics":
        """
        Create a default 7-DOF anthropomorphic arm.
        
        Returns:
            ArmKinematics with standard prosthetic arm configuration
        """
        L = cls.DEFAULT_LINK_LENGTHS
        
        # DH parameters for 7-DOF arm
        # [d, theta, a, alpha]
        dh_params = [
            DHParameters(d=L["base_to_shoulder"], theta=0, a=0, alpha=-np.pi/2),  # Shoulder ab/ad
            DHParameters(d=0, theta=0, a=0, alpha=np.pi/2),                        # Shoulder flex/ext
            DHParameters(d=L["upper_arm"], theta=0, a=0, alpha=-np.pi/2),         # Shoulder rot
            DHParameters(d=0, theta=0, a=0, alpha=np.pi/2),                        # Elbow
            DHParameters(d=L["forearm"], theta=0, a=0, alpha=-np.pi/2),           # Wrist pro/sup
            DHParameters(d=0, theta=0, a=0, alpha=np.pi/2),                        # Wrist flex/ext
            DHParameters(d=L["wrist"], theta=0, a=0, alpha=0),                    # Wrist dev
        ]
        
        # Joint limits (conservative for safety)
        joint_limits = [
            JointLimits(lower=-np.pi/2, upper=np.pi/2),     # Shoulder ab/ad
            JointLimits(lower=-np.pi/3, upper=np.pi),       # Shoulder flex/ext
            JointLimits(lower=-np.pi, upper=np.pi),         # Shoulder rotation
            JointLimits(lower=0, upper=2.5),                # Elbow (0-140 deg)
            JointLimits(lower=-np.pi/2, upper=np.pi/2),     # Wrist pro/sup
            JointLimits(lower=-np.pi/3, upper=np.pi/3),     # Wrist flex/ext
            JointLimits(lower=-np.pi/4, upper=np.pi/4),     # Wrist deviation
        ]
        
        return cls(dh_params, joint_limits)
    
    def _dh_transform(self, dh: DHParameters, q: float) -> FloatArray:
        """
        Compute transformation matrix for a single joint.
        
        Args:
            dh: DH parameters for the joint
            q: Joint variable
            
        Returns:
            4x4 transformation matrix
        """
        theta = dh.get_theta(q)
        d = dh.get_d(q)
        a = dh.a
        alpha = dh.alpha
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])
    
    def forward_kinematics(
        self,
        q: FloatArray,
        return_all_transforms: bool = False
    ) -> Transform:
        """
        Compute forward kinematics.
        
        Args:
            q: Joint angles (rad), shape (n_joints,)
            return_all_transforms: If True, return list of all link transforms
            
        Returns:
            End-effector transform (or list of all transforms)
            
        Raises:
            ValueError: If q has wrong length
        """
        q = np.asarray(q).flatten()
        
        if len(q) != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joints, got {len(q)}")
        
        T = self.base_transform.matrix.copy()
        transforms = [Transform(T)]
        
        for i, (dh, qi) in enumerate(zip(self.dh_params, q)):
            T_i = self._dh_transform(dh, qi)
            T = T @ T_i
            transforms.append(Transform(T.copy()))
        
        # Apply tool transform
        T = T @ self.tool_transform.matrix
        
        if return_all_transforms:
            return transforms
        
        return Transform(T)
    
    def jacobian(
        self,
        q: FloatArray,
        reference_frame: str = "base"
    ) -> FloatArray:
        """
        Compute geometric Jacobian.
        
        The Jacobian relates joint velocities to end-effector
        linear and angular velocities:
        
            [v]   [J_v]
            [ω] = [J_ω] · q̇
        
        Args:
            q: Joint angles (rad)
            reference_frame: "base" or "tool"
            
        Returns:
            6 x n_joints Jacobian matrix
        """
        q = np.asarray(q).flatten()
        
        # Get all transforms
        transforms = self.forward_kinematics(q, return_all_transforms=True)
        
        # End-effector position
        p_ee = transforms[-1].position
        
        J = np.zeros((6, self.n_joints))
        
        for i in range(self.n_joints):
            # Get transform up to joint i
            T_i = transforms[i].matrix
            
            # z-axis of joint i
            z_i = T_i[:3, 2]
            
            # Position of joint i
            p_i = T_i[:3, 3]
            
            if self.dh_params[i].is_revolute:
                # Revolute joint
                J[:3, i] = np.cross(z_i, p_ee - p_i)  # Linear velocity
                J[3:, i] = z_i                          # Angular velocity
            else:
                # Prismatic joint
                J[:3, i] = z_i
                J[3:, i] = 0
        
        return J
    
    def jacobian_pinv(
        self,
        q: FloatArray,
        damping: float = 0.01
    ) -> FloatArray:
        """
        Compute damped pseudo-inverse of Jacobian.
        
        Uses Damped Least Squares (DLS) for numerical stability:
            J^† = J^T (J J^T + λ² I)^{-1}
        
        Args:
            q: Joint angles
            damping: Damping factor λ
            
        Returns:
            n_joints x 6 pseudo-inverse Jacobian
        """
        J = self.jacobian(q)
        
        # Damped least squares
        JJT = J @ J.T
        damped = JJT + (damping ** 2) * np.eye(6)
        
        return J.T @ np.linalg.inv(damped)
    
    def inverse_kinematics(
        self,
        target: Transform,
        q_init: Optional[FloatArray] = None,
        max_iterations: int = 100,
        position_tolerance: float = 1e-4,
        rotation_tolerance: float = 1e-3,
        damping: float = 0.05,
        use_null_space: bool = True
    ) -> Tuple[FloatArray, bool]:
        """
        Compute inverse kinematics using damped least squares.
        
        Solves for joint angles that achieve the target end-effector pose
        using numerical optimization.
        
        Null Space Usage:
            With 7 joints and 6 task dimensions, there is 1 DOF of redundancy.
            This can be used to optimize secondary objectives:
            - Stay close to joint centers
            - Avoid singularities
            - Manipulability optimization
        
        Args:
            target: Desired end-effector pose
            q_init: Initial joint angles (uses last solution if None)
            max_iterations: Maximum number of iterations
            position_tolerance: Position error tolerance (m)
            rotation_tolerance: Rotation error tolerance (rad)
            damping: Damping factor for stability
            use_null_space: Whether to use null space for secondary objectives
            
        Returns:
            Tuple of (joint_angles, success)
        """
        # Initial guess
        if q_init is not None:
            q = np.asarray(q_init).flatten().copy()
        elif self._last_solution is not None:
            q = self._last_solution.copy()
        else:
            q = np.array([lim.center() for lim in self.joint_limits])
        
        for iteration in range(max_iterations):
            # Current end-effector pose
            current = self.forward_kinematics(q)
            
            # Position error
            pos_error = target.position - current.position
            
            # Rotation error (using rotation matrices)
            R_current = current.rotation
            R_target = target.rotation
            R_error = R_target @ R_current.T
            
            # Convert rotation error to axis-angle
            angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            if abs(angle) < 1e-10:
                axis_angle = np.zeros(3)
            else:
                # Rodrigues formula inverse
                axis = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1]
                ])
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-10:
                    axis = axis / axis_norm * angle
                else:
                    axis = np.zeros(3)
                axis_angle = axis
            
            # Check convergence
            pos_err_norm = np.linalg.norm(pos_error)
            rot_err_norm = np.linalg.norm(axis_angle)
            
            if pos_err_norm < position_tolerance and rot_err_norm < rotation_tolerance:
                self._last_solution = q.copy()
                logger.debug(f"IK converged in {iteration + 1} iterations")
                return q, True
            
            # Task space error
            error = np.concatenate([pos_error, axis_angle])
            
            # Jacobian pseudo-inverse
            J_pinv = self.jacobian_pinv(q, damping)
            
            # Primary solution
            dq = J_pinv @ error
            
            # Null space optimization
            if use_null_space:
                J = self.jacobian(q)
                # Null space projector
                N = np.eye(self.n_joints) - J_pinv @ J
                
                # Secondary objective: move toward joint centers
                q_center = np.array([lim.center() for lim in self.joint_limits])
                q_secondary = 0.1 * (q_center - q)
                
                dq += N @ q_secondary
            
            # Update joints
            q = q + dq
            
            # Apply joint limits
            for i in range(self.n_joints):
                q[i] = self.joint_limits[i].clamp(q[i])
        
        # Did not converge
        self._last_solution = q.copy()
        logger.warning(f"IK did not converge after {max_iterations} iterations")
        return q, False
    
    def check_joint_limits(self, q: FloatArray) -> Tuple[bool, List[int]]:
        """
        Check if joint angles are within limits.
        
        Args:
            q: Joint angles
            
        Returns:
            Tuple of (all_within_limits, list_of_violated_joints)
        """
        q = np.asarray(q).flatten()
        violations = []
        
        for i, (qi, limit) in enumerate(zip(q, self.joint_limits)):
            if not limit.is_within(qi):
                violations.append(i)
        
        return len(violations) == 0, violations
    
    def clamp_joints(self, q: FloatArray) -> FloatArray:
        """
        Clamp joint angles to limits.
        
        Args:
            q: Joint angles
            
        Returns:
            Clamped joint angles
        """
        q = np.asarray(q).flatten().copy()
        
        for i in range(self.n_joints):
            q[i] = self.joint_limits[i].clamp(q[i])
        
        return q
    
    def get_manipulability(self, q: FloatArray) -> float:
        """
        Compute Yoshikawa manipulability measure.
        
        w = sqrt(det(J J^T))
        
        Higher values indicate configurations farther from singularities.
        
        Args:
            q: Joint angles
            
        Returns:
            Manipulability measure (>= 0)
        """
        J = self.jacobian(q)
        JJT = J @ J.T
        
        det = np.linalg.det(JJT)
        
        return float(np.sqrt(max(0, det)))
    
    def get_workspace_limits(self) -> Tuple[FloatArray, FloatArray]:
        """
        Estimate workspace bounding box.
        
        Returns:
            Tuple of (min_xyz, max_xyz)
        """
        # Sample random configurations
        samples = 1000
        positions = []
        
        for _ in range(samples):
            q = np.array([
                np.random.uniform(lim.lower, lim.upper)
                for lim in self.joint_limits
            ])
            pose = self.forward_kinematics(q)
            positions.append(pose.position)
        
        positions = np.array(positions)
        
        return positions.min(axis=0), positions.max(axis=0)
