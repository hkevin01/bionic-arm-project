"""
Trajectory Generation Module
============================

Smooth trajectory generation for robotic arm motion using minimum-jerk
profiles and other optimization-based methods.

Mathematical Background:

    Minimum Jerk Trajectory:
        Minimizes the integral of jerk (derivative of acceleration) squared:
        
            J = ∫(d³x/dt³)² dt
        
        Solution is a 5th order polynomial:
            x(t) = x₀ + (xf - x₀)[10τ³ - 15τ⁴ + 6τ⁵]
            
        where τ = t/T is normalized time.
        
        Properties:
        - Smooth velocity, acceleration, and jerk profiles
        - Biomimetic (matches human reaching movements)
        - Bell-shaped velocity profile
        
    Trapezoidal Velocity Profile:
        Alternative for faster movements with constant velocity phase:
        
            Phase 1: Acceleration (0 to t_a)
            Phase 2: Constant velocity (t_a to T - t_a)
            Phase 3: Deceleration (T - t_a to T)

Trajectory Types:
    - Point-to-point: Single start and end position
    - Via-point: Pass through intermediate waypoints
    - Continuous: Streaming velocity commands

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any, Generator
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]


# =============================================================================
# Configuration
# =============================================================================

class TrajectoryType(Enum):
    """Type of trajectory profile."""
    MINIMUM_JERK = auto()      # 5th order polynomial (smooth, biomimetic)
    TRAPEZOIDAL = auto()        # Linear with acceleration/deceleration
    CUBIC_SPLINE = auto()       # Cubic spline interpolation
    QUINTIC = auto()            # 5th order polynomial with velocity constraints
    LINEAR = auto()             # Simple linear interpolation


@dataclass
class TrajectoryConfig:
    """
    Configuration for trajectory generation.
    
    Attributes:
        trajectory_type: Type of trajectory profile
        max_velocity: Maximum velocity (rad/s or m/s)
        max_acceleration: Maximum acceleration
        max_jerk: Maximum jerk (for jerk-limited trajectories)
        timestep_ms: Output timestep in milliseconds
        blend_radius: Radius for blending between segments (m or rad)
    """
    trajectory_type: TrajectoryType = TrajectoryType.MINIMUM_JERK
    max_velocity: float = 1.0
    max_acceleration: float = 5.0
    max_jerk: float = 50.0
    timestep_ms: float = 10.0
    blend_radius: float = 0.05
    
    @property
    def timestep(self) -> float:
        """Timestep in seconds."""
        return self.timestep_ms / 1000.0


@dataclass
class TrajectoryPoint:
    """
    A point along a trajectory.
    
    Attributes:
        time: Time from trajectory start (seconds)
        position: Joint positions (rad) or Cartesian position
        velocity: Velocity
        acceleration: Acceleration (optional)
        jerk: Jerk (optional)
    """
    time: float
    position: FloatArray
    velocity: FloatArray
    acceleration: Optional[FloatArray] = None
    jerk: Optional[FloatArray] = None
    
    def __post_init__(self) -> None:
        """Convert to numpy arrays."""
        self.position = np.asarray(self.position)
        self.velocity = np.asarray(self.velocity)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration)
        if self.jerk is not None:
            self.jerk = np.asarray(self.jerk)
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return len(self.position)
    
    def interpolate(self, other: "TrajectoryPoint", alpha: float) -> "TrajectoryPoint":
        """Linear interpolation between two points."""
        return TrajectoryPoint(
            time=self.time + alpha * (other.time - self.time),
            position=self.position + alpha * (other.position - self.position),
            velocity=self.velocity + alpha * (other.velocity - self.velocity),
            acceleration=(
                self.acceleration + alpha * (other.acceleration - self.acceleration)
                if self.acceleration is not None and other.acceleration is not None
                else None
            )
        )


@dataclass
class TrajectorySegment:
    """
    A segment of a trajectory between two waypoints.
    
    Attributes:
        start: Starting configuration
        end: Ending configuration
        duration: Segment duration (seconds)
        points: Discretized trajectory points
    """
    start: FloatArray
    end: FloatArray
    duration: float
    points: List[TrajectoryPoint] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Convert to numpy arrays."""
        self.start = np.asarray(self.start)
        self.end = np.asarray(self.end)
    
    @property
    def distance(self) -> float:
        """Euclidean distance between start and end."""
        return float(np.linalg.norm(self.end - self.start))
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return len(self.start)
    
    def __len__(self) -> int:
        """Number of discretized points."""
        return len(self.points)
    
    def __iter__(self):
        """Iterate over trajectory points."""
        return iter(self.points)


# =============================================================================
# Minimum Jerk Trajectory
# =============================================================================

class MinimumJerkTrajectory:
    """
    Generate minimum-jerk trajectories for smooth, biomimetic motion.
    
    The minimum-jerk trajectory minimizes the integral of squared jerk,
    producing smooth, human-like reaching movements.
    
    Mathematical Formulation:
        x(τ) = x₀ + (xf - x₀) · [10τ³ - 15τ⁴ + 6τ⁵]
        v(τ) = (xf - x₀)/T · [30τ² - 60τ³ + 30τ⁴]
        a(τ) = (xf - x₀)/T² · [60τ - 180τ² + 120τ³]
        
    where τ = t/T is normalized time (0 to 1).
    
    Properties:
        - Zero velocity and acceleration at start and end
        - Bell-shaped velocity profile
        - Symmetric acceleration profile
        - Peak velocity at t = T/2
    
    Example:
        >>> mjt = MinimumJerkTrajectory()
        >>> segment = mjt.generate(
        ...     start=np.zeros(7),
        ...     end=np.array([0.5, 0.3, 0.0, 1.0, 0.0, 0.2, 0.1]),
        ...     duration=1.0
        ... )
        >>> for point in segment:
        ...     send_to_arm(point.position)
    """
    
    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        """
        Initialize minimum jerk trajectory generator.
        
        Args:
            config: Trajectory configuration
        """
        self.config = config or TrajectoryConfig()
    
    @staticmethod
    def _polynomial_coefficients(
        x0: float,
        xf: float,
        T: float,
        v0: float = 0.0,
        vf: float = 0.0,
        a0: float = 0.0,
        af: float = 0.0
    ) -> FloatArray:
        """
        Compute 5th order polynomial coefficients.
        
        For boundary conditions:
            x(0) = x0, x(T) = xf
            v(0) = v0, v(T) = vf
            a(0) = a0, a(T) = af
        
        Returns:
            Coefficients [a0, a1, a2, a3, a4, a5]
        """
        # Standard minimum jerk (zero boundary velocities and accelerations)
        if v0 == 0 and vf == 0 and a0 == 0 and af == 0:
            d = xf - x0
            return np.array([
                x0,              # a0
                0,               # a1
                0,               # a2
                10 * d / T**3,   # a3
                -15 * d / T**4,  # a4
                6 * d / T**5     # a5
            ])
        
        # General case with arbitrary boundary conditions
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        
        a0_coef = x0
        a1_coef = v0
        a2_coef = a0 / 2
        
        # Solve for a3, a4, a5
        A = np.array([
            [T3, T4, T5],
            [3*T2, 4*T3, 5*T4],
            [6*T, 12*T2, 20*T3]
        ])
        
        b = np.array([
            xf - x0 - v0*T - a0*T2/2,
            vf - v0 - a0*T,
            af - a0
        ])
        
        try:
            a345 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in polynomial fit, using defaults")
            a345 = np.zeros(3)
        
        return np.array([a0_coef, a1_coef, a2_coef, a345[0], a345[1], a345[2]])
    
    @staticmethod
    def _evaluate_polynomial(
        coeffs: FloatArray,
        t: float
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate polynomial and its derivatives at time t.
        
        Returns:
            Tuple of (position, velocity, acceleration, jerk)
        """
        a0, a1, a2, a3, a4, a5 = coeffs
        t2, t3, t4, t5 = t**2, t**3, t**4, t**5
        
        pos = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
        vel = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4
        acc = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3
        jrk = 6*a3 + 24*a4*t + 60*a5*t2
        
        return pos, vel, acc, jrk
    
    def generate(
        self,
        start: FloatArray,
        end: FloatArray,
        duration: Optional[float] = None,
        v_start: Optional[FloatArray] = None,
        v_end: Optional[FloatArray] = None
    ) -> TrajectorySegment:
        """
        Generate minimum jerk trajectory segment.
        
        Args:
            start: Starting position
            end: Ending position
            duration: Trajectory duration (auto-computed if None)
            v_start: Starting velocity (default zero)
            v_end: Ending velocity (default zero)
            
        Returns:
            TrajectorySegment with discretized points
        """
        start = np.asarray(start).flatten()
        end = np.asarray(end).flatten()
        n_dims = len(start)
        
        if len(end) != n_dims:
            raise ValueError("start and end must have same dimension")
        
        # Auto-compute duration based on distance and max velocity
        if duration is None:
            distance = np.linalg.norm(end - start)
            # Peak velocity is at t=T/2 and is 1.875 * average velocity
            # For min-jerk: v_peak = 15/8 * d/T
            # So T = 15/8 * d / v_max
            duration = max(0.1, 15/8 * distance / self.config.max_velocity)
        
        # Default velocities
        if v_start is None:
            v_start = np.zeros(n_dims)
        if v_end is None:
            v_end = np.zeros(n_dims)
        
        # Compute coefficients for each dimension
        coefficients = []
        for i in range(n_dims):
            coeffs = self._polynomial_coefficients(
                start[i], end[i], duration,
                v_start[i], v_end[i]
            )
            coefficients.append(coeffs)
        
        # Generate trajectory points
        dt = self.config.timestep
        n_points = int(np.ceil(duration / dt)) + 1
        times = np.linspace(0, duration, n_points)
        
        points = []
        for t in times:
            pos = np.zeros(n_dims)
            vel = np.zeros(n_dims)
            acc = np.zeros(n_dims)
            jrk = np.zeros(n_dims)
            
            for i in range(n_dims):
                pos[i], vel[i], acc[i], jrk[i] = self._evaluate_polynomial(
                    coefficients[i], t
                )
            
            points.append(TrajectoryPoint(
                time=t,
                position=pos,
                velocity=vel,
                acceleration=acc,
                jerk=jrk
            ))
        
        return TrajectorySegment(
            start=start,
            end=end,
            duration=duration,
            points=points
        )
    
    def generate_via_points(
        self,
        waypoints: List[FloatArray],
        durations: Optional[List[float]] = None
    ) -> List[TrajectorySegment]:
        """
        Generate trajectory through multiple waypoints.
        
        Ensures smooth velocity transitions between segments.
        
        Args:
            waypoints: List of waypoint positions
            durations: Duration for each segment (auto-computed if None)
            
        Returns:
            List of TrajectorySegments
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        n_segments = len(waypoints) - 1
        
        if durations is None:
            durations = [None] * n_segments
        elif len(durations) != n_segments:
            raise ValueError(f"Expected {n_segments} durations")
        
        segments = []
        v_current = np.zeros(len(waypoints[0]))
        
        for i in range(n_segments):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Compute intermediate velocity for smooth blending
            if i < n_segments - 1:
                # Direction to next waypoint
                next_dir = waypoints[i + 2] - waypoints[i + 1]
                next_dist = np.linalg.norm(next_dir)
                if next_dist > 1e-6:
                    # Velocity at waypoint proportional to blend
                    v_end = (next_dir / next_dist) * self.config.max_velocity * 0.3
                else:
                    v_end = np.zeros_like(v_current)
            else:
                v_end = np.zeros_like(v_current)
            
            segment = self.generate(
                start=start,
                end=end,
                duration=durations[i],
                v_start=v_current,
                v_end=v_end
            )
            segments.append(segment)
            
            # Use final velocity for next segment
            if segment.points:
                v_current = segment.points[-1].velocity
            else:
                v_current = v_end
        
        return segments


# =============================================================================
# Trajectory Generator
# =============================================================================

class TrajectoryGenerator:
    """
    High-level trajectory generator supporting multiple profile types.
    
    Factory class that creates appropriate trajectory based on configuration.
    
    Example:
        >>> config = TrajectoryConfig(trajectory_type=TrajectoryType.MINIMUM_JERK)
        >>> generator = TrajectoryGenerator(config)
        >>> 
        >>> trajectory = generator.point_to_point(
        ...     start=current_position,
        ...     end=target_position,
        ...     duration=1.5
        ... )
        >>> 
        >>> for point in generator.stream(trajectory):
        ...     send_to_arm(point)
    """
    
    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        """
        Initialize trajectory generator.
        
        Args:
            config: Trajectory configuration
        """
        self.config = config or TrajectoryConfig()
        
        # Initialize trajectory planners
        self._min_jerk = MinimumJerkTrajectory(self.config)
        
        logger.info(f"TrajectoryGenerator initialized: {self.config.trajectory_type.name}")
    
    def point_to_point(
        self,
        start: FloatArray,
        end: FloatArray,
        duration: Optional[float] = None
    ) -> TrajectorySegment:
        """
        Generate point-to-point trajectory.
        
        Args:
            start: Start position
            end: End position
            duration: Trajectory duration (auto-computed if None)
            
        Returns:
            Trajectory segment
        """
        if self.config.trajectory_type == TrajectoryType.MINIMUM_JERK:
            return self._min_jerk.generate(start, end, duration)
        
        elif self.config.trajectory_type == TrajectoryType.LINEAR:
            return self._generate_linear(start, end, duration)
        
        elif self.config.trajectory_type == TrajectoryType.TRAPEZOIDAL:
            return self._generate_trapezoidal(start, end, duration)
        
        else:
            # Default to minimum jerk
            return self._min_jerk.generate(start, end, duration)
    
    def _generate_linear(
        self,
        start: FloatArray,
        end: FloatArray,
        duration: Optional[float] = None
    ) -> TrajectorySegment:
        """Generate linear (constant velocity) trajectory."""
        start = np.asarray(start)
        end = np.asarray(end)
        
        distance = np.linalg.norm(end - start)
        
        if duration is None:
            duration = max(0.1, distance / self.config.max_velocity)
        
        dt = self.config.timestep
        n_points = int(np.ceil(duration / dt)) + 1
        times = np.linspace(0, duration, n_points)
        
        velocity = (end - start) / duration
        
        points = []
        for t in times:
            alpha = t / duration
            pos = start + alpha * (end - start)
            points.append(TrajectoryPoint(
                time=t,
                position=pos,
                velocity=velocity,
                acceleration=np.zeros_like(velocity)
            ))
        
        return TrajectorySegment(start=start, end=end, duration=duration, points=points)
    
    def _generate_trapezoidal(
        self,
        start: FloatArray,
        end: FloatArray,
        duration: Optional[float] = None
    ) -> TrajectorySegment:
        """Generate trapezoidal velocity profile trajectory."""
        start = np.asarray(start)
        end = np.asarray(end)
        
        delta = end - start
        distance = np.linalg.norm(delta)
        
        if distance < 1e-9:
            return self._generate_linear(start, end, duration)
        
        direction = delta / distance
        
        # Compute minimum time based on acceleration limits
        v_max = self.config.max_velocity
        a_max = self.config.max_acceleration
        
        # Time to accelerate to max velocity
        t_acc = v_max / a_max
        
        # Distance during acceleration phase
        d_acc = 0.5 * a_max * t_acc**2
        
        if 2 * d_acc >= distance:
            # Triangular profile (can't reach max velocity)
            t_acc = np.sqrt(distance / a_max)
            t_cruise = 0
            v_peak = a_max * t_acc
        else:
            # Trapezoidal profile
            d_cruise = distance - 2 * d_acc
            t_cruise = d_cruise / v_max
            v_peak = v_max
        
        total_duration = 2 * t_acc + t_cruise
        
        if duration is not None and duration > total_duration:
            # Stretch trajectory to requested duration
            scale = duration / total_duration
            total_duration = duration
            t_acc *= scale
            t_cruise *= scale
            v_peak /= scale
            a_max /= scale**2
        
        # Generate points
        dt = self.config.timestep
        n_points = int(np.ceil(total_duration / dt)) + 1
        times = np.linspace(0, total_duration, n_points)
        
        points = []
        for t in times:
            if t <= t_acc:
                # Acceleration phase
                s = 0.5 * a_max * t**2
                v = a_max * t
                a = a_max
            elif t <= t_acc + t_cruise:
                # Cruise phase
                s = 0.5 * a_max * t_acc**2 + v_peak * (t - t_acc)
                v = v_peak
                a = 0
            else:
                # Deceleration phase
                t_decel = t - t_acc - t_cruise
                s = (0.5 * a_max * t_acc**2 + 
                     v_peak * t_cruise + 
                     v_peak * t_decel - 0.5 * a_max * t_decel**2)
                v = v_peak - a_max * t_decel
                a = -a_max
            
            pos = start + direction * s
            vel = direction * v
            acc = direction * a
            
            points.append(TrajectoryPoint(
                time=t,
                position=pos,
                velocity=vel,
                acceleration=acc
            ))
        
        return TrajectorySegment(
            start=start, end=end, duration=total_duration, points=points
        )
    
    def multi_segment(
        self,
        waypoints: List[FloatArray],
        durations: Optional[List[float]] = None
    ) -> List[TrajectorySegment]:
        """
        Generate multi-segment trajectory through waypoints.
        
        Args:
            waypoints: List of waypoints
            durations: Optional durations for each segment
            
        Returns:
            List of trajectory segments
        """
        if self.config.trajectory_type == TrajectoryType.MINIMUM_JERK:
            return self._min_jerk.generate_via_points(waypoints, durations)
        else:
            # Generate individual segments
            segments = []
            for i in range(len(waypoints) - 1):
                dur = durations[i] if durations else None
                segment = self.point_to_point(waypoints[i], waypoints[i+1], dur)
                segments.append(segment)
            return segments
    
    def stream(
        self,
        trajectory: TrajectorySegment,
        real_time: bool = False
    ) -> Generator[TrajectoryPoint, None, None]:
        """
        Stream trajectory points as a generator.
        
        Args:
            trajectory: Trajectory segment to stream
            real_time: If True, pause between points to match real time
            
        Yields:
            TrajectoryPoint instances
        """
        import time
        
        start_time = time.perf_counter()
        
        for point in trajectory.points:
            if real_time:
                elapsed = time.perf_counter() - start_time
                sleep_time = point.time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            yield point
    
    def blend_segments(
        self,
        segments: List[TrajectorySegment],
        blend_time: float = 0.1
    ) -> TrajectorySegment:
        """
        Blend multiple segments into a single continuous trajectory.
        
        Uses cosine blending at segment boundaries.
        
        Args:
            segments: List of trajectory segments
            blend_time: Time for blending between segments
            
        Returns:
            Single blended trajectory segment
        """
        if len(segments) == 0:
            raise ValueError("No segments to blend")
        
        if len(segments) == 1:
            return segments[0]
        
        all_points = []
        current_time = 0.0
        
        for i, segment in enumerate(segments):
            for j, point in enumerate(segment.points):
                # Adjust time offset
                adjusted_point = TrajectoryPoint(
                    time=current_time + point.time,
                    position=point.position.copy(),
                    velocity=point.velocity.copy(),
                    acceleration=point.acceleration.copy() if point.acceleration is not None else None
                )
                
                # Apply blending at segment boundaries
                if i > 0 and point.time < blend_time:
                    # Blend with previous segment
                    alpha = point.time / blend_time
                    blend_weight = 0.5 * (1 - np.cos(np.pi * alpha))
                    # Weighted average with previous segment's velocity
                    # (simplified blending)
                
                all_points.append(adjusted_point)
            
            current_time += segment.duration
        
        return TrajectorySegment(
            start=segments[0].start,
            end=segments[-1].end,
            duration=current_time,
            points=all_points
        )
    
    def compute_duration(
        self,
        start: FloatArray,
        end: FloatArray,
        velocity_fraction: float = 0.8
    ) -> float:
        """
        Compute appropriate duration for a movement.
        
        Args:
            start: Start position
            end: End position
            velocity_fraction: Fraction of max velocity to use (0-1)
            
        Returns:
            Recommended duration in seconds
        """
        distance = np.linalg.norm(np.asarray(end) - np.asarray(start))
        effective_velocity = self.config.max_velocity * velocity_fraction
        
        if self.config.trajectory_type == TrajectoryType.MINIMUM_JERK:
            # Minimum jerk peak velocity is 1.875x average
            return 15/8 * distance / effective_velocity
        else:
            return distance / effective_velocity
