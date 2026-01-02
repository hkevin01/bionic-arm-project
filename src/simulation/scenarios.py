"""
Simulation Scenarios Module
===========================

Predefined test scenarios for evaluating bionic arm performance
including reaching, grasping, and manipulation tasks.

Scenarios:
    - ReachScenario: Point-to-point reaching
    - GraspScenario: Object grasping tasks
    - ManipulationScenario: Pick and place operations
    - TrackingScenario: Moving target tracking

Author: Bionic Arm Project Team
License: MIT
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
from numpy.typing import NDArray

from .arm_simulator import ArmSimulator, ArmState
from .environment import (
    SimulationEnvironment, 
    ObjectFactory, 
    ObjectConfig,
    WorkspaceConfig
)

logger = logging.getLogger(__name__)


class ScenarioStatus(Enum):
    """Scenario execution status."""
    NOT_STARTED = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    ABORTED = auto()


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    EXPERT = auto()


@dataclass
class ScenarioConfig:
    """
    Base configuration for scenarios.
    
    Attributes:
        name: Scenario identifier
        description: Human-readable description
        timeout_seconds: Maximum execution time
        difficulty: Task difficulty level
        success_threshold: Distance for success (meters)
        n_trials: Number of trial repetitions
        randomize: Randomize initial conditions
    """
    name: str = "scenario"
    description: str = ""
    timeout_seconds: float = 30.0
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    success_threshold: float = 0.02  # 2cm
    n_trials: int = 1
    randomize: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")


@dataclass
class TrialMetrics:
    """Metrics for a single trial."""
    trial_id: int = 0
    success: bool = False
    duration_seconds: float = 0.0
    final_error: float = float('inf')
    path_length: float = 0.0
    peak_velocity: float = 0.0
    smoothness: float = 0.0  # Jerk metric
    n_corrections: int = 0
    contact_count: int = 0
    grasp_force_peak: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "success": self.success,
            "duration_s": self.duration_seconds,
            "final_error_m": self.final_error,
            "path_length_m": self.path_length,
            "peak_velocity_ms": self.peak_velocity,
            "smoothness": self.smoothness,
            "n_corrections": self.n_corrections,
            "contact_count": self.contact_count,
            "grasp_force_peak_N": self.grasp_force_peak
        }


@dataclass
class ScenarioResult:
    """
    Complete results from scenario execution.
    
    Contains aggregate statistics across all trials.
    """
    scenario_name: str = ""
    status: ScenarioStatus = ScenarioStatus.NOT_STARTED
    trials: List[TrialMetrics] = field(default_factory=list)
    
    # Aggregate metrics
    success_rate: float = 0.0
    mean_duration: float = 0.0
    std_duration: float = 0.0
    mean_error: float = 0.0
    mean_path_length: float = 0.0
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from trials."""
        if not self.trials:
            return
        
        successes = [t.success for t in self.trials]
        durations = [t.duration_seconds for t in self.trials]
        errors = [t.final_error for t in self.trials]
        path_lengths = [t.path_length for t in self.trials]
        
        self.success_rate = sum(successes) / len(successes)
        self.mean_duration = np.mean(durations)
        self.std_duration = np.std(durations)
        self.mean_error = np.mean(errors)
        self.mean_path_length = np.mean(path_lengths)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "status": self.status.name,
            "success_rate": self.success_rate,
            "mean_duration_s": self.mean_duration,
            "std_duration_s": self.std_duration,
            "mean_error_m": self.mean_error,
            "mean_path_length_m": self.mean_path_length,
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials]
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Scenario: {self.scenario_name}\n"
            f"  Status: {self.status.name}\n"
            f"  Success Rate: {self.success_rate:.1%}\n"
            f"  Mean Duration: {self.mean_duration:.2f}s ± {self.std_duration:.2f}s\n"
            f"  Mean Error: {self.mean_error*1000:.1f}mm\n"
            f"  Trials: {len(self.trials)}"
        )


class Scenario(ABC):
    """
    Abstract base class for scenarios.
    
    Subclasses implement specific task logic.
    """
    
    def __init__(
        self,
        config: ScenarioConfig,
        simulator: Optional[ArmSimulator] = None,
        environment: Optional[SimulationEnvironment] = None
    ) -> None:
        self.config = config
        self.simulator = simulator
        self.environment = environment or SimulationEnvironment()
        
        # State
        self._status = ScenarioStatus.NOT_STARTED
        self._current_trial = 0
        self._trial_start_time = 0.0
        
        # Path tracking for metrics
        self._path_points: List[NDArray] = []
        self._velocity_samples: List[float] = []
        
        # Random state
        if config.seed is not None:
            np.random.seed(config.seed)
    
    @abstractmethod
    def setup(self) -> None:
        """Setup scenario (called before each trial)."""
        pass
    
    @abstractmethod
    def get_target(self) -> NDArray:
        """Get current target position."""
        pass
    
    @abstractmethod
    def check_success(self, arm_state: ArmState) -> bool:
        """Check if trial is successful."""
        pass
    
    def check_failure(self, arm_state: ArmState) -> bool:
        """Check for failure conditions (optional override)."""
        return False
    
    def on_step(self, arm_state: ArmState) -> None:
        """Called each simulation step (optional override)."""
        # Track path
        self._path_points.append(arm_state.end_effector_pos.copy())
        
        # Track velocity
        vel = np.linalg.norm(arm_state.joint_velocities)
        self._velocity_samples.append(vel)
    
    def run(
        self,
        controller_callback: Callable[[ArmState, NDArray], NDArray]
    ) -> ScenarioResult:
        """
        Run all trials of the scenario.
        
        Args:
            controller_callback: Function (arm_state, target) -> joint_velocities
            
        Returns:
            Complete scenario results
        """
        result = ScenarioResult(
            scenario_name=self.config.name,
            start_time=time.time()
        )
        
        self._status = ScenarioStatus.RUNNING
        
        for trial_id in range(self.config.n_trials):
            self._current_trial = trial_id
            trial_metrics = self._run_trial(trial_id, controller_callback)
            result.trials.append(trial_metrics)
            
            logger.info(
                f"Trial {trial_id + 1}/{self.config.n_trials}: "
                f"{'SUCCESS' if trial_metrics.success else 'FAIL'} "
                f"({trial_metrics.duration_seconds:.2f}s)"
            )
        
        result.end_time = time.time()
        result.compute_aggregates()
        
        # Determine overall status
        if result.success_rate >= 0.5:
            self._status = ScenarioStatus.SUCCESS
        else:
            self._status = ScenarioStatus.FAILURE
        result.status = self._status
        
        return result
    
    def _run_trial(
        self,
        trial_id: int,
        controller_callback: Callable[[ArmState, NDArray], NDArray]
    ) -> TrialMetrics:
        """Run a single trial."""
        metrics = TrialMetrics(trial_id=trial_id)
        
        # Reset state
        self._path_points = []
        self._velocity_samples = []
        
        # Setup
        self.setup()
        if self.simulator:
            self.simulator.reset()
        
        target = self.get_target()
        self._trial_start_time = time.time()
        
        # Run trial
        while True:
            elapsed = time.time() - self._trial_start_time
            
            # Check timeout
            if elapsed > self.config.timeout_seconds:
                metrics.success = False
                metrics.duration_seconds = elapsed
                break
            
            # Get arm state
            if self.simulator:
                arm_state = self.simulator.get_state()
            else:
                arm_state = ArmState()
            
            # Check success/failure
            if self.check_success(arm_state):
                metrics.success = True
                metrics.duration_seconds = elapsed
                break
            
            if self.check_failure(arm_state):
                metrics.success = False
                metrics.duration_seconds = elapsed
                break
            
            # Get control command
            velocity_cmd = controller_callback(arm_state, target)
            
            # Apply command
            if self.simulator:
                self.simulator.set_velocity_target(velocity_cmd)
                self.simulator.step(10)  # 10 physics steps
            
            # Track metrics
            self.on_step(arm_state)
        
        # Compute final metrics
        if self.simulator:
            final_state = self.simulator.get_state()
        else:
            final_state = ArmState()
        
        metrics.final_error = float(np.linalg.norm(
            final_state.end_effector_pos - target
        ))
        metrics.path_length = self._compute_path_length()
        metrics.peak_velocity = max(self._velocity_samples) if self._velocity_samples else 0
        metrics.smoothness = self._compute_smoothness()
        
        return metrics
    
    def _compute_path_length(self) -> float:
        """Compute total path length from tracked points."""
        if len(self._path_points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self._path_points)):
            total += np.linalg.norm(
                self._path_points[i] - self._path_points[i-1]
            )
        return total
    
    def _compute_smoothness(self) -> float:
        """Compute smoothness metric (negative jerk)."""
        if len(self._velocity_samples) < 3:
            return 0.0
        
        # Approximate jerk as changes in velocity
        velocities = np.array(self._velocity_samples)
        accelerations = np.diff(velocities)
        jerks = np.diff(accelerations)
        
        # Lower jerk = smoother
        rms_jerk = np.sqrt(np.mean(jerks ** 2))
        return -rms_jerk  # Negative so higher is better
    
    @property
    def status(self) -> ScenarioStatus:
        return self._status


class ReachScenario(Scenario):
    """
    Point-to-point reaching scenario.
    
    Task: Move end-effector from start to target position.
    """
    
    def __init__(
        self,
        target_position: Optional[NDArray] = None,
        **kwargs
    ) -> None:
        config = kwargs.pop('config', ScenarioConfig(
            name="reach",
            description="Point-to-point reaching task"
        ))
        super().__init__(config, **kwargs)
        
        self._target = np.array(target_position) if target_position is not None else np.array([0.5, 0, 0.3])
        self._initial_target = self._target.copy()
    
    def setup(self) -> None:
        """Setup reaching scenario."""
        if self.config.randomize:
            # Random target within workspace
            workspace = self.environment.workspace
            self._target = workspace.random_point()
        else:
            self._target = self._initial_target.copy()
        
        # Add visual target
        self.environment.clear()
        target_obj = ObjectFactory.create_target(
            position=tuple(self._target)
        )
        self.environment.add_object(target_obj)
    
    def get_target(self) -> NDArray:
        return self._target
    
    def check_success(self, arm_state: ArmState) -> bool:
        error = np.linalg.norm(arm_state.end_effector_pos - self._target)
        return error < self.config.success_threshold


class GraspScenario(Scenario):
    """
    Object grasping scenario.
    
    Task: Reach object and apply stable grasp.
    """
    
    def __init__(
        self,
        object_config: Optional[ObjectConfig] = None,
        min_grasp_force: float = 1.0,
        max_grasp_force: float = 20.0,
        hold_time: float = 1.0,
        **kwargs
    ) -> None:
        config = kwargs.pop('config', ScenarioConfig(
            name="grasp",
            description="Object grasping task",
            timeout_seconds=45.0
        ))
        super().__init__(config, **kwargs)
        
        self._object_config = object_config or ObjectFactory.create_cube()
        self._min_grasp_force = min_grasp_force
        self._max_grasp_force = max_grasp_force
        self._hold_time = hold_time
        
        # State
        self._object = None
        self._grasp_start_time: Optional[float] = None
        self._phase = "reach"  # "reach", "grasp", "hold"
    
    def setup(self) -> None:
        """Setup grasping scenario."""
        self.environment.clear()
        
        if self.config.randomize:
            # Random object position
            workspace = self.environment.workspace
            pos = workspace.random_point()
            pos[2] = 0.05  # On table
            self._object_config.position = tuple(pos)
        
        self._object = self.environment.add_object(self._object_config)
        self._phase = "reach"
        self._grasp_start_time = None
    
    def get_target(self) -> NDArray:
        if self._object:
            return self._object.position.copy()
        return np.array([0.5, 0, 0.1])
    
    def check_success(self, arm_state: ArmState) -> bool:
        if self._object is None:
            return False
        
        # Phase: Reach
        if self._phase == "reach":
            error = np.linalg.norm(
                arm_state.end_effector_pos - self._object.position
            )
            if error < self.config.success_threshold:
                self._phase = "grasp"
                logger.debug("Phase: grasp")
        
        # Phase: Grasp
        elif self._phase == "grasp":
            if self._object.contact_points >= 2:
                force = self._object.grasp_force
                if self._min_grasp_force <= force <= self._max_grasp_force:
                    self._phase = "hold"
                    self._grasp_start_time = time.time()
                    logger.debug("Phase: hold")
        
        # Phase: Hold
        elif self._phase == "hold":
            if self._grasp_start_time is None:
                return False
            
            hold_duration = time.time() - self._grasp_start_time
            if hold_duration >= self._hold_time:
                return True
            
            # Check grasp maintained
            if self._object.contact_points < 2:
                self._phase = "grasp"
                self._grasp_start_time = None
        
        return False
    
    def check_failure(self, arm_state: ArmState) -> bool:
        if self._object is None:
            return True
        
        # Object dropped or crushed
        force = self._object.grasp_force
        if force > self._max_grasp_force * 1.5:
            logger.warning("Object crushed - excessive force")
            return True
        
        return False


class TrackingScenario(Scenario):
    """
    Moving target tracking scenario.
    
    Task: Follow a moving target point.
    """
    
    def __init__(
        self,
        trajectory_type: str = "circle",
        trajectory_speed: float = 0.1,  # m/s
        trajectory_radius: float = 0.15,
        tracking_duration: float = 10.0,
        **kwargs
    ) -> None:
        config = kwargs.pop('config', ScenarioConfig(
            name="tracking",
            description="Moving target tracking task",
            timeout_seconds=tracking_duration + 5.0,
            success_threshold=0.05  # 5cm for tracking
        ))
        super().__init__(config, **kwargs)
        
        self._trajectory_type = trajectory_type
        self._trajectory_speed = trajectory_speed
        self._trajectory_radius = trajectory_radius
        self._tracking_duration = tracking_duration
        
        # Center of trajectory
        self._center = np.array([0.5, 0, 0.3])
        
        # Tracking metrics
        self._tracking_errors: List[float] = []
    
    def setup(self) -> None:
        """Setup tracking scenario."""
        self._tracking_errors = []
        
        if self.config.randomize:
            # Random center
            workspace = self.environment.workspace
            self._center = workspace.random_point()
    
    def get_target(self) -> NDArray:
        """Get current target on trajectory."""
        elapsed = time.time() - self._trial_start_time
        
        if self._trajectory_type == "circle":
            angle = (elapsed * self._trajectory_speed / self._trajectory_radius)
            offset = np.array([
                self._trajectory_radius * np.cos(angle),
                self._trajectory_radius * np.sin(angle),
                0
            ])
        elif self._trajectory_type == "figure8":
            t = elapsed * self._trajectory_speed
            offset = np.array([
                self._trajectory_radius * np.sin(t),
                self._trajectory_radius * np.sin(2 * t) / 2,
                0
            ])
        elif self._trajectory_type == "line":
            # Back and forth
            t = elapsed * self._trajectory_speed
            phase = (t / self._trajectory_radius) % 4
            if phase < 1:
                x = phase * self._trajectory_radius
            elif phase < 2:
                x = self._trajectory_radius
            elif phase < 3:
                x = (3 - phase) * self._trajectory_radius
            else:
                x = 0
            offset = np.array([x - self._trajectory_radius/2, 0, 0])
        else:
            offset = np.zeros(3)
        
        return self._center + offset
    
    def check_success(self, arm_state: ArmState) -> bool:
        elapsed = time.time() - self._trial_start_time
        
        # Track error
        target = self.get_target()
        error = np.linalg.norm(arm_state.end_effector_pos - target)
        self._tracking_errors.append(error)
        
        # Success if completed duration with acceptable tracking
        if elapsed >= self._tracking_duration:
            mean_error = np.mean(self._tracking_errors)
            return mean_error < self.config.success_threshold
        
        return False
    
    def on_step(self, arm_state: ArmState) -> None:
        super().on_step(arm_state)
        
        # Track error
        target = self.get_target()
        error = np.linalg.norm(arm_state.end_effector_pos - target)
        self._tracking_errors.append(error)


class ScenarioRunner:
    """
    Manager for running multiple scenarios.
    
    Coordinates scenario execution and result collection.
    
    Example:
        >>> runner = ScenarioRunner(simulator, environment)
        >>> 
        >>> # Add scenarios
        >>> runner.add_scenario(ReachScenario(target=[0.5, 0, 0.3]))
        >>> runner.add_scenario(GraspScenario())
        >>> 
        >>> # Run all
        >>> results = runner.run_all(controller_callback)
        >>> 
        >>> # Print summary
        >>> runner.print_summary()
    """
    
    def __init__(
        self,
        simulator: Optional[ArmSimulator] = None,
        environment: Optional[SimulationEnvironment] = None
    ) -> None:
        self.simulator = simulator
        self.environment = environment or SimulationEnvironment()
        
        self._scenarios: List[Scenario] = []
        self._results: List[ScenarioResult] = []
    
    def add_scenario(self, scenario: Scenario) -> None:
        """Add a scenario to run."""
        # Inject simulator and environment
        scenario.simulator = self.simulator
        if scenario.environment is None:
            scenario.environment = self.environment
        
        self._scenarios.append(scenario)
    
    def create_reach_scenario(
        self,
        target: Optional[NDArray] = None,
        **kwargs
    ) -> ReachScenario:
        """Create and add a reach scenario."""
        scenario = ReachScenario(
            target_position=target,
            simulator=self.simulator,
            environment=self.environment,
            **kwargs
        )
        self._scenarios.append(scenario)
        return scenario
    
    def create_grasp_scenario(
        self,
        object_config: Optional[ObjectConfig] = None,
        **kwargs
    ) -> GraspScenario:
        """Create and add a grasp scenario."""
        scenario = GraspScenario(
            object_config=object_config,
            simulator=self.simulator,
            environment=self.environment,
            **kwargs
        )
        self._scenarios.append(scenario)
        return scenario
    
    def run_all(
        self,
        controller_callback: Callable[[ArmState, NDArray], NDArray]
    ) -> List[ScenarioResult]:
        """
        Run all scenarios.
        
        Args:
            controller_callback: Control function
            
        Returns:
            List of results
        """
        self._results = []
        
        for i, scenario in enumerate(self._scenarios):
            logger.info(f"Running scenario {i+1}/{len(self._scenarios)}: {scenario.config.name}")
            
            result = scenario.run(controller_callback)
            self._results.append(result)
            
            logger.info(f"  Result: {result.status.name}, Success Rate: {result.success_rate:.1%}")
        
        return self._results
    
    def get_results(self) -> List[ScenarioResult]:
        """Get all results."""
        return self._results
    
    def get_aggregate_stats(self) -> Dict[str, float]:
        """Get aggregate statistics across all scenarios."""
        if not self._results:
            return {}
        
        success_rates = [r.success_rate for r in self._results]
        durations = [r.mean_duration for r in self._results]
        errors = [r.mean_error for r in self._results]
        
        return {
            "overall_success_rate": np.mean(success_rates),
            "mean_duration": np.mean(durations),
            "mean_error": np.mean(errors),
            "n_scenarios": len(self._results),
            "n_successful": sum(1 for r in self._results if r.success_rate >= 0.5)
        }
    
    def print_summary(self) -> None:
        """Print summary of all results."""
        print("\n" + "=" * 50)
        print("SCENARIO RESULTS SUMMARY")
        print("=" * 50)
        
        for result in self._results:
            print(result.summary())
            print("-" * 50)
        
        stats = self.get_aggregate_stats()
        print("\nOVERALL STATISTICS:")
        print(f"  Scenarios: {stats.get('n_scenarios', 0)}")
        print(f"  Successful: {stats.get('n_successful', 0)}")
        print(f"  Overall Success Rate: {stats.get('overall_success_rate', 0):.1%}")
        print(f"  Mean Duration: {stats.get('mean_duration', 0):.2f}s")
        print(f"  Mean Error: {stats.get('mean_error', 0)*1000:.1f}mm")
        print("=" * 50)
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        import json
        
        data = {
            "aggregate": self.get_aggregate_stats(),
            "scenarios": [r.to_dict() for r in self._results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def clear(self) -> None:
        """Clear all scenarios and results."""
        self._scenarios = []
        self._results = []


# Convenience function
def create_standard_benchmark(
    simulator: Optional[ArmSimulator] = None,
    n_reach_trials: int = 5,
    n_grasp_trials: int = 3
) -> ScenarioRunner:
    """
    Create standard benchmark scenario suite.
    
    Args:
        simulator: Arm simulator
        n_reach_trials: Number of reach trials per position
        n_grasp_trials: Number of grasp trials per object
        
    Returns:
        Configured ScenarioRunner
    """
    runner = ScenarioRunner(simulator=simulator)
    
    # Easy reach targets
    easy_targets = [
        [0.4, 0, 0.2],
        [0.5, 0.1, 0.2],
        [0.5, -0.1, 0.2]
    ]
    
    for target in easy_targets:
        runner.create_reach_scenario(
            target=np.array(target),
            config=ScenarioConfig(
                name=f"reach_easy_{target[1]:.1f}",
                difficulty=TaskDifficulty.EASY,
                n_trials=n_reach_trials
            )
        )
    
    # Medium reach targets
    runner.create_reach_scenario(
        config=ScenarioConfig(
            name="reach_random",
            difficulty=TaskDifficulty.MEDIUM,
            n_trials=n_reach_trials * 2,
            randomize=True
        )
    )
    
    # Grasp scenarios
    runner.create_grasp_scenario(
        object_config=ObjectFactory.create_cube(size=0.05),
        config=ScenarioConfig(
            name="grasp_cube",
            difficulty=TaskDifficulty.MEDIUM,
            n_trials=n_grasp_trials
        )
    )
    
    runner.create_grasp_scenario(
        object_config=ObjectFactory.create_sphere(radius=0.03),
        config=ScenarioConfig(
            name="grasp_sphere",
            difficulty=TaskDifficulty.HARD,
            n_trials=n_grasp_trials
        )
    )
    
    return runner
