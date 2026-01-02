"""
Simulation Module
=================

Physics-based simulation environment for the bionic arm using
PyBullet or MuJoCo backends for testing and development.

Components:
    - ArmSimulator: Physics simulation of 7-DOF arm
    - SimulationEnvironment: World setup and object management
    - ScenarioRunner: Test scenario execution

Author: Bionic Arm Project Team
License: MIT
"""

from .arm_simulator import (
    SimulatorConfig,
    PhysicsBackend,
    ArmSimulator,
)

from .environment import (
    ObjectConfig,
    SimObject,
    SimulationEnvironment,
)

from .scenarios import (
    ScenarioConfig,
    ScenarioResult,
    Scenario,
    ReachScenario,
    GraspScenario,
    ScenarioRunner,
)

__version__ = "0.1.0"

__all__ = [
    "SimulatorConfig",
    "PhysicsBackend",
    "ArmSimulator",
    "ObjectConfig",
    "SimObject",
    "SimulationEnvironment",
    "ScenarioConfig",
    "ScenarioResult",
    "Scenario",
    "ReachScenario",
    "GraspScenario",
    "ScenarioRunner",
]
