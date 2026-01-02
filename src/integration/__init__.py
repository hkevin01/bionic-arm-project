"""
Integration Module
==================

System integration and orchestration for the bionic arm,
connecting BCI, control, feedback, and simulation components.

Components:
    - BionicArmSystem: Main system orchestrator
    - SystemConfig: Configuration management
    - SystemMonitor: Health and performance monitoring

Author: Bionic Arm Project Team
License: MIT
"""

from .system import (
    SystemConfig,
    SystemState,
    SystemMode,
    BionicArmSystem,
)

from .monitor import (
    MonitorConfig,
    SystemMetrics,
    SystemMonitor,
)

__version__ = "0.1.0"

__all__ = [
    "SystemConfig",
    "SystemState",
    "SystemMode",
    "BionicArmSystem",
    "MonitorConfig",
    "SystemMetrics",
    "SystemMonitor",
]
