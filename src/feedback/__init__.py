"""
Feedback Module
===============

Sensory feedback systems for the bionic arm including vibrotactile
feedback for touch sensation and visual feedback for monitoring.

Components:
    - VibrotactileFeedback: Vibration patterns for touch
    - VisualFeedback: Visual display of arm state

Author: Bionic Arm Project Team
License: MIT
"""

from .vibrotactile import (
    VibrotactileConfig,
    VibrationPattern,
    VibrotactileEncoder,
    VibrotactileFeedback,
)

from .visual import (
    VisualFeedbackConfig,
    VisualFeedback,
)

__version__ = "0.1.0"

__all__ = [
    "VibrotactileConfig",
    "VibrationPattern", 
    "VibrotactileEncoder",
    "VibrotactileFeedback",
    "VisualFeedbackConfig",
    "VisualFeedback",
]
