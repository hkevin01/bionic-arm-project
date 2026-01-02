"""
Pytest Configuration and Fixtures
==================================

Shared test configuration and fixtures for all test modules.
Handles path setup for importing src modules.

Author: Bionic Arm Project Team
License: MIT
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np


# =============================================================================
# Global Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path."""
    return project_root


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "hardware: marks tests requiring hardware")
    config.addinivalue_line("markers", "integration: marks integration tests")
