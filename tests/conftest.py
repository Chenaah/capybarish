"""
Shared test fixtures and utilities for capybarish tests.

This module provides common fixtures, mock objects, and utilities
used across multiple test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from capybarish.plugin_system import PluginMetadata, PluginType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration object for testing."""
    config_dict = {
        "interface": {
            "module_ids": [1, 2, 3],
            "torso_module_id": 1,
            "sources": ["imu", "optitrack"],
            "struct_format": "test_format",
            "protocol": "UDP",
            "dashboard": True,
            "check_action_safety": True,
            "optitrack_rigibody": 1,
            "enable_filter": True,
            "kp_ratio": 1.0,
            "kd_ratio": 1.0,
            "calibration_modes": None,
            "broken_motors": None,
        },
        "robot": {
            "dt": 0.02,
            "motor_range": [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
        },
        "agent": {"filter_action": True},
        "logging": {"robot_data_dir": None},
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_module_data():
    """Create mock data that would come from a robot module."""
    return {
        "motor_pos": 0.5,
        "motor_vel": 0.1,
        "motor_torque": 1.2,
        "motor_mode": 2,
        "motor_error": 0,
        "motor_on": True,
        "voltage": 12.5,
        "current": 0.8,
        "temperature": 45.2,
        "energy": 10.5,
        "acc_body_imu": [0.1, 0.2, 9.8],
        "body_omega_imu": [0.01, 0.02, 0.03],
        "body_rot_imu": [0, 0, 0, 1],
        "pos_world_uwb": [1.0, 2.0, 0.5],
        "vel_world_uwb": [0.1, 0.2, 0.0],
        "esp_errors": [1],
        "log_info": "Test log message",
        "switch_off_request": False,
        "last_rcv_timestamp": 12345.67,
        "latency": 0.005,
    }


@pytest.fixture
def mock_imu_data():
    """Create mock IMU data for testing."""
    return {
        "acc_body_imu": np.array([0.1, 0.2, 9.8]),
        "body_omega_imu": np.array([0.01, 0.02, 0.03]),
        "body_rot_imu": np.array([0, 0, 0, 1]),
        "temperature": 25.0,
        "timestamp": 12345.67,
    }


@pytest.fixture
def mock_optitrack_data():
    """Create mock OptiTrack data for testing."""
    return {
        "frame_number": 12345,
        "rigid_bodies": {
            1: {
                "position": [1.0, 2.0, 0.5],
                "rotation": [0, 0, 0, 1],
                "tracking_valid": True,
            }
        },
        "timestamp": 12345.67,
    }


@pytest.fixture
def mock_communication_manager():
    """Create a mock communication manager for testing."""
    mock_comm = Mock()
    mock_comm.setup.return_value = True
    mock_comm.send_command.return_value = True
    mock_comm.receive_data_batch.return_value = {}
    mock_comm.get_connected_modules.return_value = [1, 2, 3]
    mock_comm.get_all_modules_info.return_value = {}
    mock_comm.close.return_value = None
    return mock_comm


@pytest.fixture
def mock_plugin_metadata():
    """Create mock plugin metadata for testing."""
    return PluginMetadata(
        name="TestPlugin",
        version="1.0.0",
        description="Test plugin for unit testing",
        author="Test Author",
        plugin_type=PluginType.DATA_PROCESSOR,
        dependencies=[],
        tags={"test", "mock"},
    )


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create a sample YAML configuration file for testing."""
    config_content = """
interface:
  module_ids: [1, 2, 3]
  torso_module_id: 1
  sources: ["imu"]
  struct_format: "test_format"
  protocol: "UDP"
  dashboard: true
  check_action_safety: true

robot:
  dt: 0.02
  motor_range: [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]]

agent:
  filter_action: true

logging:
  robot_data_dir: null
"""
    config_path = temp_dir / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


class MockNatNetClient:
    """Mock NatNet client for testing OptiTrack functionality."""

    def __init__(self):
        self.client_address = "0.0.0.0"
        self.server_address = "127.0.0.1"
        self.use_multicast = False
        self.print_level = 0
        self.new_frame_listener = None
        self.rigid_body_listener = None
        self.is_running = False

    def set_client_address(self, address):
        self.client_address = address

    def set_server_address(self, address):
        self.server_address = address

    def set_use_multicast(self, use_multicast):
        self.use_multicast = use_multicast

    def set_print_level(self, level):
        self.print_level = level

    def run(self):
        self.is_running = True
        return True

    def shutdown(self):
        self.is_running = False

    def simulate_frame(self, frame_data):
        """Simulate receiving a frame from OptiTrack."""
        if self.new_frame_listener:
            self.new_frame_listener(frame_data)

    def simulate_rigid_body(self, body_id, position, rotation):
        """Simulate receiving rigid body data from OptiTrack."""
        if self.rigid_body_listener:
            self.rigid_body_listener(body_id, position, rotation)


class MockUDPSocket:
    """Mock UDP socket for testing network communication."""

    def __init__(self):
        self.bound_address = None
        self.sent_data = []
        self.receive_queue = []

    def bind(self, address):
        self.bound_address = address

    def sendto(self, data, address):
        self.sent_data.append((data, address))
        return len(data)

    def recvfrom(self, buffer_size):
        if self.receive_queue:
            return self.receive_queue.pop(0)
        raise BlockingIOError("No data available")

    def settimeout(self, timeout):
        pass

    def setblocking(self, blocking):
        pass

    def close(self):
        pass

    def add_receive_data(self, data, address):
        """Add data to the receive queue for testing."""
        self.receive_queue.append((data, address))


def create_test_plugin_config(plugin_type: str = "data_processor", **kwargs) -> Dict[str, Any]:
    """Create a test plugin configuration."""
    config = {
        "enabled": True,
        "debug": False,
        **kwargs,
    }
    return config


def assert_dict_almost_equal(dict1: Dict[str, Any], dict2: Dict[str, Any], places: int = 7):
    """Assert that two dictionaries are approximately equal, handling numpy arrays."""
    assert dict1.keys() == dict2.keys(), f"Dictionary keys don't match: {dict1.keys()} vs {dict2.keys()}"

    for key in dict1.keys():
        val1, val2 = dict1[key], dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            np.testing.assert_array_almost_equal(val1, val2, decimal=places)
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            assert abs(val1 - val2) < 10**(-places), f"Values for key '{key}' not close: {val1} vs {val2}"
        else:
            assert val1 == val2, f"Values for key '{key}' don't match: {val1} vs {val2}"


def create_mock_robot_modules(module_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Create mock data for multiple robot modules."""
    modules = {}
    for module_id in module_ids:
        modules[module_id] = {
            "module_id": module_id,
            "motor_pos": 0.0 + module_id * 0.1,
            "motor_vel": 0.0 + module_id * 0.01,
            "motor_torque": 0.0 + module_id * 0.1,
            "motor_mode": 2,
            "motor_error": 0,
            "motor_on": True,
            "voltage": 12.0 + module_id * 0.1,
            "current": 0.5 + module_id * 0.1,
            "temperature": 25.0 + module_id,
            "energy": 5.0 + module_id,
            "acc_body_imu": [0.1 * module_id, 0.2 * module_id, 9.8],
            "body_omega_imu": [0.01 * module_id, 0.02 * module_id, 0.03 * module_id],
            "body_rot_imu": [0, 0, 0, 1],
            "esp_errors": [1],
            "log_info": f"Module {module_id} log",
            "switch_off_request": False,
            "last_rcv_timestamp": 12345.67 + module_id,
            "latency": 0.005 + module_id * 0.001,
        }
    return modules