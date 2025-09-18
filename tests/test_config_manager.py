"""
Tests for the config_manager module.

These tests verify configuration loading, validation, environment
overrides, and hot-reloading functionality.
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from capybarish.config_manager import (
    ConfigFormat,
    ConfigManager,
    ConfigSource,
    RobotConfigValidator,
    create_robot_config_manager,
)


class TestConfigSource:
    """Test ConfigSource class."""

    def test_config_source_creation(self):
        """Test creating a ConfigSource."""
        source = ConfigSource(
            name="test",
            path="/path/to/config.yaml",
            format=ConfigFormat.YAML,
            required=True,
            watch=False,
        )

        assert source.name == "test"
        assert source.path == "/path/to/config.yaml"
        assert source.format == ConfigFormat.YAML
        assert source.required == True
        assert source.watch == False
        assert source.last_modified is None

    def test_config_source_defaults(self):
        """Test ConfigSource with default values."""
        source = ConfigSource(name="test", path="/path", format=ConfigFormat.JSON)

        assert source.required == True
        assert source.watch == False
        assert source.last_modified is None


class TestRobotConfigValidator:
    """Test RobotConfigValidator class."""

    def test_valid_config(self):
        """Test validation of a valid configuration."""
        validator = RobotConfigValidator()
        config = {
            "interface": {
                "module_ids": [1, 2, 3],
                "protocol": "UDP",
                "struct_format": "test_format",
            },
            "robot": {"dt": 0.02},
            "logging": {"robot_data_dir": None},
        }

        errors = validator.validate(config)
        assert errors == []

    def test_missing_required_sections(self):
        """Test validation with missing required sections."""
        validator = RobotConfigValidator()
        config = {"interface": {"module_ids": [1, 2, 3]}}

        errors = validator.validate(config)
        assert "Missing required section: robot" in errors
        assert "Missing required section: logging" in errors

    def test_invalid_module_ids(self):
        """Test validation with invalid module_ids."""
        validator = RobotConfigValidator()
        
        # Missing module_ids
        config1 = {
            "interface": {"protocol": "UDP"},
            "robot": {"dt": 0.02},
            "logging": {},
        }
        errors1 = validator.validate(config1)
        assert "interface.module_ids is required" in errors1

        # Non-list module_ids
        config2 = {
            "interface": {"module_ids": "not_a_list", "protocol": "UDP", "struct_format": "test"},
            "robot": {"dt": 0.02},
            "logging": {},
        }
        errors2 = validator.validate(config2)
        assert "interface.module_ids must be a list" in errors2

        # Empty module_ids
        config3 = {
            "interface": {"module_ids": [], "protocol": "UDP", "struct_format": "test"},
            "robot": {"dt": 0.02},
            "logging": {},
        }
        errors3 = validator.validate(config3)
        assert "interface.module_ids cannot be empty" in errors3

    def test_invalid_protocol(self):
        """Test validation with invalid protocol."""
        validator = RobotConfigValidator()
        config = {
            "interface": {
                "module_ids": [1, 2, 3],
                "protocol": "INVALID",
                "struct_format": "test_format",
            },
            "robot": {"dt": 0.02},
            "logging": {},
        }

        errors = validator.validate(config)
        assert "interface.protocol must be one of: UDP, TCP, USB" in errors

    def test_missing_struct_format(self):
        """Test validation with missing struct_format."""
        validator = RobotConfigValidator()
        config = {
            "interface": {"module_ids": [1, 2, 3], "protocol": "UDP"},
            "robot": {"dt": 0.02},
            "logging": {},
        }

        errors = validator.validate(config)
        assert "interface.struct_format is required" in errors

    def test_invalid_dt(self):
        """Test validation with invalid dt."""
        validator = RobotConfigValidator()
        
        # Missing dt
        config1 = {
            "interface": {"module_ids": [1, 2, 3], "protocol": "UDP", "struct_format": "test"},
            "robot": {},
            "logging": {},
        }
        errors1 = validator.validate(config1)
        assert "robot.dt is required" in errors1

        # Non-numeric dt
        config2 = {
            "interface": {"module_ids": [1, 2, 3], "protocol": "UDP", "struct_format": "test"},
            "robot": {"dt": "not_a_number"},
            "logging": {},
        }
        errors2 = validator.validate(config2)
        assert "robot.dt must be a positive number" in errors2

        # Negative dt
        config3 = {
            "interface": {"module_ids": [1, 2, 3], "protocol": "UDP", "struct_format": "test"},
            "robot": {"dt": -0.01},
            "logging": {},
        }
        errors3 = validator.validate(config3)
        assert "robot.dt must be a positive number" in errors3


class TestConfigManager:
    """Test ConfigManager class."""

    def test_config_manager_creation(self):
        """Test creating a ConfigManager."""
        manager = ConfigManager()
        assert manager.config == {}
        assert manager.change_callbacks == []
        assert manager.environment == "development"

    def test_add_source(self):
        """Test adding configuration sources."""
        manager = ConfigManager()
        source = ConfigSource("test", "/path", ConfigFormat.YAML)

        manager.add_source(source)
        assert len(manager.sources) == 1
        assert manager.sources[0] == source

    def test_add_validator(self):
        """Test adding configuration validators."""
        manager = ConfigManager()
        validator = RobotConfigValidator()

        manager.add_validator(validator)
        assert len(manager.validators) == 1
        assert manager.validators[0] == validator

    def test_add_change_callback(self):
        """Test adding change callbacks."""
        manager = ConfigManager()
        callback = Mock()

        manager.add_change_callback(callback)
        assert len(manager.change_callbacks) == 1
        assert manager.change_callbacks[0] == callback

    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML configuration."""
        manager = ConfigManager()

        # Create a test YAML file
        config_file = temp_dir / "test.yaml"
        config_data = {"test_key": "test_value", "number": 42}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        source = ConfigSource("test", config_file, ConfigFormat.YAML)
        manager.add_source(source)

        config = manager.load()
        assert config["test_key"] == "test_value"
        assert config["number"] == 42

    def test_load_json_config(self, temp_dir):
        """Test loading JSON configuration."""
        manager = ConfigManager()

        # Create a test JSON file
        config_file = temp_dir / "test.json"
        config_data = {"test_key": "test_value", "number": 42}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        source = ConfigSource("test", config_file, ConfigFormat.JSON)
        manager.add_source(source)

        config = manager.load()
        assert config["test_key"] == "test_value"
        assert config["number"] == 42

    def test_load_nonexistent_required_file(self, temp_dir):
        """Test loading non-existent required file."""
        manager = ConfigManager()
        source = ConfigSource("test", temp_dir / "nonexistent.yaml", ConfigFormat.YAML, required=True)
        manager.add_source(source)

        with pytest.raises(RuntimeError, match="Failed to load required config source"):
            manager.load()

    def test_load_nonexistent_optional_file(self, temp_dir):
        """Test loading non-existent optional file."""
        manager = ConfigManager()
        source = ConfigSource("test", temp_dir / "nonexistent.yaml", ConfigFormat.YAML, required=False)
        manager.add_source(source)

        config = manager.load()
        assert config == {}

    def test_merge_configs(self):
        """Test configuration merging."""
        manager = ConfigManager()

        base_config = {"a": 1, "b": {"x": 10, "y": 20}}
        override_config = {"b": {"y": 30, "z": 40}, "c": 3}

        merged = manager._merge_configs(base_config, override_config)

        assert merged["a"] == 1
        assert merged["b"]["x"] == 10
        assert merged["b"]["y"] == 30  # Override
        assert merged["b"]["z"] == 40
        assert merged["c"] == 3

    def test_validation_failure(self, temp_dir):
        """Test configuration validation failure."""
        manager = ConfigManager()

        # Create invalid config
        config_file = temp_dir / "invalid.yaml"
        config_data = {"interface": {"module_ids": "not_a_list"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        source = ConfigSource("test", config_file, ConfigFormat.YAML)
        manager.add_source(source)
        manager.add_validator(RobotConfigValidator())

        with pytest.raises(ValueError, match="Configuration validation failed"):
            manager.load()

    def test_change_callback_notification(self, temp_dir):
        """Test that change callbacks are called."""
        manager = ConfigManager()
        callback = Mock()
        manager.add_change_callback(callback)

        # Create config file
        config_file = temp_dir / "test.yaml"
        config_data = {"test": "value"}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        source = ConfigSource("test", config_file, ConfigFormat.YAML)
        manager.add_source(source)

        manager.load()
        callback.assert_called_once()

    def test_get_config_value(self, temp_dir):
        """Test getting configuration values using dot notation."""
        manager = ConfigManager()

        # Create config file
        config_file = temp_dir / "test.yaml"
        config_data = {"section": {"subsection": {"value": 42}}, "top_level": "test"}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        source = ConfigSource("test", config_file, ConfigFormat.YAML)
        manager.add_source(source)
        manager.load()

        assert manager.get("section.subsection.value") == 42
        assert manager.get("top_level") == "test"
        assert manager.get("nonexistent.key", "default") == "default"

    def test_set_config_value(self):
        """Test setting configuration values using dot notation."""
        manager = ConfigManager()
        callback = Mock()
        manager.add_change_callback(callback)

        manager.set("section.subsection.value", 42)
        assert manager.get("section.subsection.value") == 42

        # Should call change callbacks
        callback.assert_called()

    def test_unsupported_format(self, temp_dir):
        """Test loading unsupported configuration format."""
        manager = ConfigManager()

        config_file = temp_dir / "test.txt"
        config_file.write_text("not yaml or json")

        source = ConfigSource("test", config_file, ConfigFormat.YAML)
        manager.add_source(source)
        
        # Mock the format to be unsupported
        with patch.object(source, 'format', 'UNSUPPORTED'):
            with pytest.raises(RuntimeError, match="Failed to load required config source"):
                manager.load()

    def test_file_watching_start_stop(self):
        """Test starting and stopping file watching."""
        manager = ConfigManager()

        # Test starting
        manager.start_watching()
        assert manager._watch_thread is not None
        assert manager._watch_thread.is_alive()

        # Test stopping
        manager.stop_watching()
        assert not manager._watch_thread.is_alive()


class TestCreateRobotConfigManager:
    """Test the create_robot_config_manager function."""

    def test_create_robot_config_manager_defaults(self):
        """Test creating robot config manager with defaults."""
        with patch("capybarish.config_manager.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            manager = create_robot_config_manager()

            assert manager.environment == "development"
            assert len(manager.sources) >= 1  # At least main config
            assert len(manager.validators) >= 1  # At least robot validator

    def test_create_robot_config_manager_with_env_override(self, temp_dir):
        """Test creating robot config manager with environment override."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create main config
        main_config = config_dir / "test.yaml"
        main_config.write_text("main: true")

        # Create environment override
        env_config = config_dir / "test.production.yaml"
        env_config.write_text("env: production")

        manager = create_robot_config_manager(
            config_dir=config_dir, config_name="test", environment="production"
        )

        assert manager.environment == "production"
        assert len(manager.sources) == 2  # Main + environment override


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def test_full_config_lifecycle(self, temp_dir):
        """Test complete configuration lifecycle."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create main config
        main_config = config_dir / "main.yaml"
        config_data = {
            "interface": {
                "module_ids": [1, 2, 3],
                "protocol": "UDP",
                "struct_format": "test_format",
            },
            "robot": {"dt": 0.02},
            "logging": {"robot_data_dir": None},
        }
        with open(main_config, "w") as f:
            yaml.dump(config_data, f)

        # Create manager
        manager = create_robot_config_manager(config_dir=config_dir, config_name="main")

        # Load and validate
        config = manager.load()
        assert config["interface"]["module_ids"] == [1, 2, 3]
        assert config["robot"]["dt"] == 0.02

        # Test getting values
        assert manager.get("interface.protocol") == "UDP"
        assert manager.get("robot.dt") == 0.02

        # Test setting values
        manager.set("robot.dt", 0.01)
        assert manager.get("robot.dt") == 0.01

    def test_config_with_environment_override(self, temp_dir):
        """Test configuration with environment-specific overrides."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create base config
        base_config = config_dir / "app.yaml"
        base_data = {
            "interface": {
                "module_ids": [1, 2],
                "protocol": "UDP",
                "struct_format": "base_format",
            },
            "robot": {"dt": 0.02},
            "logging": {"robot_data_dir": None},
        }
        with open(base_config, "w") as f:
            yaml.dump(base_data, f)

        # Create environment override
        env_config = config_dir / "app.production.yaml"
        env_data = {
            "interface": {"module_ids": [1, 2, 3, 4]},  # More modules in production
            "robot": {"dt": 0.01},  # Faster in production
        }
        with open(env_config, "w") as f:
            yaml.dump(env_data, f)

        # Create manager with production environment
        manager = create_robot_config_manager(
            config_dir=config_dir, config_name="app", environment="production"
        )

        config = manager.load()

        # Check that environment overrides were applied
        assert config["interface"]["module_ids"] == [1, 2, 3, 4]  # From env override
        assert config["robot"]["dt"] == 0.01  # From env override
        assert config["interface"]["protocol"] == "UDP"  # From base config