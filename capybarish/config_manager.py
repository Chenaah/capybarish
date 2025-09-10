"""
Modern configuration management system for robot middleware.

This module provides centralized configuration management with validation,
environment support, and hot-reloading capabilities.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: Union[str, Path]
    format: ConfigFormat
    required: bool = True
    watch: bool = False
    last_modified: Optional[float] = None


class ConfigValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of error messages.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of error messages (empty if valid)
        """
        pass


class RobotConfigValidator(ConfigValidator):
    """Validator for robot-specific configuration."""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate robot configuration."""
        errors = []
        
        # Check required top-level sections
        required_sections = ['interface', 'robot', 'logging']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate interface section
        if 'interface' in config:
            interface_config = config['interface']
            
            if 'module_ids' not in interface_config:
                errors.append("interface.module_ids is required")
            elif not isinstance(interface_config['module_ids'], list):
                errors.append("interface.module_ids must be a list")
            elif not interface_config['module_ids']:
                errors.append("interface.module_ids cannot be empty")
            
            if 'protocol' in interface_config:
                if interface_config['protocol'] not in ['UDP', 'TCP', 'USB']:
                    errors.append("interface.protocol must be one of: UDP, TCP, USB")
            
            if 'struct_format' not in interface_config:
                errors.append("interface.struct_format is required")
        
        # Validate robot section
        if 'robot' in config:
            robot_config = config['robot']
            
            if 'dt' not in robot_config:
                errors.append("robot.dt is required")
            elif not isinstance(robot_config['dt'], (int, float)) or robot_config['dt'] <= 0:
                errors.append("robot.dt must be a positive number")
        
        return errors


@dataclass
class ConfigManager:
    """
    Modern configuration manager with validation and hot-reloading.
    
    Features:
    - Multiple configuration sources
    - Environment-specific overrides
    - Configuration validation
    - Hot-reloading support
    - Change notifications
    """
    
    sources: List[ConfigSource] = field(default_factory=list)
    validators: List[ConfigValidator] = field(default_factory=list)
    environment: str = "development"
    
    def __post_init__(self):
        self.config: Dict[str, Any] = {}
        self.change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        self._lock = threading.RLock()
        
    def add_source(self, source: ConfigSource) -> None:
        """Add a configuration source."""
        with self._lock:
            self.sources.append(source)
    
    def add_validator(self, validator: ConfigValidator) -> None:
        """Add a configuration validator."""
        self.validators.append(validator)
    
    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be called when configuration changes."""
        self.change_callbacks.append(callback)
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from all sources."""
        with self._lock:
            merged_config = {}
            
            for source in self.sources:
                try:
                    source_config = self._load_source(source)
                    merged_config = self._merge_configs(merged_config, source_config)
                    logger.info(f"Loaded config from {source.name}")
                except Exception as e:
                    if source.required:
                        raise RuntimeError(f"Failed to load required config source {source.name}: {e}")
                    else:
                        logger.warning(f"Failed to load optional config source {source.name}: {e}")
            
            # Apply environment-specific overrides
            merged_config = self._apply_environment_overrides(merged_config)
            
            # Validate configuration
            self._validate_config(merged_config)
            
            # Update internal state
            old_config = self.config.copy()
            self.config = merged_config
            
            # Notify callbacks if config changed
            if old_config != self.config:
                self._notify_change_callbacks()
            
            return self.config
    
    def _load_source(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from a single source."""
        path = Path(source.path)
        
        if not path.exists():
            if source.required:
                raise FileNotFoundError(f"Required config file not found: {path}")
            return {}
        
        # Update last modified time
        source.last_modified = path.stat().st_mtime
        
        # Load based on format
        with open(path, 'r') as f:
            if source.format == ConfigFormat.YAML:
                return yaml.safe_load(f) or {}
            elif source.format == ConfigFormat.JSON:
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {source.format}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        env_key = f"environments.{self.environment}"
        
        if env_key.replace('.', '_') in config:
            env_config = config[env_key.replace('.', '_')]
            config = self._merge_configs(config, env_config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using all validators."""
        all_errors = []
        
        for validator in self.validators:
            errors = validator.validate(config)
            all_errors.extend(errors)
        
        if all_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in all_errors)
            raise ValueError(error_msg)
    
    def _notify_change_callbacks(self) -> None:
        """Notify all change callbacks."""
        for callback in self.change_callbacks:
            try:
                callback(self.config.copy())
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        with self._lock:
            keys = key.split('.')
            target = self.config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the value
            target[keys[-1]] = value
            
            # Notify callbacks
            self._notify_change_callbacks()
    
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        if self._watch_thread and self._watch_thread.is_alive():
            return
        
        self._stop_watching.clear()
        self._watch_thread = threading.Thread(target=self._watch_files, daemon=True)
        self._watch_thread.start()
        logger.info("Started configuration file watching")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self._watch_thread:
            self._stop_watching.set()
            self._watch_thread.join(timeout=5.0)
            logger.info("Stopped configuration file watching")
    
    def _watch_files(self) -> None:
        """Watch configuration files for changes."""
        while not self._stop_watching.is_set():
            try:
                changed = False
                
                for source in self.sources:
                    if not source.watch:
                        continue
                    
                    path = Path(source.path)
                    if path.exists():
                        current_mtime = path.stat().st_mtime
                        if source.last_modified and current_mtime > source.last_modified:
                            logger.info(f"Configuration file changed: {source.name}")
                            changed = True
                            break
                
                if changed:
                    self.load()
                
            except Exception as e:
                logger.error(f"Error watching config files: {e}")
            
            self._stop_watching.wait(1.0)  # Check every second


def create_robot_config_manager(config_dir: Union[str, Path] = "config", 
                               config_name: str = "default",
                               environment: str = "development") -> ConfigManager:
    """
    Create a configuration manager for robot applications.
    
    Args:
        config_dir: Directory containing configuration files
        config_name: Base configuration file name (without extension)
        environment: Environment name for overrides
        
    Returns:
        Configured ConfigManager instance
    """
    config_manager = ConfigManager(environment=environment)
    
    # Add main configuration source
    main_config_path = Path(config_dir) / f"{config_name}.yaml"
    config_manager.add_source(ConfigSource(
        name="main",
        path=main_config_path,
        format=ConfigFormat.YAML,
        required=True,
        watch=True
    ))
    
    # Add environment-specific override if it exists
    env_config_path = Path(config_dir) / f"{config_name}.{environment}.yaml"
    if env_config_path.exists():
        config_manager.add_source(ConfigSource(
            name=f"environment-{environment}",
            path=env_config_path,
            format=ConfigFormat.YAML,
            required=False,
            watch=True
        ))
    
    # Add robot configuration validator
    config_manager.add_validator(RobotConfigValidator())
    
    return config_manager


# Backward compatibility function
def load_cfg(config_name: str, config_dir: str = "config") -> Any:
    """
    Load configuration using the legacy interface for backward compatibility.
    
    This function maintains compatibility with existing code while using
    the new configuration manager internally.
    """
    config_manager = create_robot_config_manager(config_dir, config_name)
    config_dict = config_manager.load()
    
    # Convert to object-like access (maintaining backward compatibility)
    return _dict_to_object(config_dict)


def _dict_to_object(d: Dict[str, Any]) -> Any:
    """Convert dictionary to object with attribute access."""
    class ConfigObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, _dict_to_object(value))
                else:
                    setattr(self, key, value)
    
    return ConfigObject(d)
