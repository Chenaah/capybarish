"""
Plugin system for extensible robot middleware.

This module provides a flexible plugin architecture for adding new data sources,
processors, and components without modifying the core system.

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

import importlib
import inspect
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported."""
    DATA_SOURCE = "data_source"
    DATA_PROCESSOR = "data_processor"
    COMMUNICATION = "communication"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    FILTER = "filter"
    LOGGER = "logger"
    DASHBOARD = "dashboard"


class PluginStatus(Enum):
    """Plugin status enumeration."""
    LOADED = "loaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: Set[str] = field(default_factory=set)


class Plugin(ABC):
    """
    Abstract base class for all plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the plugin with configuration."""
        self.config = config
        self.status = PluginStatus.LOADED
        self.last_error: Optional[str] = None
        self._stop_event = threading.Event()
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start the plugin.
        
        Returns:
            True if start successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the plugin.
        
        Returns:
            True if stop successful, False otherwise
        """
        pass
    
    def get_status(self) -> PluginStatus:
        """Get current plugin status."""
        return self.status
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message."""
        return self.last_error
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        # Default implementation - can be overridden
        return []


class DataSourcePlugin(Plugin):
    """Base class for data source plugins."""
    
    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Get data from the source.
        
        Returns:
            Dictionary containing the data
        """
        pass
    
    def supports_streaming(self) -> bool:
        """Return True if this data source supports streaming."""
        return False
    
    def start_streaming(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Start streaming data to a callback function.
        
        Args:
            callback: Function to call with new data
            
        Returns:
            True if streaming started successfully
        """
        return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming data."""
        return False


class DataProcessorPlugin(Plugin):
    """Base class for data processor plugins."""
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return processed data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        pass
    
    def supports_batch_processing(self) -> bool:
        """Return True if this processor supports batch processing."""
        return False
    
    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of data items.
        
        Args:
            data_list: List of data items to process
            
        Returns:
            List of processed data items
        """
        return [self.process(data) for data in data_list]


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    plugin: Plugin
    metadata: PluginMetadata
    status: PluginStatus
    loaded_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0


class PluginManager:
    """
    Plugin manager for loading, managing, and coordinating plugins.
    
    Provides plugin discovery, loading, lifecycle management, and communication
    between plugins and the core system.
    """
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.plugin_directories = plugin_directories or ['plugins']
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {}
        
        # Event callbacks
        self.on_plugin_loaded: List[Callable[[str, PluginInfo], None]] = []
        self.on_plugin_started: List[Callable[[str, PluginInfo], None]] = []
        self.on_plugin_stopped: List[Callable[[str, PluginInfo], None]] = []
        self.on_plugin_error: List[Callable[[str, PluginInfo, str], None]] = []
        
        # Data flow management
        self.data_sources: Dict[str, DataSourcePlugin] = {}
        self.data_processors: Dict[str, DataProcessorPlugin] = {}
        self.data_pipelines: List[List[str]] = []  # List of plugin chains
        
        # Threading
        self._lock = threading.RLock()
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()
        
        # Statistics
        self.stats = {
            'plugins_loaded': 0,
            'plugins_started': 0,
            'plugins_stopped': 0,
            'errors': 0,
            'data_processed': 0
        }
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directories.
        
        Returns:
            List of discovered plugin module names
        """
        discovered = []
        
        for directory in self.plugin_directories:
            plugin_dir = Path(directory)
            if not plugin_dir.exists():
                continue
            
            # Look for Python files
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = py_file.stem
                discovered.append(f"{directory.replace('/', '.')}.{module_name}")
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def load_plugin(self, module_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from a module.
        
        Args:
            module_name: Name of the module to load
            config: Configuration for the plugin
            
        Returns:
            True if plugin loaded successfully
        """
        try:
            with self._lock:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find plugin classes in the module
                plugin_classes = []
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Plugin) and 
                        obj != Plugin and
                        not inspect.isabstract(obj)):
                        plugin_classes.append(obj)
                
                if not plugin_classes:
                    logger.error(f"No plugin classes found in {module_name}")
                    return False
                
                # Use the first plugin class found
                plugin_class = plugin_classes[0]
                
                # Create plugin instance
                plugin_config = config or {}
                plugin_instance = plugin_class(plugin_config)
                
                # Get metadata
                metadata = plugin_instance.metadata
                plugin_name = metadata.name
                
                # Validate configuration
                validation_errors = plugin_instance.validate_config(plugin_config)
                if validation_errors:
                    logger.error(f"Plugin {plugin_name} configuration invalid: {validation_errors}")
                    return False
                
                # Store plugin info
                plugin_info = PluginInfo(
                    plugin=plugin_instance,
                    metadata=metadata,
                    status=PluginStatus.LOADED
                )
                
                self.plugins[plugin_name] = plugin_info
                
                # Update type index
                if metadata.plugin_type not in self.plugin_types:
                    self.plugin_types[metadata.plugin_type] = []
                self.plugin_types[metadata.plugin_type].append(plugin_name)
                
                # Register data sources and processors
                if isinstance(plugin_instance, DataSourcePlugin):
                    self.data_sources[plugin_name] = plugin_instance
                elif isinstance(plugin_instance, DataProcessorPlugin):
                    self.data_processors[plugin_name] = plugin_instance
                
                # Update statistics
                self.stats['plugins_loaded'] += 1
                
                # Notify callbacks
                for callback in self.on_plugin_loaded:
                    try:
                        callback(plugin_name, plugin_info)
                    except Exception as e:
                        logger.error(f"Error in plugin loaded callback: {e}")
                
                logger.info(f"Loaded plugin: {plugin_name} v{metadata.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load plugin {module_name}: {e}")
            self.stats['errors'] += 1
            return False
    
    def initialize_plugin(self, plugin_name: str) -> bool:
        """Initialize a loaded plugin."""
        with self._lock:
            if plugin_name not in self.plugins:
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            try:
                success = plugin_info.plugin.initialize()
                if success:
                    plugin_info.status = PluginStatus.INITIALIZED
                    plugin_info.last_updated = time.time()
                    logger.info(f"Initialized plugin: {plugin_name}")
                else:
                    plugin_info.status = PluginStatus.ERROR
                    self.stats['errors'] += 1
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_count += 1
                self.stats['errors'] += 1
                
                # Notify error callbacks
                for callback in self.on_plugin_error:
                    try:
                        callback(plugin_name, plugin_info, str(e))
                    except Exception as cb_e:
                        logger.error(f"Error in plugin error callback: {cb_e}")
                
                return False
    
    def start_plugin(self, plugin_name: str) -> bool:
        """Start an initialized plugin."""
        with self._lock:
            if plugin_name not in self.plugins:
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            if plugin_info.status != PluginStatus.INITIALIZED:
                # Try to initialize first
                if not self.initialize_plugin(plugin_name):
                    return False
            
            try:
                success = plugin_info.plugin.start()
                if success:
                    plugin_info.status = PluginStatus.RUNNING
                    plugin_info.last_updated = time.time()
                    self.stats['plugins_started'] += 1
                    
                    # Notify callbacks
                    for callback in self.on_plugin_started:
                        try:
                            callback(plugin_name, plugin_info)
                        except Exception as e:
                            logger.error(f"Error in plugin started callback: {e}")
                    
                    logger.info(f"Started plugin: {plugin_name}")
                else:
                    plugin_info.status = PluginStatus.ERROR
                    self.stats['errors'] += 1
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to start plugin {plugin_name}: {e}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_count += 1
                self.stats['errors'] += 1
                return False
    
    def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a running plugin."""
        with self._lock:
            if plugin_name not in self.plugins:
                return False
            
            plugin_info = self.plugins[plugin_name]
            
            try:
                success = plugin_info.plugin.stop()
                if success:
                    plugin_info.status = PluginStatus.STOPPED
                    plugin_info.last_updated = time.time()
                    self.stats['plugins_stopped'] += 1
                    
                    # Notify callbacks
                    for callback in self.on_plugin_stopped:
                        try:
                            callback(plugin_name, plugin_info)
                        except Exception as e:
                            logger.error(f"Error in plugin stopped callback: {e}")
                    
                    logger.info(f"Stopped plugin: {plugin_name}")
                else:
                    plugin_info.status = PluginStatus.ERROR
                    self.stats['errors'] += 1
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to stop plugin {plugin_name}: {e}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_count += 1
                self.stats['errors'] += 1
                return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a plugin instance by name."""
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info.plugin if plugin_info else None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """Get list of plugin names by type."""
        return self.plugin_types.get(plugin_type, [])
    
    def get_data_from_source(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get data from a data source plugin."""
        if source_name in self.data_sources:
            try:
                data = self.data_sources[source_name].get_data()
                self.stats['data_processed'] += 1
                return data
            except Exception as e:
                logger.error(f"Error getting data from {source_name}: {e}")
                self.stats['errors'] += 1
        return None
    
    def process_data(self, processor_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process data using a data processor plugin."""
        if processor_name in self.data_processors:
            try:
                processed_data = self.data_processors[processor_name].process(data)
                self.stats['data_processed'] += 1
                return processed_data
            except Exception as e:
                logger.error(f"Error processing data with {processor_name}: {e}")
                self.stats['errors'] += 1
        return None
    
    def create_data_pipeline(self, pipeline: List[str]) -> bool:
        """
        Create a data processing pipeline.
        
        Args:
            pipeline: List of plugin names forming the pipeline
            
        Returns:
            True if pipeline is valid and created
        """
        # Validate pipeline
        for plugin_name in pipeline:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin {plugin_name} not found for pipeline")
                return False
        
        self.data_pipelines.append(pipeline)
        logger.info(f"Created data pipeline: {' -> '.join(pipeline)}")
        return True
    
    def execute_pipeline(self, pipeline_index: int, initial_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a data pipeline."""
        if pipeline_index >= len(self.data_pipelines):
            return None
        
        pipeline = self.data_pipelines[pipeline_index]
        current_data = initial_data
        
        for plugin_name in pipeline:
            if plugin_name in self.data_processors:
                current_data = self.process_data(plugin_name, current_data)
                if current_data is None:
                    return None
            elif plugin_name in self.data_sources:
                # For data sources, merge their data with current data
                source_data = self.get_data_from_source(plugin_name)
                if source_data:
                    current_data.update(source_data)
        
        return current_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        with self._lock:
            return {
                **self.stats,
                'total_plugins': len(self.plugins),
                'running_plugins': len([p for p in self.plugins.values() if p.status == PluginStatus.RUNNING]),
                'error_plugins': len([p for p in self.plugins.values() if p.status == PluginStatus.ERROR]),
                'data_sources': len(self.data_sources),
                'data_processors': len(self.data_processors),
                'data_pipelines': len(self.data_pipelines)
            }
    
    def shutdown(self) -> None:
        """Shutdown plugin manager and all plugins."""
        logger.info("Shutting down plugin manager...")
        
        # Stop background tasks
        self._stop_background.set()
        if self._background_thread:
            self._background_thread.join(timeout=5.0)
        
        # Stop all plugins
        with self._lock:
            for plugin_name in list(self.plugins.keys()):
                self.stop_plugin(plugin_name)
        
        logger.info("Plugin manager shutdown complete")
