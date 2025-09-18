"""
Tests for the plugin_system module.

These tests verify plugin loading, lifecycle management, and
the plugin architecture without requiring actual hardware.
"""

import threading
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from capybarish.plugin_system import (
    DataProcessorPlugin,
    DataSourcePlugin,
    Plugin,
    PluginInfo,
    PluginManager,
    PluginMetadata,
    PluginStatus,
    PluginType,
)


class MockDataSourcePlugin(DataSourcePlugin):
    """Mock data source plugin for testing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data = {"test": "value", "counter": 0}

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockDataSource",
            version="1.0.0",
            description="Mock data source for testing",
            author="Test Author",
            plugin_type=PluginType.DATA_SOURCE,
            tags={"test", "mock"},
        )

    def initialize(self) -> bool:
        self.status = PluginStatus.INITIALIZED
        return True

    def start(self) -> bool:
        self.status = PluginStatus.RUNNING
        return True

    def stop(self) -> bool:
        self.status = PluginStatus.STOPPED
        return True

    def get_data(self) -> Dict[str, Any]:
        self.data["counter"] += 1
        return self.data.copy()

    def supports_streaming(self) -> bool:
        return True

    def start_streaming(self, callback) -> bool:
        self.stream_callback = callback
        return True

    def stop_streaming(self) -> bool:
        self.stream_callback = None
        return True


class MockDataProcessorPlugin(DataProcessorPlugin):
    """Mock data processor plugin for testing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.process_count = 0

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockDataProcessor",
            version="1.0.0",
            description="Mock data processor for testing",
            author="Test Author",
            plugin_type=PluginType.DATA_PROCESSOR,
            tags={"test", "mock"},
        )

    def initialize(self) -> bool:
        self.status = PluginStatus.INITIALIZED
        return True

    def start(self) -> bool:
        self.status = PluginStatus.RUNNING
        return True

    def stop(self) -> bool:
        self.status = PluginStatus.STOPPED
        return True

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.process_count += 1
        processed = data.copy()
        processed["processed"] = True
        processed["process_count"] = self.process_count
        return processed

    def supports_batch_processing(self) -> bool:
        return True


class FailingPlugin(Plugin):
    """Plugin that fails for testing error handling."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fail_on = config.get("fail_on", [])

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="FailingPlugin",
            version="1.0.0",
            description="Plugin that fails for testing",
            author="Test Author",
            plugin_type=PluginType.DATA_PROCESSOR,
            tags={"test", "fail"},
        )

    def initialize(self) -> bool:
        if "initialize" in self.fail_on:
            raise RuntimeError("Initialization failed")
        return True

    def start(self) -> bool:
        if "start" in self.fail_on:
            raise RuntimeError("Start failed")
        return True

    def stop(self) -> bool:
        if "stop" in self.fail_on:
            raise RuntimeError("Stop failed")
        return True


class TestPluginMetadata:
    """Test PluginMetadata class."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="Test description",
            author="Test Author",
            plugin_type=PluginType.DATA_SOURCE,
            dependencies=["dep1", "dep2"],
            tags={"tag1", "tag2"},
        )

        assert metadata.name == "TestPlugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test description"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == PluginType.DATA_SOURCE
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.tags == {"tag1", "tag2"}

    def test_plugin_metadata_defaults(self):
        """Test plugin metadata with default values."""
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="Test description",
            author="Test Author",
            plugin_type=PluginType.DATA_SOURCE,
        )

        assert metadata.dependencies == []
        assert metadata.tags == set()


class TestPluginBase:
    """Test base Plugin class functionality."""

    def test_mock_data_source_plugin(self):
        """Test mock data source plugin."""
        config = {"test_param": "value"}
        plugin = MockDataSourcePlugin(config)

        assert plugin.config == config
        assert plugin.status == PluginStatus.LOADED
        assert plugin.metadata.name == "MockDataSource"
        assert plugin.metadata.plugin_type == PluginType.DATA_SOURCE

        # Test lifecycle
        assert plugin.initialize() == True
        assert plugin.status == PluginStatus.INITIALIZED

        assert plugin.start() == True
        assert plugin.status == PluginStatus.RUNNING

        # Test data functionality
        data = plugin.get_data()
        assert data["test"] == "value"
        assert data["counter"] == 1

        # Test streaming
        assert plugin.supports_streaming() == True
        callback = Mock()
        assert plugin.start_streaming(callback) == True
        assert plugin.stop_streaming() == True

        assert plugin.stop() == True
        assert plugin.status == PluginStatus.STOPPED

    def test_mock_data_processor_plugin(self):
        """Test mock data processor plugin."""
        config = {}
        plugin = MockDataProcessorPlugin(config)

        assert plugin.metadata.name == "MockDataProcessor"
        assert plugin.metadata.plugin_type == PluginType.DATA_PROCESSOR

        # Test lifecycle
        assert plugin.initialize() == True
        assert plugin.start() == True

        # Test processing
        input_data = {"input": "test"}
        processed = plugin.process(input_data)
        assert processed["input"] == "test"
        assert processed["processed"] == True
        assert processed["process_count"] == 1

        # Test batch processing
        assert plugin.supports_batch_processing() == True
        batch_data = [{"a": 1}, {"b": 2}]
        batch_result = plugin.process_batch(batch_data)
        assert len(batch_result) == 2
        assert batch_result[0]["processed"] == True
        assert batch_result[1]["processed"] == True

        assert plugin.stop() == True

    def test_failing_plugin(self):
        """Test plugin that fails during lifecycle operations."""
        # Test initialization failure
        config = {"fail_on": ["initialize"]}
        plugin = FailingPlugin(config)

        with pytest.raises(RuntimeError, match="Initialization failed"):
            plugin.initialize()

        # Test start failure
        config = {"fail_on": ["start"]}
        plugin = FailingPlugin(config)
        plugin.initialize()

        with pytest.raises(RuntimeError, match="Start failed"):
            plugin.start()


class TestPluginManager:
    """Test PluginManager class."""

    def test_plugin_manager_creation(self):
        """Test creating a plugin manager."""
        manager = PluginManager()

        assert manager.plugin_directories == ["plugins"]
        assert manager.plugins == {}
        assert manager.plugin_types == {}
        assert manager.data_sources == {}
        assert manager.data_processors == {}
        assert len(manager.on_plugin_loaded) == 0

    def test_plugin_manager_custom_directories(self):
        """Test creating plugin manager with custom directories."""
        directories = ["custom_plugins", "more_plugins"]
        manager = PluginManager(plugin_directories=directories)

        assert manager.plugin_directories == directories

    @patch("capybarish.plugin_system.importlib.import_module")
    def test_load_plugin_success(self, mock_import):
        """Test successful plugin loading."""
        manager = PluginManager()

        # Mock module with plugin class
        mock_module = Mock()
        mock_module.MockDataSourcePlugin = MockDataSourcePlugin
        mock_import.return_value = mock_module

        # Add callback to test notification
        callback = Mock()
        manager.on_plugin_loaded.append(callback)

        config = {"test": "value"}
        result = manager.load_plugin("test_module", config)

        assert result == True
        assert "MockDataSource" in manager.plugins
        assert PluginType.DATA_SOURCE in manager.plugin_types
        assert "MockDataSource" in manager.plugin_types[PluginType.DATA_SOURCE]
        assert "MockDataSource" in manager.data_sources

        # Check callback was called
        callback.assert_called_once()

    @patch("capybarish.plugin_system.importlib.import_module")
    def test_load_plugin_no_plugin_class(self, mock_import):
        """Test loading module with no plugin classes."""
        manager = PluginManager()

        # Mock module without plugin classes
        mock_module = Mock()
        # Remove any potential plugin classes
        for attr in dir(mock_module):
            if not attr.startswith("_"):
                delattr(mock_module, attr)
        mock_import.return_value = mock_module

        result = manager.load_plugin("test_module")
        assert result == False

    @patch("capybarish.plugin_system.importlib.import_module")
    def test_load_plugin_import_error(self, mock_import):
        """Test plugin loading with import error."""
        manager = PluginManager()
        mock_import.side_effect = ImportError("Module not found")

        result = manager.load_plugin("nonexistent_module")
        assert result == False

    def test_plugin_lifecycle_management(self):
        """Test plugin lifecycle management."""
        manager = PluginManager()

        # Manually add a plugin for testing
        plugin = MockDataSourcePlugin({})
        plugin_info = PluginInfo(
            plugin=plugin, metadata=plugin.metadata, status=PluginStatus.LOADED
        )
        manager.plugins["MockDataSource"] = plugin_info
        manager.data_sources["MockDataSource"] = plugin

        # Test initialization
        result = manager.initialize_plugin("MockDataSource")
        assert result == True
        assert plugin_info.status == PluginStatus.INITIALIZED

        # Test starting
        start_callback = Mock()
        manager.on_plugin_started.append(start_callback)

        result = manager.start_plugin("MockDataSource")
        assert result == True
        assert plugin_info.status == PluginStatus.RUNNING
        start_callback.assert_called_once()

        # Test stopping
        stop_callback = Mock()
        manager.on_plugin_stopped.append(stop_callback)

        result = manager.stop_plugin("MockDataSource")
        assert result == True
        assert plugin_info.status == PluginStatus.STOPPED
        stop_callback.assert_called_once()

    def test_plugin_lifecycle_failure(self):
        """Test plugin lifecycle failure handling."""
        manager = PluginManager()

        # Add failing plugin
        plugin = FailingPlugin({"fail_on": ["initialize"]})
        plugin_info = PluginInfo(
            plugin=plugin, metadata=plugin.metadata, status=PluginStatus.LOADED
        )
        manager.plugins["FailingPlugin"] = plugin_info

        # Test initialization failure
        error_callback = Mock()
        manager.on_plugin_error.append(error_callback)

        result = manager.initialize_plugin("FailingPlugin")
        assert result == False
        assert plugin_info.status == PluginStatus.ERROR
        error_callback.assert_called_once()

    def test_nonexistent_plugin_operations(self):
        """Test operations on non-existent plugins."""
        manager = PluginManager()

        assert manager.initialize_plugin("NonExistent") == False
        assert manager.start_plugin("NonExistent") == False
        assert manager.stop_plugin("NonExistent") == False
        assert manager.get_plugin("NonExistent") is None

    def test_data_operations(self):
        """Test data source and processor operations."""
        manager = PluginManager()

        # Add plugins manually
        source_plugin = MockDataSourcePlugin({})
        processor_plugin = MockDataProcessorPlugin({})

        manager.data_sources["source"] = source_plugin
        manager.data_processors["processor"] = processor_plugin

        # Test data source
        data = manager.get_data_from_source("source")
        assert data is not None
        assert data["test"] == "value"
        assert data["counter"] == 1

        # Test data processor
        input_data = {"input": "test"}
        processed = manager.process_data("processor", input_data)
        assert processed is not None
        assert processed["processed"] == True

        # Test non-existent sources/processors
        assert manager.get_data_from_source("nonexistent") is None
        assert manager.process_data("nonexistent", {}) is None

    def test_data_pipeline(self):
        """Test data pipeline creation and execution."""
        manager = PluginManager()

        # Add plugins
        source_plugin = MockDataSourcePlugin({})
        processor_plugin = MockDataProcessorPlugin({})

        source_info = PluginInfo(
            plugin=source_plugin, metadata=source_plugin.metadata, status=PluginStatus.RUNNING
        )
        processor_info = PluginInfo(
            plugin=processor_plugin,
            metadata=processor_plugin.metadata,
            status=PluginStatus.RUNNING,
        )

        manager.plugins["source"] = source_info
        manager.plugins["processor"] = processor_info
        manager.data_sources["source"] = source_plugin
        manager.data_processors["processor"] = processor_plugin

        # Create pipeline
        pipeline = ["source", "processor"]
        result = manager.create_data_pipeline(pipeline)
        assert result == True

        # Execute pipeline
        initial_data = {"initial": "value"}
        result_data = manager.execute_pipeline(0, initial_data)
        assert result_data is not None
        assert result_data["initial"] == "value"
        assert result_data["test"] == "value"  # From source
        assert result_data["processed"] == True  # From processor

    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        manager = PluginManager()

        # Add plugins of different types
        manager.plugin_types[PluginType.DATA_SOURCE] = ["source1", "source2"]
        manager.plugin_types[PluginType.DATA_PROCESSOR] = ["processor1"]

        sources = manager.get_plugins_by_type(PluginType.DATA_SOURCE)
        assert sources == ["source1", "source2"]

        processors = manager.get_plugins_by_type(PluginType.DATA_PROCESSOR)
        assert processors == ["processor1"]

        filters = manager.get_plugins_by_type(PluginType.FILTER)
        assert filters == []

    def test_statistics(self):
        """Test plugin manager statistics."""
        manager = PluginManager()

        # Initial stats
        stats = manager.get_statistics()
        assert stats["total_plugins"] == 0
        assert stats["running_plugins"] == 0
        assert stats["plugins_loaded"] == 0

        # Add some plugins
        plugin1 = MockDataSourcePlugin({})
        plugin2 = MockDataProcessorPlugin({})

        plugin1_info = PluginInfo(
            plugin=plugin1, metadata=plugin1.metadata, status=PluginStatus.RUNNING
        )
        plugin2_info = PluginInfo(
            plugin=plugin2, metadata=plugin2.metadata, status=PluginStatus.ERROR
        )

        manager.plugins["plugin1"] = plugin1_info
        manager.plugins["plugin2"] = plugin2_info
        manager.data_sources["plugin1"] = plugin1
        manager.data_processors["plugin2"] = plugin2

        # Check updated stats
        stats = manager.get_statistics()
        assert stats["total_plugins"] == 2
        assert stats["running_plugins"] == 1
        assert stats["error_plugins"] == 1
        assert stats["data_sources"] == 1
        assert stats["data_processors"] == 1

    def test_discover_plugins(self, temp_dir):
        """Test plugin discovery."""
        # Create fake plugin directory structure
        plugins_dir = temp_dir / "plugins"
        plugins_dir.mkdir()

        # Create some fake plugin files
        (plugins_dir / "plugin1.py").touch()
        (plugins_dir / "plugin2.py").touch()
        (plugins_dir / "__init__.py").touch()  # Should be ignored
        (plugins_dir / "__pycache__").mkdir()  # Should be ignored

        manager = PluginManager(plugin_directories=[str(plugins_dir)])

        discovered = manager.discover_plugins()
        expected = [f"{plugins_dir.name}.plugin1", f"{plugins_dir.name}.plugin2"]

        # Check that plugins were discovered (order may vary)
        assert len(discovered) == 2
        for expected_plugin in expected:
            assert any(expected_plugin in plugin for plugin in discovered)

    def test_shutdown(self):
        """Test plugin manager shutdown."""
        manager = PluginManager()

        # Add running plugin
        plugin = MockDataSourcePlugin({})
        plugin.status = PluginStatus.RUNNING
        plugin_info = PluginInfo(
            plugin=plugin, metadata=plugin.metadata, status=PluginStatus.RUNNING
        )
        manager.plugins["test"] = plugin_info

        # Test shutdown
        manager.shutdown()

        # Plugin should be stopped
        assert plugin.status == PluginStatus.STOPPED


class TestPluginSystemIntegration:
    """Integration tests for the plugin system."""

    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle from loading to shutdown."""
        manager = PluginManager()

        # Create and add plugin manually (simulating loading)
        plugin = MockDataSourcePlugin({"param": "value"})
        plugin_info = PluginInfo(
            plugin=plugin, metadata=plugin.metadata, status=PluginStatus.LOADED
        )
        manager.plugins["TestPlugin"] = plugin_info
        manager.data_sources["TestPlugin"] = plugin

        # Initialize
        assert manager.initialize_plugin("TestPlugin") == True

        # Start
        assert manager.start_plugin("TestPlugin") == True

        # Use plugin
        data = manager.get_data_from_source("TestPlugin")
        assert data is not None
        assert data["test"] == "value"

        # Stop
        assert manager.stop_plugin("TestPlugin") == True

        # Shutdown manager
        manager.shutdown()

    def test_data_processing_pipeline(self):
        """Test data flowing through a processing pipeline."""
        manager = PluginManager()

        # Create plugins
        source = MockDataSourcePlugin({})
        processor1 = MockDataProcessorPlugin({})
        processor2 = MockDataProcessorPlugin({})

        # Add to manager
        manager.plugins["source"] = PluginInfo(
            plugin=source, metadata=source.metadata, status=PluginStatus.RUNNING
        )
        manager.plugins["proc1"] = PluginInfo(
            plugin=processor1, metadata=processor1.metadata, status=PluginStatus.RUNNING
        )
        manager.plugins["proc2"] = PluginInfo(
            plugin=processor2, metadata=processor2.metadata, status=PluginStatus.RUNNING
        )

        manager.data_sources["source"] = source
        manager.data_processors["proc1"] = processor1
        manager.data_processors["proc2"] = processor2

        # Create pipeline
        pipeline = ["source", "proc1", "proc2"]
        assert manager.create_data_pipeline(pipeline) == True

        # Execute pipeline
        initial_data = {"start": True}
        result = manager.execute_pipeline(0, initial_data)

        assert result is not None
        assert result["start"] == True  # Original data
        assert result["test"] == "value"  # From source
        assert result["processed"] == True  # From processors
        assert result["process_count"] == 1  # From last processor