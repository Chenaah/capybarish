"""
Integration tests for capybarish package.

These tests verify component interactions and end-to-end functionality
without requiring actual hardware. They test how different modules
work together in realistic scenarios.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from capybarish.config_manager import create_robot_config_manager
from capybarish.plugin_system import PluginManager
from capybarish.utils import convert_np_arrays_to_lists, load_cfg
from plugins.imu_processor import IMUProcessor
from plugins.optitrack_source import OptiTrackSource
from tests.conftest import MockNatNetClient, create_mock_robot_modules


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_config_plugin_integration(self, temp_dir):
        """Test configuration loading and plugin system integration."""
        # Create config directory
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create configuration file
        config_data = {
            "interface": {
                "module_ids": [1, 2, 3],
                "torso_module_id": 1,
                "sources": ["imu", "optitrack"],
                "struct_format": "test_format",
                "protocol": "UDP",
                "dashboard": True,
                "optitrack_rigibody": 1,
                "enable_filter": True,
                "kp_ratio": 1.0,
                "kd_ratio": 1.0,
                "calibration_modes": None,
                "broken_motors": None,
                "check_action_safety": True,
            },
            "robot": {"dt": 0.02, "motor_range": [[-3.14, 3.14]] * 3},
            "agent": {"filter_action": True},
            "logging": {"robot_data_dir": None},
            "plugins": {
                "imu_processor": {"filter_alpha": 0.2, "calibration_samples": 50},
                "optitrack_source": {
                    "server_address": "192.168.1.100",
                    "rigid_body_id": 1,
                },
            },
        }

        config_file = config_dir / "integration_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        config_manager = create_robot_config_manager(
            config_dir=config_dir, config_name="integration_test"
        )
        config = config_manager.load()

        # Verify configuration
        assert config["interface"]["module_ids"] == [1, 2, 3]
        assert config["plugins"]["imu_processor"]["filter_alpha"] == 0.2

        # Test plugin configuration
        if "plugins" in config:
            imu_config = config["plugins"]["imu_processor"]
            processor = IMUProcessor(imu_config)
            assert processor.filter_alpha == 0.2
            assert processor.calibration_samples == 50

    def test_config_environment_override_integration(self, temp_dir):
        """Test environment-specific configuration overrides."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Base configuration
        base_config = {
            "interface": {
                "module_ids": [1, 2],
                "protocol": "UDP",
                "struct_format": "base_format",
                "sources": ["imu"],
                "torso_module_id": 1,
                "dashboard": False,
                "check_action_safety": True,
                "enable_filter": True,
                "kp_ratio": 1.0,
                "kd_ratio": 1.0,
                "calibration_modes": None,
                "broken_motors": None,
            },
            "robot": {"dt": 0.02, "motor_range": [[-3.14, 3.14]] * 2},
            "agent": {"filter_action": True},
            "logging": {"robot_data_dir": None},
        }

        # Production environment override
        prod_config = {
            "interface": {"module_ids": [1, 2, 3, 4], "dashboard": True},
            "robot": {"dt": 0.01},  # Higher frequency in production
        }

        # Save configs
        with open(config_dir / "app.yaml", "w") as f:
            yaml.dump(base_config, f)
        with open(config_dir / "app.production.yaml", "w") as f:
            yaml.dump(prod_config, f)

        # Load production config
        config_manager = create_robot_config_manager(
            config_dir=config_dir, config_name="app", environment="production"
        )
        config = config_manager.load()

        # Verify overrides were applied
        assert config["interface"]["module_ids"] == [1, 2, 3, 4]  # From override
        assert config["robot"]["dt"] == 0.01  # From override
        assert config["interface"]["protocol"] == "UDP"  # From base


class TestPluginSystemIntegration:
    """Test plugin system integration with data flow."""

    def test_imu_optitrack_plugin_chain(self):
        """Test IMU and OptiTrack plugins working together."""
        plugin_manager = PluginManager()

        # Create and add IMU processor
        imu_config = {"filter_alpha": 0.3, "calibration_samples": 3}
        imu_processor = IMUProcessor(imu_config)
        
        # Initialize and start IMU processor
        assert imu_processor.initialize() == True
        assert imu_processor.start() == True

        # Create and add OptiTrack source
        opti_config = {"server_address": "192.168.1.100", "rigid_body_id": 1}
        with patch("capybarish.natnet.NatNetClient.NatNetClient") as mock_natnet:
            mock_client = MockNatNetClient()
            mock_client.run = Mock(return_value=True)
            mock_natnet.return_value = mock_client

            opti_source = OptiTrackSource(opti_config)
            assert opti_source.initialize() == True
            assert opti_source.start() == True

        # Add plugins to manager
        plugin_manager.data_processors["imu_processor"] = imu_processor
        plugin_manager.data_sources["optitrack_source"] = opti_source

        # Simulate data flow
        # 1. Generate raw sensor data
        raw_data = {
            "acc_body_imu": [0.1, 0.2, 9.8],
            "body_omega_imu": [0.01, 0.02, 0.03],
            "timestamp": time.time(),
        }

        # 2. Process through IMU processor
        processed_imu = plugin_manager.process_data("imu_processor", raw_data)
        assert processed_imu is not None
        assert "imu_acc_filtered" in processed_imu
        assert "imu_orientation_quat" in processed_imu

        # 3. Simulate OptiTrack data
        opti_source._on_rigid_body_frame(1, [1.5, 2.5, 0.8], [0.1, 0.2, 0.3, 0.9])
        opti_data = plugin_manager.get_data_from_source("optitrack_source")
        assert opti_data is not None
        assert "pos_world_opti" in opti_data

        # 4. Combine data sources
        combined_data = {**processed_imu, **opti_data}
        assert "imu_acc_filtered" in combined_data
        assert "pos_world_opti" in combined_data

        # Stop plugins
        assert imu_processor.stop() == True
        assert opti_source.stop() == True

    def test_data_pipeline_execution(self):
        """Test executing a data processing pipeline."""
        plugin_manager = PluginManager()

        # Create multiple processors
        processor1 = IMUProcessor({"filter_alpha": 0.5})
        processor2 = IMUProcessor({"filter_alpha": 0.3})  # Different config

        processor1.initialize()
        processor1.start()
        processor2.initialize()  
        processor2.start()

        # Register plugins properly (need to add them to self.plugins as well)
        plugin_manager.data_processors["stage1"] = processor1
        plugin_manager.data_processors["stage2"] = processor2
        
        # Manually add to plugins dict for pipeline creation
        from capybarish.plugin_system import PluginInfo, PluginStatus
        plugin_manager.plugins["stage1"] = PluginInfo(
            plugin=processor1, 
            metadata=processor1.metadata,
            status=PluginStatus.RUNNING
        )
        plugin_manager.plugins["stage2"] = PluginInfo(
            plugin=processor2,
            metadata=processor2.metadata, 
            status=PluginStatus.RUNNING
        )

        # Create processing pipeline
        pipeline = ["stage1", "stage2"]
        assert plugin_manager.create_data_pipeline(pipeline) == True

        # Execute pipeline
        initial_data = {
            "acc_body_imu": [1.0, 0.5, 9.5],
            "body_omega_imu": [0.1, 0.05, 0.02],
        }

        result = plugin_manager.execute_pipeline(0, initial_data)
        assert result is not None

        # Should have data from both processing stages
        assert "imu_acc_filtered" in result
        assert "imu_processor_stats" in result

        # Process count should be from the last processor
        stats = result["imu_processor_stats"]
        assert stats["samples_processed"] > 0


class TestDataFlowIntegration:
    """Test realistic data flow scenarios."""

    def test_robot_sensor_data_processing(self):
        """Test processing data from multiple robot modules."""
        # Create mock robot module data
        module_data = create_mock_robot_modules([1, 2, 3])

        # Create IMU processor for sensor data
        imu_processor = IMUProcessor({"filter_alpha": 0.2})
        imu_processor.initialize()
        imu_processor.start()

        # Process data from each module
        processed_modules = {}
        for module_id, data in module_data.items():
            processed = imu_processor.process(data)
            processed_modules[module_id] = processed

        # Verify processing
        assert len(processed_modules) == 3

        for module_id, processed in processed_modules.items():
            # Should contain original data
            assert processed["module_id"] == module_id
            assert "motor_pos" in processed

            # Should contain processed IMU data
            assert "imu_acc_filtered" in processed
            assert "imu_gyro_filtered" in processed

            # Processed data should be reasonable
            acc_filtered = processed["imu_acc_filtered"]
            assert len(acc_filtered) == 3
            assert all(isinstance(x, (int, float)) for x in acc_filtered)

    def test_multi_source_data_fusion(self):
        """Test fusing data from multiple sources."""
        # Create processors and sources
        imu_processor = IMUProcessor({"filter_alpha": 0.1})
        imu_processor.initialize()
        imu_processor.start()

        with patch("capybarish.natnet.NatNetClient.NatNetClient") as mock_natnet:
            mock_client = MockNatNetClient()
            mock_client.run = Mock(return_value=True)
            mock_natnet.return_value = mock_client

            opti_source = OptiTrackSource({"rigid_body_id": 1})
            opti_source.initialize()
            opti_source.start()

            # Simulate sensor data sequence
            sensor_sequence = [
                {
                    "acc_body_imu": [0.1, 0.2, 9.8],
                    "body_omega_imu": [0.01, 0.02, 0.03],
                    "timestamp": time.time(),
                },
                {
                    "acc_body_imu": [0.2, 0.3, 9.7],
                    "body_omega_imu": [0.02, 0.03, 0.04],
                    "timestamp": time.time() + 0.02,
                },
                {
                    "acc_body_imu": [0.3, 0.1, 9.9],
                    "body_omega_imu": [0.01, 0.01, 0.02],
                    "timestamp": time.time() + 0.04,
                },
            ]

            # Process sequence
            fused_data = []
            for i, sensor_data in enumerate(sensor_sequence):
                # Process IMU data
                processed_imu = imu_processor.process(sensor_data)

                # Simulate corresponding OptiTrack data
                pos = [0.1 * i, 0.2 * i, 0.5 + 0.1 * i]
                rot = [0, 0, 0.1 * i, 1]
                opti_source._on_rigid_body_frame(1, pos, rot)
                opti_data = opti_source.get_data()

                # Fuse data
                fused = {
                    "timestamp": sensor_data["timestamp"],
                    "imu_data": {
                        "acc_filtered": processed_imu["imu_acc_filtered"],
                        "gyro_filtered": processed_imu["imu_gyro_filtered"],
                        "orientation": processed_imu["imu_orientation_quat"],
                    },
                    "optitrack_data": {
                        "position": opti_data.get("pos_world_opti", [0, 0, 0]),
                        "rotation": opti_data.get("quat_world_opti", [0, 0, 0, 1]),
                        "tracking_valid": opti_data.get("optitrack_tracking_valid", False),
                    },
                }

                fused_data.append(fused)

            # Verify fused data
            assert len(fused_data) == len(sensor_sequence)

            for i, data in enumerate(fused_data):
                # Should have both IMU and OptiTrack data
                assert "imu_data" in data
                assert "optitrack_data" in data

                # OptiTrack position should progress
                assert data["optitrack_data"]["position"][0] == 0.1 * i
                assert data["optitrack_data"]["tracking_valid"] == True

                # IMU data should be filtered
                assert len(data["imu_data"]["acc_filtered"]) == 3


class TestUtilsIntegration:
    """Test utility function integration."""

    def test_data_conversion_integration(self):
        """Test data conversion with realistic robot data."""
        # Create data with numpy arrays (common in robot systems)
        robot_data = {
            "positions": np.array([0.1, 0.2, 0.3]),
            "velocities": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "imu_data": {
                "acceleration": np.array([0.5, 0.6, 9.8]),
                "angular_velocity": np.array([0.01, 0.02, 0.03]),
                "quaternion": np.array([0, 0, 0, 1]),
            },
            "module_data": [
                {"id": 1, "sensor_reading": np.array([1.1, 2.2])},
                {"id": 2, "sensor_reading": np.array([3.3, 4.4])},
            ],
            "timestamp": time.time(),
        }

        # Convert for serialization (e.g., for logging or dashboard)
        converted = convert_np_arrays_to_lists(robot_data)

        # Verify conversion
        assert converted["positions"] == [0.1, 0.2, 0.3]
        assert converted["velocities"] == [[1.0, 2.0], [3.0, 4.0]]
        assert np.array_equal(converted["imu_data"]["acceleration"], [0.5, 0.6, 9.8])

        # Nested objects should remain unchanged (function doesn't recurse)
        assert isinstance(converted["imu_data"]["quaternion"], np.ndarray)

        # Module data should have top-level arrays converted
        for module in converted["module_data"]:
            assert isinstance(module["sensor_reading"], np.ndarray)  # Not converted (nested)

    def test_config_plugin_data_flow(self, temp_dir):
        """Test complete data flow from config to plugin processing."""
        # Create configuration
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        config_data = {
            "interface": {
                "module_ids": [1, 2],
                "sources": ["imu"],
                "struct_format": "test_format",
                "protocol": "UDP",
                "torso_module_id": 1,
                "dashboard": False,
                "check_action_safety": True,
                "enable_filter": True,
                "kp_ratio": 1.0,
                "kd_ratio": 1.0,
                "calibration_modes": None,
                "broken_motors": None,
            },
            "robot": {"dt": 0.02, "motor_range": [[-3.14, 3.14]] * 2},
            "agent": {"filter_action": True},
            "logging": {"robot_data_dir": None},
            "plugins": {
                "imu_processor": {
                    "filter_alpha": 0.25,
                    "calibration_samples": 10,
                    "gravity_threshold": 0.8,
                }
            },
        }

        config_file = config_dir / "flow_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load configuration
        with patch("capybarish.utils.os.path.exists", return_value=True):
            with patch("capybarish.utils.OmegaConf.load") as mock_load:
                mock_load.return_value = type("Config", (), config_data)()
                
                # For this test, we'll create the plugin directly
                plugin_config = config_data["plugins"]["imu_processor"]
                processor = IMUProcessor(plugin_config)

                # Verify configuration was applied
                assert processor.filter_alpha == 0.25
                assert processor.calibration_samples == 10
                assert processor.gravity_threshold == 0.8

                # Test processing with configured plugin
                processor.initialize()
                processor.start()

                test_data = {
                    "acc_body_imu": [0.2, 0.3, 9.7],
                    "body_omega_imu": [0.05, 0.03, 0.01],
                }

                result = processor.process(test_data)
                assert "imu_acc_filtered" in result
                assert "imu_processor_stats" in result

                # Stop plugin
                processor.stop()


class TestErrorHandlingIntegration:
    """Test error handling across system components."""

    def test_plugin_failure_recovery(self):
        """Test system behavior when plugins fail."""
        plugin_manager = PluginManager()

        # Create a processor that will fail
        class FailingProcessor(IMUProcessor):
            def process(self, data):
                raise RuntimeError("Processing failed")

        failing_processor = FailingProcessor({})
        failing_processor.initialize()
        failing_processor.start()

        plugin_manager.data_processors["failing"] = failing_processor

        # Try to process data (should handle error gracefully)
        result = plugin_manager.process_data("failing", {"test": "data"})
        assert result is None  # Should return None on error

        # Statistics should track the error
        stats = plugin_manager.get_statistics()
        assert stats["errors"] > 0

    def test_configuration_validation_integration(self, temp_dir):
        """Test configuration validation in realistic scenarios."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create invalid configuration
        invalid_config = {
            "interface": {
                "module_ids": "not_a_list",  # Invalid type
                "protocol": "INVALID_PROTOCOL",  # Invalid value
                # Missing required struct_format
            },
            "robot": {
                "dt": -0.01,  # Invalid value (negative)
            },
            # Missing required logging section
        }

        config_file = config_dir / "invalid.yaml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Try to load invalid configuration
        config_manager = create_robot_config_manager(
            config_dir=config_dir, config_name="invalid"
        )

        with pytest.raises(ValueError, match="Configuration validation failed"):
            config_manager.load()

    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies."""
        # Test OptiTrack source without NatNet client
        with patch("capybarish.natnet.NatNetClient.NatNetClient", side_effect=ImportError("Module not found")):
            source = OptiTrackSource({})
            
            # Should fail gracefully
            assert source.initialize() == False
            assert source.last_error is not None
            assert "not available" in source.last_error


class TestPerformanceIntegration:
    """Test performance aspects of component integration."""

    def test_high_frequency_data_processing(self):
        """Test processing data at high frequencies."""
        processor = IMUProcessor({"filter_alpha": 0.1})
        processor.initialize()
        processor.start()

        # Generate high-frequency data sequence
        data_sequence = []
        base_time = time.time()
        for i in range(100):  # 100 samples
            data = {
                "acc_body_imu": [0.1 + 0.01 * i, 0.2, 9.8],
                "body_omega_imu": [0.01, 0.02 + 0.001 * i, 0.03],
                "timestamp": base_time + i * 0.001,  # 1kHz data
            }
            data_sequence.append(data)

        # Process all data
        start_time = time.time()
        results = []
        for data in data_sequence:
            result = processor.process(data)
            results.append(result)
        processing_time = time.time() - start_time

        # Verify processing
        assert len(results) == 100
        assert processing_time < 1.0  # Should process quickly

        # Check that filtering is working
        first_result = results[0]["imu_acc_filtered"]
        last_result = results[-1]["imu_acc_filtered"]
        assert first_result != last_result  # Should show filtering progression

    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets."""
        processor = IMUProcessor({})
        processor.initialize()
        processor.start()

        # Process large batch
        large_batch = []
        for i in range(1000):
            data = {
                "acc_body_imu": [i * 0.001, 0.2, 9.8],
                "body_omega_imu": [0.01, i * 0.0001, 0.03],
            }
            large_batch.append(data)

        # Process batch
        batch_results = processor.process_batch(large_batch)

        # Verify results
        assert len(batch_results) == 1000
        assert all("imu_acc_filtered" in result for result in batch_results)

        # Memory should not grow unbounded (basic check)
        # In a real system, you'd monitor actual memory usage
        assert processor.samples_processed == 1000


@pytest.mark.integration
class TestFullSystemIntegration:
    """Test full system integration scenarios."""

    def test_complete_robot_data_pipeline(self, temp_dir):
        """Test complete robot data processing pipeline."""
        # This test simulates a complete robot system without hardware
        
        # 1. Setup configuration
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        system_config = {
            "interface": {
                "module_ids": [1, 2, 3],
                "torso_module_id": 1,
                "sources": ["imu", "optitrack"],
                "struct_format": "complete_test",
                "protocol": "UDP",
                "dashboard": False,
                "check_action_safety": True,
                "enable_filter": True,
                "kp_ratio": 1.0,
                "kd_ratio": 1.0,
                "calibration_modes": None,
                "broken_motors": None,
                "optitrack_rigibody": 1,
            },
            "robot": {"dt": 0.02, "motor_range": [[-3.14, 3.14]] * 3},
            "agent": {"filter_action": True},
            "logging": {"robot_data_dir": str(temp_dir / "logs")},
        }

        config_file = config_dir / "system.yaml"
        with open(config_file, "w") as f:
            yaml.dump(system_config, f)

        # 2. Load configuration
        config_manager = create_robot_config_manager(
            config_dir=config_dir, config_name="system"
        )
        config = config_manager.load()

        # 3. Setup plugin system
        plugin_manager = PluginManager()

        # Create and setup IMU processor
        imu_processor = IMUProcessor({"filter_alpha": 0.2})
        imu_processor.initialize()
        imu_processor.start()
        plugin_manager.data_processors["imu"] = imu_processor

        # Create and setup OptiTrack source
        with patch("capybarish.natnet.NatNetClient.NatNetClient") as mock_natnet:
            mock_client = MockNatNetClient()
            mock_client.run = Mock(return_value=True)
            mock_natnet.return_value = mock_client

            opti_source = OptiTrackSource({"rigid_body_id": 1})
            opti_source.initialize()
            opti_source.start()
            plugin_manager.data_sources["optitrack"] = opti_source

            # 4. Simulate robot operation cycle
            operation_results = []
            
            for cycle in range(10):  # 10 control cycles
                cycle_time = time.time()
                
                # Generate mock robot module data
                module_data = create_mock_robot_modules(config["interface"]["module_ids"])
                
                # Process each module's data
                processed_modules = {}
                for module_id, data in module_data.items():
                    # Add cycle-specific variations
                    data["acc_body_imu"][0] += cycle * 0.01
                    data["body_omega_imu"][1] += cycle * 0.001
                    
                    # Process through IMU processor
                    processed = plugin_manager.process_data("imu", data)
                    processed_modules[module_id] = processed

                # Get OptiTrack data
                opti_source._on_rigid_body_frame(
                    1, 
                    [cycle * 0.1, cycle * 0.05, 0.5], 
                    [0, 0, cycle * 0.01, 1]
                )
                opti_data = plugin_manager.get_data_from_source("optitrack")

                # Combine all data for this cycle
                cycle_data = {
                    "cycle": cycle,
                    "timestamp": cycle_time,
                    "modules": processed_modules,
                    "optitrack": opti_data,
                    "config": {
                        "dt": config["robot"]["dt"],
                        "module_count": len(config["interface"]["module_ids"]),
                    }
                }

                # Convert for serialization (simulating logging/dashboard)
                cycle_data_serializable = convert_np_arrays_to_lists(cycle_data)
                operation_results.append(cycle_data_serializable)

            # 5. Verify complete system operation
            assert len(operation_results) == 10

            for i, result in enumerate(operation_results):
                # Should have all expected components
                assert "modules" in result
                assert "optitrack" in result
                assert "config" in result

                # Module data should be processed
                assert len(result["modules"]) == 3
                for module_id, module_result in result["modules"].items():
                    assert "imu_acc_filtered" in module_result
                    assert "imu_processor_stats" in module_result

                # OptiTrack data should be present
                assert "pos_world_opti" in result["optitrack"]
                assert result["optitrack"]["pos_world_opti"][0] == i * 0.1  # Position progression

            # 6. Cleanup
            imu_processor.stop()
            opti_source.stop()
            plugin_manager.shutdown()

            # Test completed successfully - demonstrates full integration
            # without requiring actual hardware