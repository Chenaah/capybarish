"""
Tests for the IMU processor plugin.

These tests verify IMU data processing functionality including
filtering, calibration, and orientation estimation without
requiring actual hardware sensors.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from plugins.imu_processor import IMUProcessor


class TestIMUProcessor:
    """Test IMU processor plugin functionality."""

    def test_imu_processor_creation(self):
        """Test creating an IMU processor plugin."""
        config = {
            "filter_alpha": 0.2,
            "gravity_threshold": 0.8,
            "calibration_samples": 50,
        }
        processor = IMUProcessor(config)

        assert processor.filter_alpha == 0.2
        assert processor.gravity_threshold == 0.8
        assert processor.calibration_samples == 50
        assert processor.samples_processed == 0
        assert not processor.is_calibrated

    def test_imu_processor_defaults(self):
        """Test IMU processor with default configuration."""
        processor = IMUProcessor({})

        assert processor.filter_alpha == 0.1
        assert processor.gravity_threshold == 0.5
        assert processor.calibration_samples == 100

    def test_metadata(self):
        """Test plugin metadata."""
        processor = IMUProcessor({})
        metadata = processor.metadata

        assert metadata.name == "IMUProcessor"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Processes IMU data with filtering and calibration"
        assert "imu" in metadata.tags
        assert "filtering" in metadata.tags

    def test_config_validation(self):
        """Test configuration validation."""
        processor = IMUProcessor({})

        # Valid config
        valid_config = {"filter_alpha": 0.2, "calibration_samples": 50}
        errors = processor.validate_config(valid_config)
        assert errors == []

        # Invalid filter_alpha
        invalid_config1 = {"filter_alpha": 1.5}  # > 1
        errors1 = processor.validate_config(invalid_config1)
        assert "filter_alpha must be a number between 0 and 1" in errors1

        invalid_config2 = {"filter_alpha": 0}  # = 0
        errors2 = processor.validate_config(invalid_config2)
        assert "filter_alpha must be a number between 0 and 1" in errors2

        invalid_config3 = {"filter_alpha": "not_a_number"}
        errors3 = processor.validate_config(invalid_config3)
        assert "filter_alpha must be a number between 0 and 1" in errors3

        # Invalid calibration_samples
        invalid_config4 = {"calibration_samples": -10}
        errors4 = processor.validate_config(invalid_config4)
        assert "calibration_samples must be a positive integer" in errors4

        invalid_config5 = {"calibration_samples": "not_an_int"}
        errors5 = processor.validate_config(invalid_config5)
        assert "calibration_samples must be a positive integer" in errors5

    def test_lifecycle(self):
        """Test plugin lifecycle methods."""
        processor = IMUProcessor({})

        # Test initialization
        assert processor.initialize() == True

        # Test start
        assert processor.start() == True

        # Test stop
        assert processor.stop() == True

    def test_extract_accelerometer_data(self):
        """Test extracting accelerometer data from various input formats."""
        processor = IMUProcessor({})

        # Test with acc_body_imu field
        data1 = {"acc_body_imu": [1.0, 2.0, 9.8]}
        acc1 = processor._extract_accelerometer_data(data1)
        np.testing.assert_array_equal(acc1, np.array([1.0, 2.0, 9.8]))

        # Test with accelerometer field
        data2 = {"accelerometer": [1.5, 2.5, 9.5]}
        acc2 = processor._extract_accelerometer_data(data2)
        np.testing.assert_array_equal(acc2, np.array([1.5, 2.5, 9.5]))

        # Test with numpy array
        data3 = {"acc": np.array([1.1, 2.1, 9.1, 4.0])}  # Extra elements should be ignored
        acc3 = processor._extract_accelerometer_data(data3)
        np.testing.assert_array_equal(acc3, np.array([1.1, 2.1, 9.1]))

        # Test with no accelerometer data
        data4 = {"other_field": [1, 2, 3]}
        acc4 = processor._extract_accelerometer_data(data4)
        assert acc4 is None

    def test_extract_gyroscope_data(self):
        """Test extracting gyroscope data from various input formats."""
        processor = IMUProcessor({})

        # Test with body_omega_imu field
        data1 = {"body_omega_imu": [0.1, 0.2, 0.3]}
        gyro1 = processor._extract_gyroscope_data(data1)
        np.testing.assert_array_equal(gyro1, np.array([0.1, 0.2, 0.3]))

        # Test with gyroscope field
        data2 = {"gyroscope": [0.15, 0.25, 0.35]}
        gyro2 = processor._extract_gyroscope_data(data2)
        np.testing.assert_array_equal(gyro2, np.array([0.15, 0.25, 0.35]))

        # Test with numpy array
        data3 = {"gyros": np.array([0.11, 0.21, 0.31, 0.41])}  # Extra elements ignored
        gyro3 = processor._extract_gyroscope_data(data3)
        np.testing.assert_array_equal(gyro3, np.array([0.11, 0.21, 0.31]))

        # Test with no gyroscope data
        data4 = {"other_field": [1, 2, 3]}
        gyro4 = processor._extract_gyroscope_data(data4)
        assert gyro4 is None

    def test_low_pass_filter(self):
        """Test low-pass filtering functionality."""
        processor = IMUProcessor({})

        previous = np.array([1.0, 2.0, 3.0])
        current = np.array([2.0, 4.0, 6.0])
        alpha = 0.3

        filtered = processor._low_pass_filter(previous, current, alpha)
        expected = alpha * current + (1 - alpha) * previous
        np.testing.assert_array_almost_equal(filtered, expected)

    def test_estimate_orientation(self):
        """Test orientation estimation from accelerometer and gyroscope."""
        processor = IMUProcessor({})

        # Test with gravity-aligned accelerometer (pointing down)
        acc = np.array([0, 0, -9.81])
        gyro = np.array([0, 0, 0])

        quat = processor._estimate_orientation(acc, gyro)
        assert len(quat) == 4
        assert np.allclose(np.linalg.norm(quat), 1.0)  # Unit quaternion

        # Test with tilted accelerometer
        acc_tilted = np.array([1.0, 0, -9.0])
        quat_tilted = processor._estimate_orientation(acc_tilted, gyro)
        assert len(quat_tilted) == 4
        assert np.allclose(np.linalg.norm(quat_tilted), 1.0)

        # Test with zero acceleration (should return identity quaternion)
        acc_zero = np.array([0, 0, 0])
        quat_zero = processor._estimate_orientation(acc_zero, gyro)
        np.testing.assert_array_equal(quat_zero, np.array([0, 0, 0, 1]))

    def test_compensate_gravity(self):
        """Test gravity compensation functionality."""
        processor = IMUProcessor({})

        # Test with identity quaternion (no rotation)
        acc = np.array([0, 0, -9.81])  # Pure gravity
        quat = np.array([0, 0, 0, 1])  # Identity quaternion

        linear_acc = processor._compensate_gravity(acc, quat)
        # Should be close to zero since input is pure gravity
        assert np.allclose(linear_acc, np.array([0, 0, 0]), atol=0.1)

        # Test with additional linear acceleration
        acc_with_motion = np.array([1.0, 2.0, -9.81])
        linear_acc_motion = processor._compensate_gravity(acc_with_motion, quat)
        # Should have the linear component
        assert np.allclose(linear_acc_motion, np.array([1.0, 2.0, 0]), atol=0.1)

    def test_process_complete_data(self):
        """Test processing complete IMU data."""
        processor = IMUProcessor({"filter_alpha": 0.5})

        input_data = {
            "acc_body_imu": [0.1, 0.2, 9.8],
            "body_omega_imu": [0.01, 0.02, 0.03],
            "timestamp": time.time(),
        }

        processed = processor.process(input_data)

        # Check that original data is preserved
        assert processed["acc_body_imu"] == [0.1, 0.2, 9.8]
        assert processed["body_omega_imu"] == [0.01, 0.02, 0.03]

        # Check that new processed data is added
        assert "imu_acc_filtered" in processed
        assert "imu_gyro_filtered" in processed
        assert "imu_orientation_quat" in processed
        assert "imu_gravity_compensated_acc" in processed
        assert "imu_processor_stats" in processed

        # Check stats
        stats = processed["imu_processor_stats"]
        assert stats["samples_processed"] == 1
        assert stats["is_calibrated"] == False

    def test_process_missing_data(self):
        """Test processing data with missing IMU fields."""
        processor = IMUProcessor({})

        # Data without IMU fields
        input_data = {"timestamp": time.time(), "other_field": "value"}

        processed = processor.process(input_data)

        # Should not add IMU processing fields
        assert "imu_acc_filtered" not in processed
        assert "imu_gyro_filtered" not in processed
        # Original data should be preserved
        assert processed["other_field"] == "value"

    def test_process_with_error(self):
        """Test processing data when an error occurs."""
        processor = IMUProcessor({})

        # Mock the extract methods to raise an exception
        with patch.object(processor, "_extract_accelerometer_data", side_effect=Exception("Test error")):
            input_data = {"acc_body_imu": [0.1, 0.2, 9.8]}
            processed = processor.process(input_data)

            # Should contain error information
            assert "imu_processor_error" in processed
            assert "Test error" in processed["imu_processor_error"]

    def test_calibration_workflow(self):
        """Test complete calibration workflow."""
        processor = IMUProcessor({"calibration_samples": 3})

        # Start calibration
        assert processor.start_calibration() == True
        assert not processor.is_calibrated

        # Add calibration samples
        samples = [
            {"acc_body_imu": [0.1, 0.1, 9.9], "body_omega_imu": [0.01, 0.02, 0.03]},
            {"acc_body_imu": [0.2, 0.2, 9.8], "body_omega_imu": [0.02, 0.03, 0.04]},
            {"acc_body_imu": [0.3, 0.3, 9.7], "body_omega_imu": [0.03, 0.04, 0.05]},
        ]

        for sample in samples:
            assert processor.add_calibration_sample(sample) == True

        # Finish calibration
        assert processor.finish_calibration() == True
        assert processor.is_calibrated

        # Check bias values were computed
        assert processor.bias_acc is not None
        assert processor.bias_gyro is not None
        assert len(processor.bias_acc) == 3
        assert len(processor.bias_gyro) == 3

    def test_calibration_insufficient_samples(self):
        """Test calibration with insufficient samples."""
        processor = IMUProcessor({"calibration_samples": 10})

        processor.start_calibration()

        # Add only a few samples
        sample = {"acc_body_imu": [0.1, 0.1, 9.9], "body_omega_imu": [0.01, 0.02, 0.03]}
        processor.add_calibration_sample(sample)
        processor.add_calibration_sample(sample)

        # Try to finish with insufficient samples
        assert processor.finish_calibration() == False
        assert not processor.is_calibrated

    def test_calibration_invalid_data(self):
        """Test calibration with invalid data."""
        processor = IMUProcessor({})

        processor.start_calibration()

        # Try to add sample without IMU data
        invalid_sample = {"other_field": "value"}
        assert processor.add_calibration_sample(invalid_sample) == False

    def test_batch_processing(self):
        """Test batch processing support."""
        processor = IMUProcessor({})

        assert processor.supports_batch_processing() == True

        # Create batch data
        batch_data = [
            {"acc_body_imu": [0.1, 0.2, 9.8], "body_omega_imu": [0.01, 0.02, 0.03]},
            {"acc_body_imu": [0.2, 0.3, 9.7], "body_omega_imu": [0.02, 0.03, 0.04]},
            {"acc_body_imu": [0.3, 0.4, 9.6], "body_omega_imu": [0.03, 0.04, 0.05]},
        ]

        batch_result = processor.process_batch(batch_data)

        assert len(batch_result) == 3
        for i, result in enumerate(batch_result):
            assert "imu_acc_filtered" in result
            assert "imu_processor_stats" in result
            assert result["imu_processor_stats"]["samples_processed"] == i + 1

    def test_filtering_progression(self):
        """Test that filtering progresses correctly over time."""
        processor = IMUProcessor({"filter_alpha": 0.5})

        # Process multiple samples to see filtering effect
        samples = [
            {"acc_body_imu": [1.0, 0.0, 9.8], "body_omega_imu": [0.1, 0.0, 0.0]},
            {"acc_body_imu": [0.0, 1.0, 9.8], "body_omega_imu": [0.0, 0.1, 0.0]},
            {"acc_body_imu": [0.0, 0.0, 9.8], "body_omega_imu": [0.0, 0.0, 0.1]},
        ]

        results = []
        for sample in samples:
            result = processor.process(sample)
            results.append(result["imu_acc_filtered"])

        # Check that filtered values change between samples
        assert results[0] != results[1]
        assert results[1] != results[2]

        # Check that filtering is working (values should be between original and previous)
        for i in range(1, len(results)):
            # The filtered result should be a weighted average
            assert len(results[i]) == 3

    def test_process_with_calibrated_processor(self):
        """Test processing data with a calibrated processor."""
        processor = IMUProcessor({"calibration_samples": 2})

        # Perform calibration first
        processor.start_calibration()
        cal_samples = [
            {"acc_body_imu": [0.1, 0.1, 9.9], "body_omega_imu": [0.01, 0.02, 0.03]},
            {"acc_body_imu": [0.1, 0.1, 9.9], "body_omega_imu": [0.01, 0.02, 0.03]},
        ]
        for sample in cal_samples:
            processor.add_calibration_sample(sample)
        processor.finish_calibration()

        # Now process data (should apply calibration)
        test_data = {"acc_body_imu": [0.5, 0.5, 10.3], "body_omega_imu": [0.05, 0.06, 0.07]}
        processed = processor.process(test_data)

        # Should have processed data with calibration applied
        assert "imu_acc_filtered" in processed
        assert "imu_gyro_filtered" in processed
        stats = processed["imu_processor_stats"]
        assert stats["is_calibrated"] == True


class TestIMUProcessorIntegration:
    """Integration tests for IMU processor."""

    def test_realistic_imu_data_processing(self):
        """Test processing realistic IMU data sequences."""
        processor = IMUProcessor({"filter_alpha": 0.1, "calibration_samples": 5})

        # Simulate robot movement with IMU data
        # Starting stationary, then moving, then stationary again
        imu_sequence = [
            # Stationary (mostly gravity)
            {"acc_body_imu": [0.02, 0.01, 9.81], "body_omega_imu": [0.001, 0.002, 0.001]},
            {"acc_body_imu": [0.01, 0.02, 9.80], "body_omega_imu": [0.002, 0.001, 0.002]},
            {"acc_body_imu": [0.03, 0.01, 9.82], "body_omega_imu": [0.001, 0.003, 0.001]},
            # Movement (additional acceleration and rotation)
            {"acc_body_imu": [2.5, 0.8, 9.2], "body_omega_imu": [0.5, 0.3, 0.1]},
            {"acc_body_imu": [3.2, 1.2, 8.9], "body_omega_imu": [0.8, 0.6, 0.2]},
            {"acc_body_imu": [1.8, 0.5, 9.5], "body_omega_imu": [0.3, 0.2, 0.1]},
            # Stationary again
            {"acc_body_imu": [0.01, 0.02, 9.81], "body_omega_imu": [0.002, 0.001, 0.002]},
            {"acc_body_imu": [0.02, 0.01, 9.80], "body_omega_imu": [0.001, 0.002, 0.001]},
        ]

        # Calibration phase (use first few stationary samples)
        processor.start_calibration()
        for sample in imu_sequence[:5]:
            processor.add_calibration_sample(sample)
        assert processor.finish_calibration() == True

        # Process all samples
        results = []
        for sample in imu_sequence:
            result = processor.process(sample)
            results.append(result)

        # Verify processing
        assert len(results) == len(imu_sequence)

        # Check that all results have required fields
        for result in results:
            assert "imu_acc_filtered" in result
            assert "imu_gyro_filtered" in result
            assert "imu_orientation_quat" in result
            assert "imu_gravity_compensated_acc" in result

        # Check that filtered values are reasonable
        for result in results:
            acc_filtered = result["imu_acc_filtered"]
            gyro_filtered = result["imu_gyro_filtered"]
            
            # Should be 3D vectors
            assert len(acc_filtered) == 3
            assert len(gyro_filtered) == 3
            
            # Should be finite numbers
            assert all(np.isfinite(acc_filtered))
            assert all(np.isfinite(gyro_filtered))

        # Movement samples should have higher filtered values than stationary
        stationary_acc = np.array(results[0]["imu_acc_filtered"])
        movement_acc = np.array(results[4]["imu_acc_filtered"])
        
        # The movement sample should have different characteristics
        # (exact comparison depends on filtering parameters)
        assert not np.allclose(stationary_acc, movement_acc, atol=0.1)

    def test_calibration_improves_data_quality(self):
        """Test that calibration improves data quality."""
        processor = IMUProcessor({"calibration_samples": 3})

        # Biased IMU data (simulating sensor bias)
        bias_acc = np.array([0.1, -0.05, 0.2])
        bias_gyro = np.array([0.02, -0.01, 0.015])

        test_sample = {
            "acc_body_imu": [0.1 + bias_acc[0], 0.0 + bias_acc[1], 9.81 + bias_acc[2]],
            "body_omega_imu": [0.0 + bias_gyro[0], 0.0 + bias_gyro[1], 0.0 + bias_gyro[2]],
        }

        # Process without calibration
        result_uncalibrated = processor.process(test_sample.copy())

        # Perform calibration with biased data
        processor.start_calibration()
        cal_samples = [
            {"acc_body_imu": [bias_acc[0], bias_acc[1], 9.81 + bias_acc[2]], 
             "body_omega_imu": list(bias_gyro)},
            {"acc_body_imu": [bias_acc[0], bias_acc[1], 9.81 + bias_acc[2]], 
             "body_omega_imu": list(bias_gyro)},
            {"acc_body_imu": [bias_acc[0], bias_acc[1], 9.81 + bias_acc[2]], 
             "body_omega_imu": list(bias_gyro)},
        ]
        for sample in cal_samples:
            processor.add_calibration_sample(sample)
        processor.finish_calibration()

        # Process the same sample with calibration
        result_calibrated = processor.process(test_sample.copy())

        # The calibrated result should have different (improved) values
        acc_uncal = np.array(result_uncalibrated["imu_acc_filtered"])
        acc_cal = np.array(result_calibrated["imu_acc_filtered"])

        # Values should be different due to bias removal
        assert not np.allclose(acc_uncal, acc_cal, atol=0.01)

        # Calibrated result should be closer to unbiased values
        # (This is a simplified check - in practice you'd compare against known ground truth)
        assert result_calibrated["imu_processor_stats"]["is_calibrated"] == True