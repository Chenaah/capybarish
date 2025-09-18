"""
IMU Data Processor Plugin

This plugin processes IMU data from robot modules, providing filtering,
calibration, and orientation estimation capabilities.

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

import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    # Fallback Rotation class for testing without scipy
    class Rotation:
        def __init__(self, quat=None):
            self._quat = quat if quat is not None else np.array([0, 0, 0, 1])
        
        @classmethod
        def from_rotvec(cls, vec):
            # Simple fallback - identity quaternion
            return cls(np.array([0, 0, 0, 1]))
        
        @classmethod  
        def from_euler(cls, seq, angles):
            # Simple fallback - identity quaternion
            return cls(np.array([0, 0, 0, 1]))
            
        @classmethod
        def from_quat(cls, quat):
            return cls(quat)
        
        def as_quat(self):
            return self._quat
            
        def inv(self):
            # Return identity for fallback
            return Rotation(np.array([0, 0, 0, 1]))
            
        def apply(self, vector):
            # Return vector unchanged for fallback
            return np.array(vector)
    
    SCIPY_AVAILABLE = False

from capybarish.plugin_system import DataProcessorPlugin, PluginMetadata, PluginType


class IMUProcessor(DataProcessorPlugin):
    """
    IMU data processor plugin for filtering and processing inertial measurement data.

    Features:
    - Low-pass filtering for accelerometer and gyroscope data
    - Gravity compensation
    - Orientation estimation
    - Calibration support
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration parameters
        self.filter_alpha = config.get("filter_alpha", 0.1)
        self.gravity_threshold = config.get("gravity_threshold", 0.5)
        self.calibration_samples = config.get("calibration_samples", 100)

        # State variables
        self.filtered_acc = np.zeros(3)
        self.filtered_gyro = np.zeros(3)
        self.gravity_vector = np.array([0, 0, -9.81])
        self.bias_acc = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        self.calibration_data = {"acc": [], "gyro": []}
        self.is_calibrated = False

        # Statistics
        self.samples_processed = 0
        self.last_process_time = 0

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="IMUProcessor",
            version="1.0.0",
            description="Processes IMU data with filtering and calibration",
            author="Robot Middleware Team",
            plugin_type=PluginType.DATA_PROCESSOR,
            dependencies=[],
            tags={"imu", "sensor", "filtering", "calibration"},
        )

    def initialize(self) -> bool:
        """Initialize the IMU processor."""
        try:
            self.status = (
                self.status.INITIALIZED if hasattr(self.status, "INITIALIZED") else self.status
            )
            print(f"[{self.metadata.name}] Initialized with filter_alpha={self.filter_alpha}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def start(self) -> bool:
        """Start the IMU processor."""
        try:
            self.status = self.status.RUNNING if hasattr(self.status, "RUNNING") else self.status
            print(f"[{self.metadata.name}] Started")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Stop the IMU processor."""
        try:
            self.status = self.status.STOPPED if hasattr(self.status, "STOPPED") else self.status
            print(f"[{self.metadata.name}] Stopped")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []

        if "filter_alpha" in config:
            alpha = config["filter_alpha"]
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 1:
                errors.append("filter_alpha must be a number between 0 and 1")

        if "calibration_samples" in config:
            samples = config["calibration_samples"]
            if not isinstance(samples, int) or samples <= 0:
                errors.append("calibration_samples must be a positive integer")

        return errors

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process IMU data.

        Args:
            data: Input data containing IMU measurements

        Returns:
            Processed data with filtered and calibrated IMU values
        """
        processed_data = data.copy()
        current_time = time.time()
        self.last_process_time = current_time

        try:
            # Extract IMU data from different possible sources
            acc_data = self._extract_accelerometer_data(data)
            gyro_data = self._extract_gyroscope_data(data)

            if acc_data is not None and gyro_data is not None:
                # Apply calibration if available
                if self.is_calibrated:
                    acc_data = acc_data - self.bias_acc
                    gyro_data = gyro_data - self.bias_gyro

                # Apply low-pass filtering
                self.filtered_acc = self._low_pass_filter(
                    self.filtered_acc, acc_data, self.filter_alpha
                )
                self.filtered_gyro = self._low_pass_filter(
                    self.filtered_gyro, gyro_data, self.filter_alpha
                )

                # Estimate orientation
                orientation_quat = self._estimate_orientation(self.filtered_acc, self.filtered_gyro)

                # Increment counter first
                self.samples_processed += 1

                # Add processed data to output
                processed_data.update(
                    {
                        "imu_acc_filtered": self.filtered_acc.tolist(),
                        "imu_gyro_filtered": self.filtered_gyro.tolist(),
                        "imu_orientation_quat": orientation_quat.tolist(),
                        "imu_gravity_compensated_acc": self._compensate_gravity(
                            self.filtered_acc, orientation_quat
                        ).tolist(),
                        "imu_processor_stats": {
                            "samples_processed": self.samples_processed,
                            "is_calibrated": self.is_calibrated,
                            "last_process_time": current_time,
                        },
                    }
                )

        except Exception as e:
            self.last_error = f"Error processing IMU data: {e}"
            processed_data["imu_processor_error"] = self.last_error

        return processed_data

    def _extract_accelerometer_data(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract accelerometer data from input."""
        # Try different possible field names
        for field in ["acc_body_imu", "accelerometer", "acc", "accs"]:
            if field in data:
                acc_value = data[field]
                if isinstance(acc_value, (list, tuple)):
                    return np.array(acc_value[:3])  # Take first 3 elements
                elif isinstance(acc_value, np.ndarray):
                    return acc_value[:3]
        return None  # Return None if no IMU data found

    def _extract_gyroscope_data(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract gyroscope data from input."""
        # Try different possible field names
        for field in ["body_omega_imu", "gyroscope", "gyro", "gyros", "ang_vel_body"]:
            if field in data:
                gyro_value = data[field]
                if isinstance(gyro_value, (list, tuple)):
                    return np.array(gyro_value[:3])  # Take first 3 elements
                elif isinstance(gyro_value, np.ndarray):
                    return gyro_value[:3]
        return None  # Return None if no IMU data found

    def _low_pass_filter(
        self, previous: np.ndarray, current: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Apply low-pass filter to data."""
        return alpha * current + (1 - alpha) * previous

    def _estimate_orientation(self, acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """
        Estimate orientation from accelerometer and gyroscope data.

        This is a simplified implementation. In practice, you'd use a more
        sophisticated algorithm like complementary filter or Kalman filter.
        """
        try:
            # Normalize accelerometer data
            acc_norm = np.linalg.norm(acc)
            if acc_norm > 0:
                acc_normalized = acc / acc_norm

                # Simple tilt estimation from accelerometer
                # This assumes the accelerometer primarily measures gravity when stationary
                roll = np.arctan2(acc_normalized[1], acc_normalized[2])
                pitch = np.arctan2(
                    -acc_normalized[0], np.sqrt(acc_normalized[1] ** 2 + acc_normalized[2] ** 2)
                )
                yaw = 0  # Cannot determine yaw from accelerometer alone

                # Convert to quaternion
                rotation = Rotation.from_euler("xyz", [roll, pitch, yaw])
                return rotation.as_quat()
            else:
                return np.array([0, 0, 0, 1])  # Identity quaternion

        except Exception:
            return np.array([0, 0, 0, 1])  # Identity quaternion on error

    def _compensate_gravity(self, acc: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
        """Remove gravity component from accelerometer data."""
        try:
            # Rotate gravity vector to body frame
            rotation = Rotation.from_quat(orientation_quat)
            gravity_body = rotation.inv().apply(self.gravity_vector)

            # Remove gravity from acceleration
            linear_acc = acc - gravity_body
            return linear_acc

        except Exception:
            return acc  # Return original data on error

    def start_calibration(self) -> bool:
        """Start IMU calibration process."""
        self.calibration_data = {"acc": [], "gyro": []}
        self.is_calibrated = False
        print(f"[{self.metadata.name}] Started calibration")
        return True

    def add_calibration_sample(self, data: Dict[str, Any]) -> bool:
        """Add a sample to calibration data."""
        acc_data = self._extract_accelerometer_data(data)
        gyro_data = self._extract_gyroscope_data(data)

        if acc_data is not None and gyro_data is not None:
            self.calibration_data["acc"].append(acc_data)
            self.calibration_data["gyro"].append(gyro_data)
            return True

        return False

    def finish_calibration(self) -> bool:
        """Finish calibration and compute bias values."""
        try:
            if (
                len(self.calibration_data["acc"]) >= self.calibration_samples
                and len(self.calibration_data["gyro"]) >= self.calibration_samples
            ):
                # Compute bias as mean of calibration samples
                self.bias_acc = np.mean(self.calibration_data["acc"], axis=0)
                self.bias_gyro = np.mean(self.calibration_data["gyro"], axis=0)

                # For accelerometer, subtract expected gravity
                # (assuming calibration is done with sensor stationary)
                gravity_magnitude = np.linalg.norm(self.bias_acc)
                if gravity_magnitude > 8:  # Reasonable gravity check
                    self.bias_acc = self.bias_acc - self.gravity_vector

                self.is_calibrated = True
                print(f"[{self.metadata.name}] Calibration completed")
                print(f"  Acc bias: {self.bias_acc}")
                print(f"  Gyro bias: {self.bias_gyro}")
                return True
            else:
                print(f"[{self.metadata.name}] Insufficient calibration samples")
                return False

        except Exception as e:
            self.last_error = f"Calibration failed: {e}"
            return False

    def supports_batch_processing(self) -> bool:
        """This processor supports batch processing."""
        return True

    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of IMU data efficiently."""
        return [self.process(data) for data in data_list]
