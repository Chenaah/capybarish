"""
Data structures for robot communication and sensor data handling.

This module defines data structures used for:
- Serializing command data to send to robot modules
- Deserializing sensor and status data received from robot modules
- Converting between different data formats and representations
- Handling IMU, motor, and system status information

The module supports both full and lite data formats for different
bandwidth and processing requirements.

Typical usage example:
    # Create and serialize command data
    cmd = SentDataStruct(target_pos=1.5, target_vel=0.0, kp=10.0, kd=1.0,
                        enable_filter=1, switch=1, calibrate=0, restart=0,
                        timestamp=int(time.time()))
    serialized = cmd.serialize()

    # Deserialize received robot data
    robot_data = RobotData.unpack(received_bytes, format_string)
    data_dict = robot_data.get_data_dict()

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

import struct
import time
from typing import Any, Dict, List, Union

import numpy as np

from .interpreter import interpret_motor_msg

# Constants for data structure formats
SENT_DATA_FORMAT = "ffffiiiif"  # target_pos, target_vel, kp, kd, enable_filter, switch, calibrate, restart, timestamp
DB_COMMAND_FORMAT = "iii"  # switch, calibrate, restart
ROBOT_DATA_FORMAT = "iiiififfffffiiifffffffffffffii"
ROBOT_DATA_LITE_FORMAT = "BBBBHBHHHHBBHHHHHHHHHHHHHBB"

# Motor mode constants
MOTOR_MODE_RESET = 0
MOTOR_MODE_CALIBRATION = 1
MOTOR_MODE_ACTIVE = 2

# Bit manipulation constants
MOTOR_MODE_MASK = 0x03
MOTOR_ERROR_MASK = 0x3F
MOTOR_MODE_SHIFT = 6

# Data conversion constants
MICROSECOND_CONVERSION = 1e-6


class SentDataStruct:
    """Data structure for commands sent to robot modules.

    This class encapsulates motor control parameters and system commands
    that are sent to robot modules. It provides serialization functionality
    to convert the data into a binary format suitable for transmission.

    Attributes:
        target_pos: Target position for the motor (float)
        target_vel: Target velocity for the motor (float)
        kp: Proportional gain for position control (float)
        kd: Derivative gain for position control (float)
        enable_filter: Flag to enable/disable filtering (int, 0 or 1)
        switch: Motor switch state (int, 0 or 1)
        calibrate: Calibration command flag (int, 0 or 1)
        restart: Restart command flag (int, 0 or 1)
        timestamp: Command timestamp (int)
    """

    def __init__(
        self,
        target_pos: float,
        target_vel: float,
        kp: float,
        kd: float,
        enable_filter: int,
        switch: int,
        calibrate: int,
        restart: int,
        timestamp: int,
    ) -> None:
        """Initialize the sent data structure.

        Args:
            target_pos: Target position for the motor
            target_vel: Target velocity for the motor
            kp: Proportional gain for position control
            kd: Derivative gain for position control
            enable_filter: Flag to enable/disable filtering (0 or 1)
            switch: Motor switch state (0 or 1)
            calibrate: Calibration command flag (0 or 1)
            restart: Restart command flag (0 or 1)
            timestamp: Command timestamp
        """
        self.target_pos = target_pos
        self.target_vel = target_vel
        self.kp = kp
        self.kd = kd
        self.enable_filter = enable_filter
        self.switch = switch
        self.calibrate = calibrate
        self.restart = restart
        self.timestamp = timestamp

    def serialize(self) -> bytes:
        """Serialize the data structure to binary format.

        Returns:
            Binary representation of the data structure suitable for transmission.

        Raises:
            struct.error: If the data cannot be packed into the specified format.
        """
        try:
            return struct.pack(
                SENT_DATA_FORMAT,
                self.target_pos,
                self.target_vel,
                self.kp,
                self.kd,
                self.enable_filter,
                self.switch,
                self.calibrate,
                self.restart,
                self.timestamp,
            )
        except struct.error as e:
            raise struct.error(f"Failed to serialize SentDataStruct: {e}") from e


class DBCommandStruct:
    """Data structure for database command operations.

    This class encapsulates system-level commands that control database
    operations and system state. It provides serialization functionality
    for network transmission.

    Attributes:
        switch: System switch state (int, 0 or 1)
        calibrate: Calibration command flag (int, 0 or 1)
        restart: System restart command flag (int, 0 or 1)
    """

    def __init__(self, switch: int, calibrate: int, restart: int) -> None:
        """Initialize the database command structure.

        Args:
            switch: System switch state (0 or 1)
            calibrate: Calibration command flag (0 or 1)
            restart: System restart command flag (0 or 1)
        """
        self.switch = switch
        self.calibrate = calibrate
        self.restart = restart

    def serialize(self) -> bytes:
        """Serialize the command structure to binary format.

        Returns:
            Binary representation of the command structure.

        Raises:
            struct.error: If the data cannot be packed into the specified format.
        """
        try:
            return struct.pack(DB_COMMAND_FORMAT, self.switch, self.calibrate, self.restart)
        except struct.error as e:
            raise struct.error(f"Failed to serialize DBCommandStruct: {e}") from e


class RobotData:
    """Data structure for robot sensor and status data.

    This class handles the deserialization and processing of data received
    from robot modules, including motor status, IMU data, system information,
    and error states.

    Attributes:
        unpacked_data: List of unpacked binary data values
        module_id: Unique identifier for the robot module
    """

    def __init__(self, unpacked_data: List[Union[int, float]]) -> None:
        """Initialize the robot data structure.

        Args:
            unpacked_data: List of values unpacked from binary data
        """
        self.unpacked_data = unpacked_data
        self.module_id = unpacked_data[0]

    def get_data_dict(self) -> Dict[str, Any]:
        """Convert unpacked data to a structured dictionary.

        Processes the raw unpacked data and converts it into a structured
        dictionary with meaningful field names and processed values.

        Returns:
            Dictionary containing all sensor data, motor status, IMU readings,
            and system information with descriptive keys.

        Raises:
            IndexError: If the unpacked data doesn't contain enough elements.
            ValueError: If data interpretation fails.
        """
        try:
            data = {}
            unpacked_data = self.unpacked_data.copy()

            # Basic system information
            data["module_id"] = unpacked_data.pop(0)  # int
            data["received_dt"] = unpacked_data.pop(0) * MICROSECOND_CONVERSION  # int -> float
            data["timestamp"] = unpacked_data.pop(0)  # int
            data["switch_off_request"] = unpacked_data.pop(0)  # int
            data["last_rcv_timestamp"] = unpacked_data.pop(0)  # float

            # Motor and system status
            info = unpacked_data.pop(0)  # int
            data["log_info"] = interpret_motor_msg(info)

            # Motor sensor data
            data["motor_pos"] = unpacked_data.pop(0)  # float
            data["energy"] = unpacked_data.pop(0)  # float
            data["motor_vel"] = unpacked_data.pop(0)  # float
            data["motor_torque"] = unpacked_data.pop(0)  # float
            data["voltage"] = unpacked_data.pop(0)  # float
            data["current"] = unpacked_data.pop(0)  # float
            data["temperature"] = unpacked_data.pop(0)  # int

            # Motor mode and error processing
            motor_mode_error = unpacked_data.pop(0)  # int
            data["add_error"] = unpacked_data.pop(0)  # int

            # IMU sensor data
            data["euler_imu"] = [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            data["body_rot_imu"] = np.array(
                [
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                ]
            )
            data["body_omega_imu"] = np.array(
                [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            )
            data["acc_body_imu"] = np.array(
                [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            )

            # System error information
            data["esp_errors"] = [unpacked_data.pop(0), unpacked_data.pop(0)]

            # Process motor mode and error flags
            data["motor_mode"] = (motor_mode_error >> MOTOR_MODE_SHIFT) & MOTOR_MODE_MASK
            data["motor_error"] = motor_mode_error & MOTOR_ERROR_MASK
            data["motor_on"] = data["motor_mode"] == MOTOR_MODE_ACTIVE

            return data

        except (IndexError, KeyError) as e:
            raise ValueError(
                f"Failed to process robot data: insufficient or invalid data - {e}"
            ) from e

    @staticmethod
    def unpack(data: bytes, struct_format: str) -> "RobotData":
        """Unpack binary data into a RobotData instance.

        Args:
            data: Binary data received from robot module
            struct_format: Format string for struct unpacking

        Returns:
            RobotData instance containing the unpacked data

        Raises:
            struct.error: If the data cannot be unpacked with the given format
            ValueError: If the struct format is not the expected format
        """
        if struct_format != ROBOT_DATA_FORMAT:
            raise ValueError(
                f"Invalid struct format. Expected {ROBOT_DATA_FORMAT}, got {struct_format}"
            )

        try:
            unpacked_data = struct.unpack(struct_format, data)
            return RobotData(list(unpacked_data))
        except struct.error as e:
            raise struct.error(f"Failed to unpack robot data: {e}") from e


def half_to_float(value: int) -> float:
    """Convert a 16-bit integer to a 32-bit float.

    Args:
        value: 16-bit integer value to convert

    Returns:
        32-bit float representation of the input value
    """
    return np.float16(value).astype(np.float32)


class RobotDataLite:
    """Lightweight data structure for robot sensor data.

    This class provides a more compact representation of robot data
    with reduced precision for applications where bandwidth is limited
    or high precision is not required.

    Attributes:
        unpacked_data: List of unpacked binary data values
        module_id: Unique identifier for the robot module
    """

    def __init__(self, unpacked_data: List[Union[int, float]]) -> None:
        """Initialize the lite robot data structure.

        Args:
            unpacked_data: List of values unpacked from binary data
        """
        self.unpacked_data = unpacked_data
        self.module_id = unpacked_data[0]

    def get_data_dict(self, start_time: float = 0.0) -> Dict[str, Any]:
        """Convert unpacked data to a structured dictionary with latency calculation.

        Processes the raw unpacked data similar to RobotData but with additional
        latency calculation based on the provided start time.

        Args:
            start_time: Reference time for latency calculation (default: 0.0)

        Returns:
            Dictionary containing all sensor data with an additional 'latency' field

        Raises:
            IndexError: If the unpacked data doesn't contain enough elements
            ValueError: If data interpretation fails
        """
        try:
            data = {}
            unpacked_data = self.unpacked_data.copy()

            # Basic system information
            data["module_id"] = unpacked_data.pop(0)
            data["received_dt"] = unpacked_data.pop(0) * MICROSECOND_CONVERSION
            data["timestamp"] = unpacked_data.pop(0)
            data["switch_off_request"] = unpacked_data.pop(0)
            data["last_rcv_timestamp"] = unpacked_data.pop(0)

            # Motor and system status
            info = unpacked_data.pop(0)
            data["log_info"] = interpret_motor_msg(info)

            # Motor sensor data (lite version - no energy and current)
            data["motor_pos"] = unpacked_data.pop(0)
            data["motor_vel"] = unpacked_data.pop(0)
            data["motor_torque"] = unpacked_data.pop(0)
            data["voltage"] = unpacked_data.pop(0)
            data["temperature"] = unpacked_data.pop(0)

            # Motor mode and error processing
            motor_mode_error = unpacked_data.pop(0)

            # IMU sensor data
            data["euler_imu"] = [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            data["body_rot_imu"] = np.array(
                [
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                    unpacked_data.pop(0),
                ]
            )
            data["body_omega_imu"] = np.array(
                [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            )
            data["acc_body_imu"] = np.array(
                [unpacked_data.pop(0), unpacked_data.pop(0), unpacked_data.pop(0)]
            )

            # System error information
            data["esp_errors"] = [unpacked_data.pop(0), unpacked_data.pop(0)]

            # Process motor mode and error flags
            data["motor_mode"] = (motor_mode_error >> MOTOR_MODE_SHIFT) & MOTOR_MODE_MASK
            data["motor_error"] = motor_mode_error & MOTOR_ERROR_MASK
            data["motor_on"] = data["motor_mode"] == MOTOR_MODE_ACTIVE

            # Calculate latency
            current_timestamp = float(time.time()) - start_time
            data["latency"] = current_timestamp - data["last_rcv_timestamp"]

            return data

        except (IndexError, KeyError) as e:
            raise ValueError(
                f"Failed to process robot lite data: insufficient or invalid data - {e}"
            ) from e

    @staticmethod
    def unpack(data: bytes, struct_format: str) -> "RobotDataLite":
        """Unpack binary data into a RobotDataLite instance.

        Args:
            data: Binary data received from robot module
            struct_format: Format string for struct unpacking

        Returns:
            RobotDataLite instance containing the unpacked data

        Raises:
            struct.error: If the data cannot be unpacked with the given format
            ValueError: If the struct format is not the expected format
        """
        if struct_format != ROBOT_DATA_LITE_FORMAT:
            raise ValueError(
                f"Invalid struct format. Expected {ROBOT_DATA_LITE_FORMAT}, got {struct_format}"
            )

        try:
            unpacked_data = struct.unpack(struct_format, data)
            data_list = list(unpacked_data)

            # Convert half-precision floats (format 'H') to full floats
            for i, (format_char, value) in enumerate(zip(struct_format, data_list)):
                if format_char == "H":
                    data_list[i] = half_to_float(value)

            return RobotDataLite(data_list)

        except struct.error as e:
            raise struct.error(f"Failed to unpack robot lite data: {e}") from e
