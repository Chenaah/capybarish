"""
Message and status code interpreters for robot communication.

This module provides functions to interpret various status codes and messages
received from robot modules, including:
- Motor operating modes
- Motor error conditions
- System reset reasons
- Motor status messages

The interpreters convert numeric codes into human-readable strings for
logging, debugging, and user interfaces.

Typical usage example:
    from capybarish.interpreter import interpret_motor_mode, interpret_motor_error

    mode_str = interpret_motor_mode(2)  # Returns "Motor"
    error_str = interpret_motor_error(0x01)  # Returns "UNDERVOLTAGE"

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

from typing import Dict

# Motor mode constants
MOTOR_MODE_RESET = 0
MOTOR_MODE_CALIBRATION = 1
MOTOR_MODE_ACTIVE = 2

# Motor error bit flags
ERROR_UNDERVOLTAGE = 0x01
ERROR_OVERCURRENT = 0x02
ERROR_OVERTEMPERATURE = 0x04
ERROR_MAGNETIC_ENCODER = 0x08
ERROR_HALL_ENCODER = 0x10
ERROR_UNCALIBRATED = 0x20

# Motor mode mapping
MOTOR_MODE_NAMES: Dict[int, str] = {
    MOTOR_MODE_RESET: "Reset",
    MOTOR_MODE_CALIBRATION: "Cali",
    MOTOR_MODE_ACTIVE: "Motor"
}

# Reset reason codes and descriptions
RESET_REASONS: Dict[int, str] = {
    1: "POWERON_RESET: Vbat power on reset",
    3: "SW_RESET: Software reset digital core",
    4: "OWDT_RESET: Legacy watch dog reset digital core",
    5: "DEEPSLEEP_RESET: Deep Sleep reset digital core",
    6: "SDIO_RESET: Reset by SLC module, reset digital core",
    7: "TG0WDT_SYS_RESET: Timer Group0 Watch dog reset digital core",
    8: "TG1WDT_SYS_RESET: Timer Group1 Watch dog reset digital core",
    9: "RTCWDT_SYS_RESET: RTC Watch dog Reset digital core",
    10: "INTRUSION_RESET: Instrusion tested to reset CPU",
    11: "TGWDT_CPU_RESET: Time Group reset CPU",
    12: "SW_CPU_RESET: Software reset CPU",
    13: "RTCWDT_CPU_RESET: RTC Watch dog Reset CPU",
    14: "EXT_CPU_RESET: for APP CPU, reseted by PRO CPU",
    15: "RTCWDT_BROWN_OUT_RESET: Reset when the vdd voltage is not stable",
    16: "RTCWDT_RTC_RESET: RTC Watch dog reset digital core and rtc module"
}

# Motor status messages
MOTOR_MESSAGES: Dict[int, str] = {
    0: "",
    100: "[WARN][Wifi] Wifi disconnected! Try to reconnect!",
    200: "[ERROR] No BNO055 detected.",
    201: "[IMU] BNO055 inited.",
    300: "[ERROR] [Motor] The motor is not calibrated.",
    301: "[Motor] Motor enabled.",
    302: "[Motor] Motor disabled.",
    303: "[WARN] [Motor] The remote swich is on. Please turn it off!!!",
    304: "[Motor] Please turn on the remote switch for motor calibration!",
    305: "[WARN] [Motor] Unsafe situations found!",
    306: "[Motor] Set to middle position",
    307: "[Motor] motor set to middle position.",
    308: "[Motor] Start auto calibration!",
    309: "[Motor] Zero position set.",
    310: "[Motor] Start manual calibration!",
    311: "[Motor] Zero position set.",
    312: "[Motor] Detect calibration command.",
    313: "[Motor] Motor initialized.",
    314: "[Motor] Swith-off request sent!",
    315: "[Motor] Restart!"
}


def interpret_motor_mode(mode: int) -> str:
    """Interpret motor operating mode code.
    
    Args:
        mode: Motor mode code (0=Reset, 1=Calibration, 2=Motor)
        
    Returns:
        Human-readable string describing the motor mode
        
    Example:
        >>> interpret_motor_mode(2)
        'Motor'
        >>> interpret_motor_mode(0)
        'Reset'
    """
    return MOTOR_MODE_NAMES.get(mode, "?")


def interpret_motor_error(error: int) -> str:
    """Interpret motor error flags.
    
    Processes a bitmask of error conditions and returns a string
    describing all active error conditions.
    
    Args:
        error: Bitmask containing error flags
        
    Returns:
        String describing all active error conditions, separated by spaces
        
    Example:
        >>> interpret_motor_error(0x01)
        'UNDERVOLTAGE'
        >>> interpret_motor_error(0x03)
        'UNDERVOLTAGE OVER CURRENT'
    """
    error_flags = []
    
    if error & ERROR_UNDERVOLTAGE:
        error_flags.append("UNDERVOLTAGE")
    if error & ERROR_OVERCURRENT:
        error_flags.append("OVER CURRENT")
    if error & ERROR_OVERTEMPERATURE:
        error_flags.append("OVER TEMPERATURE")
    if error & ERROR_MAGNETIC_ENCODER:
        error_flags.append("MAGNETIC ENCODER ERROR")
    if error & ERROR_HALL_ENCODER:
        error_flags.append("HALL ENCODER ERROR")
    if error & ERROR_UNCALIBRATED:
        error_flags.append("UNCALIBRATED")
    
    return " ".join(error_flags)


def interpret_reset_reason(reason: int) -> str:
    """Interpret system reset reason code.
    
    Converts ESP32 reset reason codes into human-readable descriptions.
    
    Args:
        reason: Reset reason code from ESP32 system
        
    Returns:
        Human-readable description of the reset reason
        
    Example:
        >>> interpret_reset_reason(1)
        'POWERON_RESET: Vbat power on reset'
        >>> interpret_reset_reason(999)
        'NO_MEAN'
    """
    return RESET_REASONS.get(reason, "NO_MEAN")


def interpret_motor_msg(msg: int) -> str:
    """Interpret motor status message code.
    
    Converts numeric motor status codes into descriptive messages
    for logging and user feedback.
    
    Args:
        msg: Motor message code
        
    Returns:
        Human-readable message string
        
    Raises:
        KeyError: If the message code is not recognized
        
    Example:
        >>> interpret_motor_msg(301)
        '[Motor] Motor enabled.'
        >>> interpret_motor_msg(0)
        ''
    """
    try:
        return MOTOR_MESSAGES[msg]
    except KeyError:
        raise KeyError(f"Unknown motor message code: {msg}. "
                      f"Valid codes are: {list(MOTOR_MESSAGES.keys())}")