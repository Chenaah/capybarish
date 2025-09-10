#!/usr/bin/env python3
"""
Basic usage example for the Capybarish motion capture system.

This example demonstrates how to:
- Initialize and configure the Capybarish interface
- Connect to robot modules
- Send sinusoidal control commands to motors
- Handle real-time keyboard input for motor control
- Receive and process sensor data

The example runs a continuous control loop that sends sinusoidal position
commands to all configured robot modules while allowing real-time control
via keyboard input.

Keyboard Controls:
    'e': Enable motors (switch_on = 1)
    'd': Disable motors (switch_on = 0)
    Ctrl+C: Exit the program

Usage:
    python basic_usage.py                    # Use default configuration
    python basic_usage.py --cfg my_config   # Use custom configuration
    python basic_usage.py --help            # Show help message

Requirements:
    - Properly configured robot modules
    - Valid configuration file in config/ directory
    - Network connectivity to robot modules

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

import argparse
import signal
import sys
import time
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from capybarish.interface import Interface
from capybarish.kbhit import KBHit
from capybarish.utils import load_cfg

# Constants
DEFAULT_CONFIG_NAME = "default"
INITIAL_DATA_COLLECTION_CYCLES = 10
CONTROL_LOOP_SLEEP_TIME = 0.02  # 50 Hz control loop
SINUSOIDAL_AMPLITUDE = 0.2
SINUSOIDAL_FREQUENCY_DIVIDER = 3
DEFAULT_KP_GAIN = 8.0
DEFAULT_KD_GAIN = 0.2

# Keyboard command mappings
KEY_ENABLE_MOTORS = "e"
KEY_DISABLE_MOTORS = "d"

# Global variables for cleanup
interface: Optional[Interface] = None
kb: Optional[KBHit] = None


def signal_handler(signum: int, frame) -> None:
    """Handle interrupt signals for graceful shutdown.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_and_exit()


def cleanup_and_exit() -> None:
    """Perform cleanup operations and exit the program."""
    global interface, kb
    
    try:
        if interface is not None:
            print("Disabling motors...")
            interface.switch_on = 0
            # Send a final command to ensure motors are disabled
            if hasattr(interface, 'send_action'):
                zero_action = np.zeros(len(interface.cfg.interface.module_ids))
                interface.send_action(zero_action)
            print("Motors disabled.")
            
        if kb is not None:
            print("Cleaning up keyboard handler...")
            # KBHit cleanup if it has a cleanup method
            if hasattr(kb, 'set_normal_term'):
                kb.set_normal_term()
                
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        print("Cleanup complete. Exiting...")
        sys.exit(0)


def initialize_system(cfg: DictConfig) -> Interface:
    """Initialize the Capybarish interface system.
    
    Args:
        cfg: Configuration object containing system parameters
        
    Returns:
        Initialized Interface instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        print("Initializing Capybarish interface...")
        interface = Interface(cfg)
        
        print(f"Collecting initial data for {INITIAL_DATA_COLLECTION_CYCLES} cycles...")
        for i in range(INITIAL_DATA_COLLECTION_CYCLES):
            interface.receive_module_data()
            print(f"  Initial data collection: {i+1}/{INITIAL_DATA_COLLECTION_CYCLES}")
            
        print("System initialization complete!")
        return interface
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize system: {e}") from e


def run_control_loop(cfg: DictConfig) -> None:
    """Run the main control loop with sinusoidal commands and keyboard input.
    
    This function implements the main control loop that:
    1. Receives sensor data from robot modules
    2. Generates sinusoidal position commands
    3. Sends commands to all configured modules
    4. Handles keyboard input for motor control
    
    Args:
        cfg: Configuration object containing system parameters
        
    Raises:
        KeyboardInterrupt: When user requests shutdown
        RuntimeError: If control loop encounters critical errors
    """
    global interface, kb
    
    try:
        # Initialize system components
        interface = initialize_system(cfg)
        kb = KBHit()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\\nStarting control loop...")
        print("Controls: 'e' = enable motors, 'd' = disable motors, Ctrl+C = exit")
        print("=" * 60)
        
        time_step = 0
        num_modules = len(cfg.interface.module_ids)
        
        while True:
            try:
                # Receive sensor data from all modules
                interface.receive_module_data()
                
                # Get observable data (for logging/monitoring)
                observable_data = interface.get_observable_data()
                
                # Generate sinusoidal control command
                sinusoidal_command = SINUSOIDAL_AMPLITUDE * np.sin(time_step / SINUSOIDAL_FREQUENCY_DIVIDER)
                action_array = np.ones(num_modules) * sinusoidal_command
                
                # Send control commands to all modules
                interface.send_action(
                    action_array,
                    kps=np.array([DEFAULT_KP_GAIN]),
                    kds=np.array([DEFAULT_KD_GAIN])
                )
                
                # Handle keyboard input
                if kb.kbhit():
                    input_key = kb.getch()
                    handle_keyboard_input(input_key, interface)
                
                # Control loop timing
                time.sleep(CONTROL_LOOP_SLEEP_TIME)
                time_step += 1
                
                # Optional: Print status every N iterations
                if time_step % 100 == 0:
                    status = "ENABLED" if interface.switch_on else "DISABLED"
                    print(f"Step {time_step}: Motors {status}, Command: {sinusoidal_command:.3f}")
                    
            except KeyboardInterrupt:
                print("\\nKeyboard interrupt received...")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                # Continue loop for non-critical errors
                continue
                
    except Exception as e:
        print(f"Critical error in control loop: {e}")
        raise RuntimeError(f"Control loop failed: {e}") from e
    finally:
        cleanup_and_exit()


def handle_keyboard_input(key: str, interface: Interface) -> None:
    """Handle keyboard input commands.
    
    Args:
        key: Pressed key character
        interface: Interface instance to control
    """
    if key == KEY_ENABLE_MOTORS:
        interface.switch_on = 1
        print("Motors ENABLED")
    elif key == KEY_DISABLE_MOTORS:
        interface.switch_on = 0
        print("Motors DISABLED")
    else:
        # Ignore unknown keys silently
        pass


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Basic usage example for Capybarish motion capture system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--cfg', 
        type=str, 
        default=DEFAULT_CONFIG_NAME,
        help='Configuration file name (without .yaml extension)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the basic usage example."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        if args.verbose:
            print(f"Loading configuration: {args.cfg}")
        
        # Load configuration
        try:
            config = load_cfg(args.cfg)
        except FileNotFoundError as e:
            print(f"Error: Configuration file not found: {e}")
            print("Make sure the configuration file exists in the config/ directory")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
        
        if args.verbose:
            print(f"Configuration loaded successfully")
            print(f"Number of modules: {len(config.interface.module_ids)}")
        
        # Run the main control loop
        run_control_loop(config)
        
    except KeyboardInterrupt:
        print("\\nProgram interrupted by user")
        cleanup_and_exit()
    except Exception as e:
        print(f"Unexpected error: {e}")
        cleanup_and_exit()


if __name__ == "__main__":
    main()