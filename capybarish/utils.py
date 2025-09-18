"""
Utility functions for the Capybarish motion capture system.

This module provides various utility functions for:
- Configuration management and loading
- Network ping utilities for connectivity testing
- Data caching and serialization
- NumPy array conversion utilities
- Command execution helpers

Typical usage example:
    from capybarish.utils import load_cfg, get_ping_time

    config = load_cfg("my_config")
    ping_time = get_ping_time("192.168.1.100")

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

import datetime
import os
import pickle
import shlex
from subprocess import PIPE, STDOUT, Popen
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf

# Constants
DEFAULT_CACHE_DIR = "cached_data"
CACHED_PINGS_FILE = "cached_pings.pickle"
DEFAULT_RECENT_THRESHOLD_MINUTES = 10
DEFAULT_PING_COUNT = 3
DEFAULT_TIMEOUT_VALUE = 9999.0


def load_cached_pings(recent_threshold: int = DEFAULT_RECENT_THRESHOLD_MINUTES) -> Dict[str, Any]:
    """Load cached ping data if the cache file is recent enough.

    Args:
        recent_threshold: Time threshold in minutes. Only load cache if file
                         was modified within this many minutes.

    Returns:
        Dictionary containing cached ping data, or empty dict if cache
        is stale or doesn't exist.

    Raises:
        OSError: If there's an issue accessing the cache file.
        pickle.PickleError: If the cached data is corrupted.
    """
    file_path = os.path.join(DEFAULT_CACHE_DIR, CACHED_PINGS_FILE)

    try:
        # Calculate the threshold datetime
        threshold_time = datetime.datetime.now() - datetime.timedelta(minutes=recent_threshold)

        if not os.path.exists(file_path):
            return {}

        # Check if file was modified recently
        file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

        if file_mod_time <= threshold_time:
            return {}

        # Load and return the cached data
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except (OSError, pickle.PickleError) as e:
        raise OSError(f"Failed to load cached pings from {file_path}: {e}") from e


def cache_pings(data: Dict[str, Any]) -> None:
    """Cache ping data to a pickle file.

    Args:
        data: Dictionary containing ping data to cache.

    Raises:
        OSError: If there's an issue creating the cache directory or writing the file.
        pickle.PickleError: If the data cannot be pickled.
    """
    file_path = os.path.join(DEFAULT_CACHE_DIR, CACHED_PINGS_FILE)

    try:
        # Ensure cache directory exists
        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

        # Save the data to the pickle file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    except (OSError, pickle.PickleError) as e:
        raise OSError(f"Failed to cache pings to {file_path}: {e}") from e


def convert_np_arrays_to_lists(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NumPy arrays in a dictionary to Python lists.

    This function recursively converts any NumPy arrays found as values
    in the input dictionary to Python lists, making the data JSON-serializable.

    Args:
        input_dict: Dictionary that may contain NumPy arrays as values.

    Returns:
        New dictionary with NumPy arrays converted to lists.

    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'scalar': 42}
        >>> convert_np_arrays_to_lists(data)
        {'array': [1, 2, 3], 'scalar': 42}
    """
    converted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            converted_dict[key] = value.tolist()
        else:
            converted_dict[key] = value
    return converted_dict


def get_simple_cmd_output(cmd: str, stderr: Optional[int] = STDOUT) -> bytes:
    """Execute a simple external command and get its output.

    Args:
        cmd: Command string to execute.
        stderr: Where to redirect stderr. Defaults to STDOUT.

    Returns:
        Command output as bytes.

    Raises:
        OSError: If the command cannot be executed.

    Example:
        >>> output = get_simple_cmd_output("echo 'hello world'")
        >>> print(output.decode().strip())
        hello world
    """
    try:
        args = shlex.split(cmd)
        return Popen(args, stdout=PIPE, stderr=stderr).communicate()[0]
    except OSError as e:
        raise OSError(f"Failed to execute command '{cmd}': {e}") from e


def get_ping_time(host: str) -> float:
    """Get average ping time to a host using fping.

    This function uses the fping utility to send multiple ping packets
    to a host and returns the average response time.

    Args:
        host: Hostname or IP address to ping. Port numbers will be stripped.

    Returns:
        Average ping time in milliseconds. Returns DEFAULT_TIMEOUT_VALUE
        if the host is unreachable or if fping is not available.

    Raises:
        OSError: If fping command fails to execute.

    Example:
        >>> ping_time = get_ping_time("google.com")
        >>> print(f"Ping time: {ping_time:.2f}ms")
        Ping time: 25.34ms
    """
    # Remove port number if present
    host = host.split(":")[0]
    cmd = f"fping {host} -C {DEFAULT_PING_COUNT} -q"

    try:
        output = get_simple_cmd_output(cmd)
        result_str = output.decode("utf-8", errors="ignore")

        # Extract ping times from the output
        # fping output format: "host : x.xx x.xx x.xx"
        if ":" not in result_str:
            return DEFAULT_TIMEOUT_VALUE

        ping_data = result_str.split(":")[-1].strip()
        # Remove any remaining artifacts and split by whitespace
        ping_values = []
        for value in ping_data.split():
            try:
                # Skip '-' values (failed pings)
                if value != "-" and value:
                    ping_values.append(float(value))
            except ValueError:
                continue

        if ping_values:
            return sum(ping_values) / len(ping_values)
        else:
            return DEFAULT_TIMEOUT_VALUE

    except (OSError, UnicodeDecodeError) as e:
        # Return timeout value if fping fails or output cannot be decoded
        return DEFAULT_TIMEOUT_VALUE


def load_cfg(name: str = "default", alg: str = "sbx") -> OmegaConf:
    """Load configuration from the local config folder.

    This function loads YAML configuration files from the local config directory.
    It first attempts to use the new configuration manager, and falls back to
    legacy loading if that fails. Configurations are merged with the default
    configuration if available.

    Args:
        name: Name of the configuration file (without .yaml extension).
        alg: Algorithm parameter (maintained for backward compatibility).

    Returns:
        OmegaConf configuration object containing the loaded configuration.

    Raises:
        FileNotFoundError: If the specified configuration file doesn't exist.
        OSError: If there's an issue reading the configuration files.

    Example:
        >>> config = load_cfg("my_config")
        >>> print(config.some_parameter)
    """
    config_dir = "config"

    try:
        # Try to use the new configuration manager (restricted to local config)
        from .config_manager import load_cfg as new_load_cfg

        return new_load_cfg(name, config_dir=config_dir)

    except ImportError:
        # Fallback to legacy configuration loading if config_manager is not available
        pass
    except Exception as e:
        # Log warning but continue with fallback
        import warnings

        warnings.warn(
            f"New config manager failed ({e}), falling back to legacy loader",
            UserWarning,
            stacklevel=2,
        )

    # Legacy configuration loading
    yaml_file = os.path.join(config_dir, f"{name}.yaml")

    if not os.path.exists(yaml_file):
        raise FileNotFoundError(
            f"Configuration file not found: {yaml_file}. "
            f"Available files should be placed in the '{config_dir}' directory."
        )

    try:
        conf = OmegaConf.load(yaml_file)

        # Try to load and merge with default config
        default_conf_path = os.path.join(config_dir, "default.yaml")

        if os.path.exists(default_conf_path):
            default_conf = OmegaConf.load(default_conf_path)
            return OmegaConf.merge(default_conf, conf)
        else:
            return conf

    except Exception as e:
        raise OSError(f"Failed to load configuration from {yaml_file}: {e}") from e
