"""
OptiTrack Data Source Plugin

This plugin provides data from OptiTrack motion capture system,
integrating with the existing NatNet client functionality.

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

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from capybarish.plugin_system import DataSourcePlugin, PluginMetadata, PluginType


class OptiTrackSource(DataSourcePlugin):
    """
    OptiTrack data source plugin for motion capture data.

    This plugin wraps the existing OptiTrack functionality and provides
    it through the plugin system interface.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.server_address = config.get("server_address", "129.105.73.172")
        self.client_address = config.get("client_address", "0.0.0.0")
        self.use_multicast = config.get("use_multicast", False)
        self.rigid_body_id = config.get("rigid_body_id", None)

        # State
        self.streaming_client = None
        self.latest_frame_data = {}
        self.latest_rigid_body_data = {}
        self.frame_number = -1
        self.is_streaming = False
        self.stream_callback: Optional[Callable] = None

        # Threading
        self._data_lock = threading.RLock()

        # Statistics
        self.frames_received = 0
        self.rigid_bodies_received = 0
        self.last_frame_time = 0

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="OptiTrackSource",
            version="1.0.0",
            description="OptiTrack motion capture data source",
            author="Robot Middleware Team",
            plugin_type=PluginType.DATA_SOURCE,
            dependencies=["capybarish.natnet"],
            tags={"optitrack", "mocap", "tracking", "position"},
        )

    def initialize(self) -> bool:
        """Initialize the OptiTrack data source."""
        try:
            # Import NatNet client (lazy import to handle missing dependencies)
            from capybarish.natnet.NatNetClient import NatNetClient

            # Create and configure NatNet client
            self.streaming_client = NatNetClient()
            self.streaming_client.set_client_address(self.client_address)
            self.streaming_client.set_server_address(self.server_address)
            self.streaming_client.set_use_multicast(self.use_multicast)
            self.streaming_client.set_print_level(0)  # Quiet mode

            # Set up callbacks
            self.streaming_client.new_frame_listener = self._on_new_frame
            self.streaming_client.rigid_body_listener = self._on_rigid_body_frame

            print(f"[{self.metadata.name}] Initialized with server {self.server_address}")
            return True

        except ImportError as e:
            self.last_error = f"NatNet client not available: {e}"
            return False
        except Exception as e:
            self.last_error = f"Initialization failed: {e}"
            return False

    def start(self) -> bool:
        """Start the OptiTrack data source."""
        try:
            if self.streaming_client is None:
                if not self.initialize():
                    return False

            # Start the NatNet client
            is_running = self.streaming_client.run()

            if is_running:
                print(f"[{self.metadata.name}] Started streaming from {self.server_address}")
                return True
            else:
                self.last_error = "Failed to start NatNet client"
                return False

        except Exception as e:
            self.last_error = f"Start failed: {e}"
            return False

    def stop(self) -> bool:
        """Stop the OptiTrack data source."""
        try:
            if self.streaming_client:
                self.streaming_client.shutdown()

            self.is_streaming = False
            print(f"[{self.metadata.name}] Stopped")
            return True

        except Exception as e:
            self.last_error = f"Stop failed: {e}"
            return False

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []

        if "server_address" in config:
            server_addr = config["server_address"]
            if not isinstance(server_addr, str) or not server_addr:
                errors.append("server_address must be a non-empty string")

        if "rigid_body_id" in config:
            body_id = config["rigid_body_id"]
            if body_id is not None and not isinstance(body_id, int):
                errors.append("rigid_body_id must be an integer or None")

        return errors

    def get_data(self) -> Dict[str, Any]:
        """Get the latest OptiTrack data."""
        with self._data_lock:
            current_time = time.time()

            data = {
                "optitrack_frame_number": self.frame_number,
                "optitrack_frames_received": self.frames_received,
                "optitrack_rigid_bodies_received": self.rigid_bodies_received,
                "optitrack_last_update": self.last_frame_time,
                "optitrack_data_age": current_time - self.last_frame_time
                if self.last_frame_time > 0
                else -1,
                "optitrack_server_address": self.server_address,
            }

            # Add frame data
            if self.latest_frame_data:
                data["optitrack_frame_data"] = self.latest_frame_data.copy()

            # Add rigid body data
            if self.latest_rigid_body_data:
                data["optitrack_rigid_bodies"] = self.latest_rigid_body_data.copy()

                # If a specific rigid body ID is configured, extract its data
                if (
                    self.rigid_body_id is not None
                    and self.rigid_body_id in self.latest_rigid_body_data
                ):
                    body_data = self.latest_rigid_body_data[self.rigid_body_id]
                    data.update(
                        {
                            "pos_world_opti": body_data.get("position", [0, 0, 0]),
                            "quat_world_opti": body_data.get("rotation", [0, 0, 0, 1]),
                            "optitrack_tracking_valid": body_data.get("tracking_valid", False),
                        }
                    )

            return data

    def supports_streaming(self) -> bool:
        """This data source supports streaming."""
        return True

    def start_streaming(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Start streaming OptiTrack data to a callback."""
        try:
            self.stream_callback = callback
            self.is_streaming = True

            # Start the OptiTrack client if not already started
            if not self.start():
                return False

            print(f"[{self.metadata.name}] Started streaming with callback")
            return True

        except Exception as e:
            self.last_error = f"Failed to start streaming: {e}"
            return False

    def stop_streaming(self) -> bool:
        """Stop streaming OptiTrack data."""
        try:
            self.is_streaming = False
            self.stream_callback = None
            print(f"[{self.metadata.name}] Stopped streaming")
            return True

        except Exception as e:
            self.last_error = f"Failed to stop streaming: {e}"
            return False

    def _on_new_frame(self, data_frame):
        """Callback for new frame data from NatNet."""
        with self._data_lock:
            self.frame_number = data_frame.get("frame_number", -1)
            self.latest_frame_data = data_frame.copy()
            self.frames_received += 1
            self.last_frame_time = time.time()

            # Call streaming callback if active
            if self.is_streaming and self.stream_callback:
                try:
                    stream_data = self.get_data()
                    self.stream_callback(stream_data)
                except Exception as e:
                    print(f"[{self.metadata.name}] Error in stream callback: {e}")

    def _on_rigid_body_frame(self, new_id, pos, rot):
        """Callback for rigid body data from NatNet."""
        with self._data_lock:
            # Store rigid body data
            self.latest_rigid_body_data[new_id] = {
                "id": new_id,
                "position": list(pos) if pos else [0, 0, 0],
                "rotation": list(rot) if rot else [0, 0, 0, 1],
                "tracking_valid": pos is not None and rot is not None,
                "timestamp": time.time(),
            }

            self.rigid_bodies_received += 1

            # Call streaming callback if active
            if self.is_streaming and self.stream_callback:
                try:
                    stream_data = self.get_data()
                    self.stream_callback(stream_data)
                except Exception as e:
                    print(f"[{self.metadata.name}] Error in stream callback: {e}")

    def get_rigid_body_data(self, body_id: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific rigid body."""
        with self._data_lock:
            return self.latest_rigid_body_data.get(body_id)

    def get_all_rigid_bodies(self) -> Dict[int, Dict[str, Any]]:
        """Get data for all tracked rigid bodies."""
        with self._data_lock:
            return self.latest_rigid_body_data.copy()

    def is_tracking_valid(self, body_id: Optional[int] = None) -> bool:
        """Check if tracking is valid for a specific body or any body."""
        with self._data_lock:
            if body_id is not None:
                body_data = self.latest_rigid_body_data.get(body_id)
                return body_data.get("tracking_valid", False) if body_data else False
            else:
                # Check if any rigid body has valid tracking
                return any(
                    body.get("tracking_valid", False)
                    for body in self.latest_rigid_body_data.values()
                )

    def get_frame_rate(self) -> float:
        """Get estimated frame rate based on recent frames."""
        # This is a simplified implementation
        # In practice, you'd maintain a sliding window of frame times
        if self.frames_received > 0 and self.last_frame_time > 0:
            # Rough estimate assuming constant rate
            return self.frames_received / max(time.time() - self.last_frame_time, 0.001)
        return 0.0
