"""
Dashboard server for real-time data visualization and control.

This module provides a UDP-based server for communicating with dashboard
visualization clients and simulation renderers. It handles:
- Receiving control commands from dashboard clients
- Broadcasting sensor data to connected clients
- Managing network connections and data serialization
- Error handling and connection management

The server uses MessagePack for efficient binary serialization and supports
multiple concurrent clients for different visualization purposes.

Network Architecture:
    - Dashboard Server: Port 6667 (this server)
    - Dashboard Client: Port 6668 (receives data, sends commands)
    - Simulation Renderer: Port 6669 (receives data only)

Typical usage example:
    from capybarish.dashboard_server import DashboardServer

    server = DashboardServer()

    # In main loop:
    enable, disable, calibrate, reset, positions = server.get_commands()
    server.send_data(sensor_data_dict)

    # Cleanup
    server.close()

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

import logging
import select
import socket
from typing import Dict, Any, Optional, Tuple, Union, List

import msgpack

# Network configuration constants
DEFAULT_SERVER_HOST = '127.0.0.1'
DASHBOARD_SERVER_PORT = 6667
DASHBOARD_CLIENT_PORT = 6668
RENDERER_CLIENT_PORT = 6669

# Socket configuration constants
DEFAULT_BUFFER_SIZE = 4096
DEFAULT_RECEIVE_BUFFER_SIZE = 1024
MAX_COMMAND_POLLING_ATTEMPTS = 100
SOCKET_TIMEOUT = 0.1

# Command field names
CMD_ENABLE = "enable"
CMD_DISABLE = "disable"
CMD_CALIBRATE = "calibrate"
CMD_RESET = "reset"
CMD_DEBUG_POSITIONS = "slide"

# Default command values
DEFAULT_COMMAND_VALUES = {
    CMD_ENABLE: 0,
    CMD_DISABLE: 0,
    CMD_CALIBRATE: 0,
    CMD_RESET: 0,
    CMD_DEBUG_POSITIONS: None
}


class DashboardServerError(Exception):
    """Base exception for dashboard server errors."""
    pass


class SocketSetupError(DashboardServerError):
    """Exception raised when socket setup fails."""
    pass


class DataTransmissionError(DashboardServerError):
    """Exception raised when data transmission fails."""
    pass


class DashboardServer:
    """UDP server for dashboard communication and data visualization.
    
    This class manages UDP socket communication between the robot system
    and visualization/control clients. It provides bidirectional communication:
    - Receives control commands from dashboard clients
    - Broadcasts sensor data to multiple visualization clients
    
    The server uses non-blocking sockets and MessagePack serialization for
    efficient real-time communication.
    
    Attributes:
        dashboard_socket: UDP socket for server communication
        dashboard_address: Address tuple for dashboard client
        renderer_address: Address tuple for simulation renderer
        is_connected: Connection status flag
    """
    
    def __init__(self, 
                 server_host: str = DEFAULT_SERVER_HOST,
                 server_port: int = DASHBOARD_SERVER_PORT,
                 dashboard_port: int = DASHBOARD_CLIENT_PORT,
                 renderer_port: int = RENDERER_CLIENT_PORT) -> None:
        """Initialize the dashboard server.
        
        Args:
            server_host: Host address for the server socket
            server_port: Port number for the server socket
            dashboard_port: Port number for dashboard client communication
            renderer_port: Port number for renderer client communication
            
        Raises:
            SocketSetupError: If socket initialization fails
        """
        self.server_host = server_host
        self.server_port = server_port
        self.dashboard_address = (server_host, dashboard_port)
        self.renderer_address = (server_host, renderer_port)
        
        self.dashboard_socket: Optional[socket.socket] = None
        self.is_connected = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        try:
            self._setup_socket()
            self.is_connected = True
            self.logger.info(f"Dashboard server initialized on {server_host}:{server_port}")
        except Exception as e:
            raise SocketSetupError(f"Failed to initialize dashboard server: {e}") from e
    
    def _setup_socket(self) -> None:
        """Setup and configure the UDP socket for communication.
        
        Raises:
            SocketSetupError: If socket setup fails
        """
        try:
            self.logger.info("Setting up dashboard socket...")
            
            # Create UDP socket
            self.dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Configure socket options
            self.dashboard_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, DEFAULT_BUFFER_SIZE)
            self.dashboard_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to server address
            server_address = (self.server_host, self.server_port)
            self.dashboard_socket.bind(server_address)
            
            # Set non-blocking mode for real-time operation
            self.dashboard_socket.setblocking(False)
            
            self.logger.info(f"Socket bound to {server_address}")
            self.logger.info(f"Dashboard client address: {self.dashboard_address}")
            self.logger.info(f"Renderer client address: {self.renderer_address}")
            
        except OSError as e:
            if self.dashboard_socket:
                self.dashboard_socket.close()
            raise SocketSetupError(f"Socket setup failed: {e}") from e
    
    def get_commands(self) -> Tuple[int, int, int, int, Optional[List[float]]]:
        """Receive and parse control commands from dashboard clients.
        
        Polls the socket for incoming command messages and parses them.
        Uses non-blocking I/O to avoid blocking the main control loop.
        
        Returns:
            Tuple containing:
                - enable: Motor enable command (0 or 1)
                - disable: Motor disable command (0 or 1) 
                - calibrate: Calibration command (0 or 1)
                - reset: Reset command (0 or 1)
                - debug_pos_list: List of debug positions or None
                
        Raises:
            DataTransmissionError: If command parsing fails
        """
        if not self.is_connected or not self.dashboard_socket:
            return tuple(DEFAULT_COMMAND_VALUES.values())
        
        received_data = None
        
        try:
            # Poll for incoming data with limited attempts
            for attempt in range(MAX_COMMAND_POLLING_ATTEMPTS):
                try:
                    data, client_address = self.dashboard_socket.recvfrom(DEFAULT_RECEIVE_BUFFER_SIZE)
                    received_data = data
                    self.logger.debug(f"Received command from {client_address}")
                except BlockingIOError:
                    # No data available, continue polling
                    break
                except OSError as e:
                    self.logger.warning(f"Socket error while receiving commands: {e}")
                    break
            
            if received_data is not None:
                try:
                    # Deserialize MessagePack data
                    parsed_data = msgpack.unpackb(received_data, raw=False)
                    
                    if not isinstance(parsed_data, dict):
                        raise ValueError("Received data is not a dictionary")
                    
                    # Extract command values with defaults
                    enable = parsed_data.get(CMD_ENABLE, DEFAULT_COMMAND_VALUES[CMD_ENABLE])
                    disable = parsed_data.get(CMD_DISABLE, DEFAULT_COMMAND_VALUES[CMD_DISABLE])
                    calibrate = parsed_data.get(CMD_CALIBRATE, DEFAULT_COMMAND_VALUES[CMD_CALIBRATE])
                    reset = parsed_data.get(CMD_RESET, DEFAULT_COMMAND_VALUES[CMD_RESET])
                    debug_pos_list = parsed_data.get(CMD_DEBUG_POSITIONS, DEFAULT_COMMAND_VALUES[CMD_DEBUG_POSITIONS])
                    
                    self.logger.debug(f"Parsed commands - Enable: {enable}, Disable: {disable}, "
                                    f"Calibrate: {calibrate}, Reset: {reset}, Positions: {debug_pos_list}")
                    
                    return enable, disable, calibrate, reset, debug_pos_list
                    
                except (msgpack.exceptions.ExtraData, 
                        msgpack.exceptions.InvalidData,
                        ValueError) as e:
                    self.logger.error(f"Failed to parse command data: {e}")
                    raise DataTransmissionError(f"Command parsing failed: {e}") from e
            
            # No data received, return defaults
            return tuple(DEFAULT_COMMAND_VALUES.values())
            
        except Exception as e:
            self.logger.error(f"Error receiving commands: {e}")
            # Return default values on error to maintain system stability
            return tuple(DEFAULT_COMMAND_VALUES.values())
    
    def send_data(self, observable_data: Dict[str, Any]) -> bool:
        """Send sensor data to connected dashboard and renderer clients.
        
        Serializes the provided data using MessagePack and broadcasts it
        to both dashboard and renderer clients via UDP.
        
        Args:
            observable_data: Dictionary containing sensor data and system status
            
        Returns:
            True if data was sent successfully to at least one client,
            False otherwise
            
        Raises:
            DataTransmissionError: If data serialization fails
        """
        if not self.is_connected or not self.dashboard_socket:
            self.logger.warning("Cannot send data: server not connected")
            return False
        
        try:
            # Serialize data using MessagePack
            try:
                serialized_data = msgpack.packb(observable_data)
            except (TypeError, ValueError) as e:
                raise DataTransmissionError(f"Data serialization failed: {e}") from e
            
            success_count = 0
            
            # Send to dashboard client
            try:
                self.dashboard_socket.sendto(serialized_data, self.dashboard_address)
                success_count += 1
                self.logger.debug(f"Data sent to dashboard client at {self.dashboard_address}")
            except OSError as e:
                self.logger.warning(f"Failed to send data to dashboard client: {e}")
            
            # Send to renderer client
            try:
                self.dashboard_socket.sendto(serialized_data, self.renderer_address)
                success_count += 1
                self.logger.debug(f"Data sent to renderer client at {self.renderer_address}")
            except OSError as e:
                self.logger.warning(f"Failed to send data to renderer client: {e}")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending data: {e}")
            return False
    
    def close(self) -> None:
        """Close the socket and cleanup resources.
        
        Properly closes the UDP socket and marks the server as disconnected.
        This method should be called when shutting down the server.
        """
        if self.dashboard_socket:
            try:
                self.dashboard_socket.close()
                self.logger.info("Dashboard socket closed")
            except OSError as e:
                self.logger.warning(f"Error closing socket: {e}")
            finally:
                self.dashboard_socket = None
                self.is_connected = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure socket cleanup."""
        self.close()
    
    @property
    def connection_status(self) -> Dict[str, Any]:
        """Get current connection status information.
        
        Returns:
            Dictionary containing connection status details
        """
        return {
            "is_connected": self.is_connected,
            "server_address": f"{self.server_host}:{self.server_port}",
            "dashboard_client": self.dashboard_address,
            "renderer_client": self.renderer_address,
            "socket_active": self.dashboard_socket is not None
        }