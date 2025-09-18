"""
Tests for the OptiTrack source plugin.

These tests verify OptiTrack data source functionality including
connection management, data streaming, and rigid body tracking
without requiring actual OptiTrack hardware.
"""

import time
from unittest.mock import Mock, patch

import pytest

from plugins.optitrack_source import OptiTrackSource
from tests.conftest import MockNatNetClient


class TestOptiTrackSource:
    """Test OptiTrack source plugin functionality."""

    def test_optitrack_source_creation(self):
        """Test creating an OptiTrack source plugin."""
        config = {
            "server_address": "192.168.1.100",
            "client_address": "192.168.1.50",
            "use_multicast": True,
            "rigid_body_id": 5,
        }
        source = OptiTrackSource(config)

        assert source.server_address == "192.168.1.100"
        assert source.client_address == "192.168.1.50"
        assert source.use_multicast == True
        assert source.rigid_body_id == 5
        assert source.frames_received == 0
        assert source.rigid_bodies_received == 0

    def test_optitrack_source_defaults(self):
        """Test OptiTrack source with default configuration."""
        source = OptiTrackSource({})

        assert source.server_address == "129.105.73.172"
        assert source.client_address == "0.0.0.0"
        assert source.use_multicast == False
        assert source.rigid_body_id is None

    def test_metadata(self):
        """Test plugin metadata."""
        source = OptiTrackSource({})
        metadata = source.metadata

        assert metadata.name == "OptiTrackSource"
        assert metadata.version == "1.0.0"
        assert metadata.description == "OptiTrack motion capture data source"
        assert "optitrack" in metadata.tags
        assert "mocap" in metadata.tags
        assert "tracking" in metadata.tags

    def test_config_validation(self):
        """Test configuration validation."""
        source = OptiTrackSource({})

        # Valid config
        valid_config = {
            "server_address": "192.168.1.100",
            "rigid_body_id": 1,
        }
        errors = source.validate_config(valid_config)
        assert errors == []

        # Invalid server_address
        invalid_config1 = {"server_address": ""}
        errors1 = source.validate_config(invalid_config1)
        assert "server_address must be a non-empty string" in errors1

        invalid_config2 = {"server_address": 123}
        errors2 = source.validate_config(invalid_config2)
        assert "server_address must be a non-empty string" in errors2

        # Invalid rigid_body_id
        invalid_config3 = {"rigid_body_id": "not_an_int"}
        errors3 = source.validate_config(invalid_config3)
        assert "rigid_body_id must be an integer or None" in errors3

        # None is valid for rigid_body_id
        valid_config_none = {"rigid_body_id": None}
        errors_none = source.validate_config(valid_config_none)
        assert errors_none == []

    def test_initialize_success(self):
        """Test successful initialization of OptiTrack source."""
        with patch("capybarish.natnet.NatNetClient.NatNetClient") as mock_natnet:
            mock_client = Mock()
            mock_natnet.return_value = mock_client
            
            source = OptiTrackSource({
                "server_address": "192.168.1.100",
                "client_address": "192.168.1.50",
            })

            result = source.initialize()
            assert result == True
            assert source.streaming_client is not None

            # Verify client methods were called correctly  
            mock_client.set_server_address.assert_called_once_with("192.168.1.100")
            mock_client.set_client_address.assert_called_once_with("192.168.1.50")
            mock_client.set_use_multicast.assert_called_once_with(False)
            mock_client.set_print_level.assert_called_once_with(0)

    def test_initialize_import_error(self):
        """Test initialization failure when NatNet import fails."""
        with patch("capybarish.natnet.NatNetClient.NatNetClient", side_effect=ImportError("NatNet not found")):
            source = OptiTrackSource({})
            result = source.initialize()
            assert result == False
            assert source.last_error is not None
            assert "NatNet client not available" in source.last_error

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_initialize_exception(self, mock_natnet_class):
        """Test initialization with exception."""
        mock_natnet_class.side_effect = Exception("Test exception")

        source = OptiTrackSource({})
        result = source.initialize()
        assert result == False
        assert source.last_error is not None
        assert "Initialization failed" in source.last_error

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_start_success(self, mock_natnet_class):
        """Test successful start."""
        mock_client = MockNatNetClient()
        mock_client.run = Mock(return_value=True)
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({})
        
        # Initialize first
        assert source.initialize() == True
        
        # Start
        result = source.start()
        assert result == True
        mock_client.run.assert_called_once()

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_start_failure(self, mock_natnet_class):
        """Test start failure."""
        mock_client = MockNatNetClient()
        mock_client.run = Mock(return_value=False)
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({})
        source.initialize()

        result = source.start()
        assert result == False
        assert source.last_error is not None
        assert "Failed to start NatNet client" in source.last_error

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_stop(self, mock_natnet_class):
        """Test stopping the source."""
        mock_client = MockNatNetClient()
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({})
        source.initialize()

        result = source.stop()
        assert result == True
        assert source.is_streaming == False

    def test_supports_streaming(self):
        """Test that streaming is supported."""
        source = OptiTrackSource({})
        assert source.supports_streaming() == True

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_streaming_workflow(self, mock_natnet_class):
        """Test streaming workflow."""
        mock_client = MockNatNetClient()
        mock_client.run = Mock(return_value=True)
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({})
        callback = Mock()

        # Start streaming
        result = source.start_streaming(callback)
        assert result == True
        assert source.is_streaming == True
        assert source.stream_callback == callback

        # Stop streaming
        result = source.stop_streaming()
        assert result == True
        assert source.is_streaming == False
        assert source.stream_callback is None

    def test_get_data_initial(self):
        """Test getting data when no data has been received."""
        source = OptiTrackSource({
            "server_address": "test_server",
            "rigid_body_id": 1,
        })

        data = source.get_data()

        assert data["optitrack_frame_number"] == -1
        assert data["optitrack_frames_received"] == 0
        assert data["optitrack_rigid_bodies_received"] == 0
        assert data["optitrack_server_address"] == "test_server"
        assert data["optitrack_data_age"] == -1  # No data received yet

    def test_frame_data_callback(self):
        """Test handling of frame data callbacks."""
        source = OptiTrackSource({"rigid_body_id": 1})

        # Simulate frame callback
        frame_data = {"frame_number": 12345, "timestamp": time.time()}
        source._on_new_frame(frame_data)

        # Check that data was stored
        assert source.frame_number == 12345
        assert source.frames_received == 1
        assert source.latest_frame_data == frame_data

        # Get data
        data = source.get_data()
        assert data["optitrack_frame_number"] == 12345
        assert data["optitrack_frames_received"] == 1
        assert "optitrack_frame_data" in data

    def test_rigid_body_callback(self):
        """Test handling of rigid body callbacks."""
        source = OptiTrackSource({"rigid_body_id": 1})

        # Simulate rigid body callback
        body_id = 1
        position = [1.5, 2.5, 0.8]
        rotation = [0.1, 0.2, 0.3, 0.9]

        source._on_rigid_body_frame(body_id, position, rotation)

        # Check that data was stored
        assert source.rigid_bodies_received == 1
        assert body_id in source.latest_rigid_body_data

        body_data = source.latest_rigid_body_data[body_id]
        assert body_data["id"] == body_id
        assert body_data["position"] == position
        assert body_data["rotation"] == rotation
        assert body_data["tracking_valid"] == True

        # Get data
        data = source.get_data()
        assert data["optitrack_rigid_bodies_received"] == 1
        assert "optitrack_rigid_bodies" in data
        assert body_id in data["optitrack_rigid_bodies"]

        # Check specific rigid body data extraction
        assert "pos_world_opti" in data
        assert "quat_world_opti" in data
        assert data["pos_world_opti"] == position
        assert data["quat_world_opti"] == rotation

    def test_rigid_body_callback_invalid_data(self):
        """Test handling of rigid body callback with invalid data."""
        source = OptiTrackSource({})

        # Simulate callback with None data (tracking lost)
        body_id = 1
        source._on_rigid_body_frame(body_id, None, None)

        body_data = source.latest_rigid_body_data[body_id]
        assert body_data["position"] == [0, 0, 0]
        assert body_data["rotation"] == [0, 0, 0, 1]
        assert body_data["tracking_valid"] == False

    def test_multiple_rigid_bodies(self):
        """Test handling multiple rigid bodies."""
        source = OptiTrackSource({})

        # Add multiple rigid bodies
        bodies = [
            (1, [1.0, 2.0, 0.5], [0, 0, 0, 1]),
            (2, [2.0, 3.0, 1.0], [0.1, 0, 0, 0.9]),
            (3, [3.0, 1.0, 1.5], [0, 0.1, 0, 0.9]),
        ]

        for body_id, pos, rot in bodies:
            source._on_rigid_body_frame(body_id, pos, rot)

        # Check all bodies are tracked
        assert len(source.latest_rigid_body_data) == 3
        assert source.rigid_bodies_received == 3

        # Get data
        data = source.get_data()
        rigid_bodies = data["optitrack_rigid_bodies"]
        assert len(rigid_bodies) == 3

        for body_id, pos, rot in bodies:
            assert body_id in rigid_bodies
            assert rigid_bodies[body_id]["position"] == pos
            assert rigid_bodies[body_id]["rotation"] == rot

    def test_get_rigid_body_data(self):
        """Test getting specific rigid body data."""
        source = OptiTrackSource({})

        # Add rigid body
        body_id = 5
        position = [1.0, 2.0, 3.0]
        rotation = [0, 0, 0, 1]
        source._on_rigid_body_frame(body_id, position, rotation)

        # Get specific body data
        body_data = source.get_rigid_body_data(body_id)
        assert body_data is not None
        assert body_data["id"] == body_id
        assert body_data["position"] == position

        # Get non-existent body
        non_existent = source.get_rigid_body_data(999)
        assert non_existent is None

    def test_get_all_rigid_bodies(self):
        """Test getting all rigid body data."""
        source = OptiTrackSource({})

        # Add multiple bodies
        source._on_rigid_body_frame(1, [1, 0, 0], [0, 0, 0, 1])
        source._on_rigid_body_frame(2, [0, 1, 0], [0, 0, 0, 1])

        all_bodies = source.get_all_rigid_bodies()
        assert len(all_bodies) == 2
        assert 1 in all_bodies
        assert 2 in all_bodies

    def test_is_tracking_valid(self):
        """Test tracking validity checks."""
        source = OptiTrackSource({})

        # No tracking initially
        assert source.is_tracking_valid() == False
        assert source.is_tracking_valid(1) == False

        # Add valid tracking
        source._on_rigid_body_frame(1, [1, 2, 3], [0, 0, 0, 1])
        assert source.is_tracking_valid() == True
        assert source.is_tracking_valid(1) == True
        assert source.is_tracking_valid(2) == False  # Body 2 doesn't exist

        # Add invalid tracking
        source._on_rigid_body_frame(2, None, None)
        assert source.is_tracking_valid() == True  # Body 1 still valid
        assert source.is_tracking_valid(2) == False  # Body 2 invalid

    def test_streaming_callback_integration(self):
        """Test streaming callback integration."""
        source = OptiTrackSource({"rigid_body_id": 1})
        
        # Set up streaming callback
        callback_data = []
        def test_callback(data):
            callback_data.append(data)

        source.is_streaming = True
        source.stream_callback = test_callback

        # Simulate frame and rigid body data
        source._on_new_frame({"frame_number": 100})
        assert len(callback_data) == 1

        source._on_rigid_body_frame(1, [1, 2, 3], [0, 0, 0, 1])
        assert len(callback_data) == 2

        # Check that callback received proper data
        last_data = callback_data[-1]
        assert "optitrack_frame_number" in last_data
        assert "pos_world_opti" in last_data

    def test_streaming_callback_error_handling(self):
        """Test error handling in streaming callbacks."""
        source = OptiTrackSource({})
        
        # Set up failing callback
        def failing_callback(data):
            raise Exception("Callback failed")

        source.is_streaming = True
        source.stream_callback = failing_callback

        # Should not raise exception, should handle gracefully
        source._on_new_frame({"frame_number": 100})
        # No assertion needed - just verifying no exception is raised

    def test_data_age_calculation(self):
        """Test data age calculation."""
        source = OptiTrackSource({})

        # Initially no data
        data = source.get_data()
        assert data["optitrack_data_age"] == -1

        # Add frame data
        source._on_new_frame({"frame_number": 100})
        
        # Small delay to ensure age > 0
        time.sleep(0.01)
        
        data = source.get_data()
        assert data["optitrack_data_age"] >= 0
        assert data["optitrack_data_age"] < 1.0  # Should be very recent

    def test_get_frame_rate_estimation(self):
        """Test frame rate estimation."""
        source = OptiTrackSource({})

        # Initially no frames
        frame_rate = source.get_frame_rate()
        assert frame_rate == 0.0

        # Add frame
        source._on_new_frame({"frame_number": 1})
        
        # Still need time to pass for meaningful rate
        frame_rate = source.get_frame_rate()
        assert frame_rate >= 0.0  # Should be non-negative


class TestOptiTrackSourceIntegration:
    """Integration tests for OptiTrack source."""

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_full_workflow(self, mock_natnet_class):
        """Test complete OptiTrack source workflow."""
        mock_client = MockNatNetClient()
        mock_client.run = Mock(return_value=True)
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({
            "server_address": "192.168.1.100",
            "rigid_body_id": 1,
        })

        # Full lifecycle
        assert source.initialize() == True
        assert source.start() == True

        # Simulate receiving data
        source._on_new_frame({"frame_number": 12345, "timestamp": time.time()})
        source._on_rigid_body_frame(1, [1.5, 2.5, 0.8], [0.1, 0.2, 0.3, 0.9])

        # Get and verify data
        data = source.get_data()
        assert data["optitrack_frame_number"] == 12345
        assert data["pos_world_opti"] == [1.5, 2.5, 0.8]
        assert data["quat_world_opti"] == [0.1, 0.2, 0.3, 0.9]
        assert data["optitrack_tracking_valid"] == True

        # Stop
        assert source.stop() == True

    @patch("capybarish.natnet.NatNetClient.NatNetClient")
    def test_realistic_motion_capture_data(self, mock_natnet_class):
        """Test with realistic motion capture data sequence."""
        mock_client = MockNatNetClient()
        mock_client.run = Mock(return_value=True)
        mock_natnet_class.return_value = mock_client

        source = OptiTrackSource({"rigid_body_id": 1})
        source.initialize()
        source.start()

        # Simulate a sequence of motion capture frames
        motion_sequence = [
            # Object starts at origin
            (1, [0.0, 0.0, 0.0], [0, 0, 0, 1]),
            # Object moves in X direction
            (1, [0.1, 0.0, 0.0], [0, 0, 0, 1]),
            (1, [0.2, 0.0, 0.0], [0, 0, 0, 1]),
            # Object rotates while moving
            (1, [0.3, 0.0, 0.0], [0.1, 0, 0, 0.995]),
            (1, [0.4, 0.1, 0.0], [0.2, 0, 0, 0.98]),
            # Tracking lost temporarily
            (1, None, None),
            # Tracking restored
            (1, [0.5, 0.2, 0.1], [0.3, 0, 0, 0.954]),
        ]

        results = []
        for frame_num, (body_id, pos, rot) in enumerate(motion_sequence):
            # Send frame data
            source._on_new_frame({"frame_number": frame_num + 1})
            
            # Send rigid body data
            source._on_rigid_body_frame(body_id, pos, rot)
            
            # Collect data
            data = source.get_data()
            results.append(data)

        # Verify data sequence
        assert len(results) == len(motion_sequence)

        # Check frame progression
        for i, result in enumerate(results):
            assert result["optitrack_frame_number"] == i + 1

        # Check position progression (excluding lost tracking frame)
        valid_results = [r for i, r in enumerate(results) if motion_sequence[i][1] is not None]
        
        for i in range(1, len(valid_results)):
            prev_pos = valid_results[i-1]["pos_world_opti"]
            curr_pos = valid_results[i]["pos_world_opti"]
            
            # X position should increase
            assert curr_pos[0] >= prev_pos[0]

        # Check tracking validity
        for i, result in enumerate(results):
            expected_valid = motion_sequence[i][1] is not None
            assert result["optitrack_tracking_valid"] == expected_valid

        # Verify final statistics
        final_data = results[-1]
        assert final_data["optitrack_frames_received"] == len(motion_sequence)
        assert final_data["optitrack_rigid_bodies_received"] == len(motion_sequence)