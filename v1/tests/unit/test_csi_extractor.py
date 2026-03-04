import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from src.hardware.csi_extractor import (
    CSIExtractor,
    CSIExtractionError,
    CSIParseError,
    CSIValidationError,
    CSIData,
    ESP32CSIParser,
    RouterCSIParser,
)
from datetime import datetime, timezone


class TestCSIExtractor:
    """Test suite for CSI Extractor following London School TDD principles"""

    @pytest.fixture
    def mock_config(self):
        """Configuration for CSI extractor"""
        return {
            'hardware_type': 'router',
            'sampling_rate': 1000,
            'buffer_size': 1024,
            'timeout': 5.0,
        }

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing"""
        return Mock()

    @pytest.fixture
    def csi_extractor(self, mock_config, mock_logger):
        """Create CSI extractor instance for testing"""
        return CSIExtractor(mock_config, logger=mock_logger)

    @pytest.fixture
    def mock_csi_data(self):
        """Generate synthetic CSI data for testing"""
        return CSIData(
            timestamp=datetime.now(timezone.utc),
            amplitude=np.random.uniform(0.1, 2.0, (3, 56)),
            phase=np.random.uniform(-np.pi, np.pi, (3, 56)),
            frequency=2.4e9,
            bandwidth=20e6,
            num_subcarriers=56,
            num_antennas=3,
            snr=15.5,
            metadata={'source': 'router'}
        )

    def test_extractor_initialization_creates_correct_configuration(self, mock_config, mock_logger):
        """Test that CSI extractor initializes with correct configuration"""
        # Act
        extractor = CSIExtractor(mock_config, logger=mock_logger)

        # Assert
        assert extractor is not None
        assert extractor.config == mock_config
        assert extractor.hardware_type == mock_config['hardware_type']
        assert extractor.sampling_rate == mock_config['sampling_rate']
        assert extractor.buffer_size == mock_config['buffer_size']
        assert extractor.timeout == mock_config['timeout']
        assert not extractor.is_connected

    def test_extractor_creates_router_parser_for_router_type(self, mock_config, mock_logger):
        """Test that router parser is created for hardware_type='router'"""
        extractor = CSIExtractor(mock_config, logger=mock_logger)
        assert isinstance(extractor.parser, RouterCSIParser)

    def test_extractor_creates_esp32_parser_for_esp32_type(self, mock_logger):
        """Test that ESP32 parser is created for hardware_type='esp32'"""
        config = {
            'hardware_type': 'esp32',
            'sampling_rate': 100,
            'buffer_size': 512,
            'timeout': 5.0,
        }
        extractor = CSIExtractor(config, logger=mock_logger)
        assert isinstance(extractor.parser, ESP32CSIParser)

    @pytest.mark.asyncio
    async def test_connect_establishes_hardware_connection(self, csi_extractor):
        """Test that connect establishes hardware connection"""
        # Arrange
        with patch.object(csi_extractor, '_establish_hardware_connection', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True

            # Act
            result = await csi_extractor.connect()

            # Assert
            assert result is True
            assert csi_extractor.is_connected is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_handles_hardware_failure(self, csi_extractor):
        """Test that connect handles hardware connection failure"""
        # Arrange
        with patch.object(csi_extractor, '_establish_hardware_connection', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Hardware not found")

            # Act
            result = await csi_extractor.connect()

            # Assert
            assert result is False
            assert csi_extractor.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_hardware_connection(self, csi_extractor):
        """Test that disconnect closes hardware connection"""
        # Arrange
        csi_extractor.is_connected = True

        with patch.object(csi_extractor, '_close_hardware_connection', new_callable=AsyncMock) as mock_close:
            # Act
            await csi_extractor.disconnect()

            # Assert
            assert csi_extractor.is_connected is False
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_csi_returns_valid_csi_data(self, csi_extractor, mock_csi_data):
        """Test that extract_csi returns valid CSI data"""
        # Arrange
        csi_extractor.is_connected = True

        with patch.object(csi_extractor, '_read_raw_data', new_callable=AsyncMock) as mock_read:
            with patch.object(csi_extractor.parser, 'parse', return_value=mock_csi_data):
                mock_read.return_value = b"raw_csi_data"

                # Act
                result = await csi_extractor.extract_csi()

                # Assert
                assert result is not None
                assert isinstance(result, CSIData)
                assert result == mock_csi_data

    @pytest.mark.asyncio
    async def test_extract_csi_requires_active_connection(self, csi_extractor):
        """Test that extract_csi requires active connection"""
        # Arrange
        csi_extractor.is_connected = False

        # Act & Assert
        with pytest.raises(CSIParseError, match="Not connected to hardware"):
            await csi_extractor.extract_csi()

    @pytest.mark.asyncio
    async def test_extract_csi_handles_timeout(self, csi_extractor, mock_csi_data):
        """Test that extract_csi handles extraction failures after retries"""
        # Arrange
        csi_extractor.is_connected = True
        csi_extractor.retry_attempts = 1

        with patch.object(csi_extractor, '_read_raw_data', new_callable=AsyncMock) as mock_read:
            mock_read.side_effect = ConnectionError("Connection lost")

            # Act & Assert
            with pytest.raises(CSIParseError, match="Extraction failed after"):
                await csi_extractor.extract_csi()

    def test_validate_csi_data_accepts_valid_data(self, csi_extractor, mock_csi_data):
        """Test that validate_csi_data accepts valid CSI data"""
        # Act
        result = csi_extractor.validate_csi_data(mock_csi_data)

        # Assert
        assert result is True

    def test_validate_csi_data_rejects_empty_amplitude(self, csi_extractor):
        """Test that validate_csi_data rejects empty amplitude"""
        invalid_data = CSIData(
            timestamp=datetime.now(timezone.utc),
            amplitude=np.array([]),
            phase=np.random.rand(3, 56),
            frequency=2.4e9,
            bandwidth=20e6,
            num_subcarriers=56,
            num_antennas=3,
            snr=15.5,
            metadata={}
        )
        with pytest.raises(CSIValidationError, match="Empty amplitude data"):
            csi_extractor.validate_csi_data(invalid_data)

    def test_validate_csi_data_rejects_invalid_frequency(self, csi_extractor):
        """Test that validate_csi_data rejects invalid frequency"""
        invalid_data = CSIData(
            timestamp=datetime.now(timezone.utc),
            amplitude=np.random.rand(3, 56),
            phase=np.random.rand(3, 56),
            frequency=0,
            bandwidth=20e6,
            num_subcarriers=56,
            num_antennas=3,
            snr=15.5,
            metadata={}
        )
        with pytest.raises(CSIValidationError, match="Invalid frequency"):
            csi_extractor.validate_csi_data(invalid_data)

    @pytest.mark.asyncio
    async def test_start_streaming_calls_callback(self, csi_extractor, mock_csi_data):
        """Test that start_streaming calls the callback with CSI data"""
        # Arrange
        csi_extractor.is_connected = True
        callback = Mock()

        with patch.object(csi_extractor, 'extract_csi', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_csi_data

            # Start streaming and stop it after brief run
            streaming_task = asyncio.create_task(csi_extractor.start_streaming(callback))
            await asyncio.sleep(0.05)
            csi_extractor.stop_streaming()
            await streaming_task

            callback.assert_called()

    def test_stop_streaming_sets_flag_to_false(self, csi_extractor):
        """Test that stop_streaming sets is_streaming to False"""
        # Arrange
        csi_extractor.is_streaming = True

        # Act
        csi_extractor.stop_streaming()

        # Assert
        assert csi_extractor.is_streaming is False

    def test_extractor_validates_configuration_missing_fields(self, mock_logger):
        """Test that CSI extractor validates required configuration parameters"""
        # Arrange — missing required fields
        invalid_config = {
            'interface': '',
            'channel': 6,
            'bandwidth': 20,
        }

        # Act & Assert
        with pytest.raises(ValueError):
            CSIExtractor(invalid_config, logger=mock_logger)

    def test_extractor_validates_positive_sampling_rate(self, mock_logger):
        """Test that CSI extractor rejects non-positive sampling_rate"""
        invalid_config = {
            'hardware_type': 'esp32',
            'sampling_rate': -1,
            'buffer_size': 1024,
            'timeout': 5.0,
        }
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            CSIExtractor(invalid_config, logger=mock_logger)

    def test_extractor_raises_for_unsupported_hardware(self, mock_logger):
        """Test that CSI extractor raises for unsupported hardware type"""
        invalid_config = {
            'hardware_type': 'unsupported_device',
            'sampling_rate': 100,
            'buffer_size': 1024,
            'timeout': 5.0,
        }
        with pytest.raises(ValueError, match="Unsupported hardware type"):
            CSIExtractor(invalid_config, logger=mock_logger)

    @pytest.mark.asyncio
    async def test_extract_csi_implements_retry_on_connection_error(self, csi_extractor, mock_csi_data):
        """Test that extract_csi implements retry logic for ConnectionError failures"""
        # Arrange
        csi_extractor.is_connected = True

        with patch.object(csi_extractor, '_read_raw_data', new_callable=AsyncMock) as mock_read:
            with patch.object(csi_extractor.parser, 'parse', return_value=mock_csi_data):
                # Fail twice, succeed on third
                mock_read.side_effect = [
                    ConnectionError("Temp failure"),
                    ConnectionError("Temp failure"),
                    b"raw_data",
                ]

                # Act
                result = await csi_extractor.extract_csi()

                # Assert
                assert result == mock_csi_data
                assert mock_read.call_count == 3
