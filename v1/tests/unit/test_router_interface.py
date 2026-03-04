import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from src.hardware.router_interface import RouterInterface, RouterConnectionError


class TestRouterInterface:
    """Test suite for Router Interface following London School TDD principles"""

    @pytest.fixture
    def mock_config(self):
        """Configuration for router interface"""
        return {
            'host': '192.168.1.1',
            'port': 22,
            'username': 'admin',
            'password': 'password',
            'command_timeout': 30,
            'connection_timeout': 10,
            'max_retries': 3,
            'retry_delay': 0.0,
        }

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing"""
        return Mock()

    @pytest.fixture
    def router_interface(self, mock_config, mock_logger):
        """Create router interface instance for testing"""
        return RouterInterface(mock_config, logger=mock_logger)

    @pytest.fixture
    def mock_ssh_client(self):
        """Mock asyncssh SSH client for testing"""
        mock_client = Mock()
        mock_client.close = Mock()
        return mock_client

    def test_interface_initialization_creates_correct_configuration(self, mock_config, mock_logger):
        """Test that router interface initializes with correct configuration"""
        # Act
        interface = RouterInterface(mock_config, logger=mock_logger)

        # Assert
        assert interface is not None
        assert interface.host == mock_config['host']
        assert interface.port == mock_config['port']
        assert interface.username == mock_config['username']
        assert interface.password == mock_config['password']
        assert interface.command_timeout == mock_config['command_timeout']
        assert interface.connection_timeout == mock_config['connection_timeout']
        assert interface.max_retries == mock_config['max_retries']
        assert not interface.is_connected

    @pytest.mark.asyncio
    async def test_connect_establishes_ssh_connection(self, router_interface, mock_ssh_client):
        """Test that connect method establishes SSH connection"""
        # Arrange
        with patch('src.hardware.router_interface.asyncssh.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ssh_client

            # Act
            result = await router_interface.connect()

            # Assert
            assert result is True
            assert router_interface.is_connected is True
            assert router_interface.ssh_client == mock_ssh_client
            mock_connect.assert_called_once_with(
                router_interface.host,
                port=router_interface.port,
                username=router_interface.username,
                password=router_interface.password,
                connect_timeout=router_interface.connection_timeout
            )

    @pytest.mark.asyncio
    async def test_connect_handles_connection_failure(self, router_interface):
        """Test that connect method handles connection failures gracefully"""
        # Arrange
        with patch('src.hardware.router_interface.asyncssh.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            # Act
            result = await router_interface.connect()

            # Assert
            assert result is False
            assert router_interface.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_ssh_connection(self, router_interface, mock_ssh_client):
        """Test that disconnect method closes SSH connection"""
        # Arrange
        router_interface.is_connected = True
        router_interface.ssh_client = mock_ssh_client

        # Act
        await router_interface.disconnect()

        # Assert
        assert router_interface.is_connected is False
        mock_ssh_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_command_runs_ssh_command(self, router_interface, mock_ssh_client):
        """Test that execute_command runs SSH commands correctly"""
        # Arrange
        mock_result = Mock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        router_interface.is_connected = True
        router_interface.ssh_client = mock_ssh_client

        with patch.object(mock_ssh_client, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            # Act
            result = await router_interface.execute_command("test command")

            # Assert
            assert result == "command output"
            mock_run.assert_called_once_with("test command", timeout=router_interface.command_timeout)

    @pytest.mark.asyncio
    async def test_execute_command_handles_command_errors(self, router_interface, mock_ssh_client):
        """Test that execute_command handles command errors"""
        # Arrange
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = "command error"
        mock_result.returncode = 1

        router_interface.is_connected = True
        router_interface.ssh_client = mock_ssh_client

        with patch.object(mock_ssh_client, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            # Act & Assert
            with pytest.raises(RouterConnectionError):
                await router_interface.execute_command("failing command")

    @pytest.mark.asyncio
    async def test_execute_command_requires_connection(self, router_interface):
        """Test that execute_command requires active connection"""
        # Arrange
        router_interface.is_connected = False

        # Act & Assert
        with pytest.raises(RouterConnectionError):
            await router_interface.execute_command("test command")

    @pytest.mark.asyncio
    async def test_get_router_status_retrieves_system_information(self, router_interface):
        """Test that get_router_status retrieves router system information"""
        # Arrange
        with patch.object(router_interface, 'execute_command', new_callable=AsyncMock) as mock_exec:
            with patch.object(router_interface, '_parse_status_response', return_value={
                'cpu_usage': 25.5,
                'memory_usage': 60.2,
                'wifi_status': 'active',
            }) as mock_parse:
                mock_exec.return_value = "status response"

                # Act
                info = await router_interface.get_router_status()

                # Assert
                assert info is not None
                assert isinstance(info, dict)
                assert 'cpu_usage' in info
                assert 'memory_usage' in info

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self, router_interface):
        """Test that health_check returns True when router is healthy"""
        # Arrange
        with patch.object(router_interface, 'execute_command', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "pong"

            # Act
            result = await router_interface.health_check()

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_configure_csi_monitoring_configures_wifi_monitoring(self, router_interface):
        """Test that configure_csi_monitoring configures WiFi monitoring"""
        # Arrange
        config = {'channel': 6, 'bandwidth': 20, 'sample_rate': 100}

        with patch.object(router_interface, 'execute_command', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "CSI monitoring configured"

            # Act
            result = await router_interface.configure_csi_monitoring(config)

            # Assert
            assert result is True
            mock_exec.assert_called_once()

    def test_interface_validates_configuration(self):
        """Test that router interface validates configuration parameters"""
        # Arrange — missing required 'host' field
        invalid_config = {
            'username': 'admin',
            'password': 'password',
            'port': 22,
        }

        # Act & Assert
        with pytest.raises(ValueError):
            RouterInterface(invalid_config)

    @pytest.mark.asyncio
    async def test_interface_implements_retry_logic(self, router_interface, mock_ssh_client):
        """Test that interface implements retry logic for failed command operations"""
        # Arrange
        mock_success = Mock()
        mock_success.stdout = "success"
        mock_success.stderr = ""
        mock_success.returncode = 0

        router_interface.is_connected = True
        router_interface.ssh_client = mock_ssh_client

        with patch.object(mock_ssh_client, 'run', new_callable=AsyncMock) as mock_run:
            # Fail twice with ConnectionError (triggers retry), then succeed
            mock_run.side_effect = [
                ConnectionError("Temp failure"),
                ConnectionError("Temp failure"),
                mock_success
            ]

            # Act
            result = await router_interface.execute_command("test command")

            # Assert
            assert result == "success"
            assert mock_run.call_count == 3
