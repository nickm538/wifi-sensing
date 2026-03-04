"""
WiFi-DensePose API Package
==========================

A comprehensive system for WiFi-based human pose estimation using CSI data
and DensePose neural networks.

This package provides:
- Real-time CSI data collection from WiFi routers
- Advanced signal processing and phase sanitization
- DensePose neural network integration for pose estimation
- RESTful API for data access and control
- Background task management for data processing
- Comprehensive monitoring and logging

Example usage:
    >>> from src.app import app
    >>> from src.config.settings import get_settings
    >>> 
    >>> settings = get_settings()
    >>> # Run with: uvicorn src.app:app --host 0.0.0.0 --port 8000

For CLI usage:
    $ wifi-densepose start --host 0.0.0.0 --port 8000
    $ wifi-densepose status
    $ wifi-densepose stop

Author: WiFi-DensePose Team
License: MIT
"""

__version__ = "1.2.0"
__author__ = "WiFi-DensePose Team"
__email__ = "team@wifi-densepose.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024 WiFi-DensePose Team"

# Package metadata
__title__ = "wifi-densepose"
__description__ = "WiFi-based human pose estimation using CSI data and DensePose neural networks"
__url__ = "https://github.com/wifi-densepose/wifi-densepose"
__download_url__ = "https://github.com/wifi-densepose/wifi-densepose/archive/main.zip"

# Version info tuple
__version_info__ = tuple(int(x) for x in __version__.split('.'))

from importlib import import_module

# Export common symbols via lazy imports to avoid loading runtime configuration
# (e.g., environment-backed settings) on package import.
__all__ = [
    # Core app
    'app',
    'get_settings',
    'Settings',
    'setup_logging',
    'get_logger',

    # Core processing
    'CSIProcessor',
    'PhaseSanitizer',
    'RouterInterface',

    # Services
    'ServiceOrchestrator',
    'HealthCheckService',
    'MetricsService',

    # Database
    'get_database_manager',
    'Device',
    'Session',
    'CSIData',
    'PoseDetection',
    'SystemMetric',
    'AuditLog',

    # Metadata
    '__version__',
    '__version_info__',
    '__author__',
    '__email__',
    '__license__',
    '__copyright__',
]


_LAZY_IMPORTS = {
    'app': ('src.app', 'app'),
    'get_settings': ('src.config.settings', 'get_settings'),
    'Settings': ('src.config.settings', 'Settings'),
    'setup_logging': ('src.logger', 'setup_logging'),
    'get_logger': ('src.logger', 'get_logger'),
    'CSIProcessor': ('src.core.csi_processor', 'CSIProcessor'),
    'PhaseSanitizer': ('src.core.phase_sanitizer', 'PhaseSanitizer'),
    'RouterInterface': ('src.core.router_interface', 'RouterInterface'),
    'ServiceOrchestrator': ('src.services.orchestrator', 'ServiceOrchestrator'),
    'HealthCheckService': ('src.services.health_check', 'HealthCheckService'),
    'MetricsService': ('src.services.metrics', 'MetricsService'),
    'get_database_manager': ('src.database.connection', 'get_database_manager'),
    'Device': ('src.database.models', 'Device'),
    'Session': ('src.database.models', 'Session'),
    'CSIData': ('src.database.models', 'CSIData'),
    'PoseDetection': ('src.database.models', 'PoseDetection'),
    'SystemMetric': ('src.database.models', 'SystemMetric'),
    'AuditLog': ('src.database.models', 'AuditLog'),
}


def __getattr__(name):
    """Lazily import optional runtime symbols from their source modules."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_version():
    """Get the package version."""
    return __version__


def get_version_info():
    """Get the package version as a tuple."""
    return __version_info__


def get_package_info():
    """Get comprehensive package information."""
    return {
        'name': __title__,
        'version': __version__,
        'version_info': __version_info__,
        'description': __description__,
        'author': __author__,
        'author_email': __email__,
        'license': __license__,
        'copyright': __copyright__,
        'url': __url__,
        'download_url': __download_url__,
    }


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    optional_deps = []
    
    # Core dependencies
    required_modules = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('redis', 'Redis'),
        ('psutil', 'psutil'),
        ('click', 'Click'),
    ]
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(display_name)
    
    # Optional dependencies
    optional_modules = [
        ('scapy', 'Scapy (for network packet capture)'),
        ('paramiko', 'Paramiko (for SSH connections)'),
        ('serial', 'PySerial (for serial communication)'),
        ('matplotlib', 'Matplotlib (for plotting)'),
        ('prometheus_client', 'Prometheus Client (for metrics)'),
    ]
    
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
        except ImportError:
            optional_deps.append(display_name)
    
    return {
        'missing_required': missing_deps,
        'missing_optional': optional_deps,
        'all_required_available': len(missing_deps) == 0,
    }


def print_system_info():
    """Print system and package information."""
    import sys
    import platform
    
    info = get_package_info()
    deps = check_dependencies()
    
    print(f"WiFi-DensePose v{info['version']}")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print()
    
    if deps['all_required_available']:
        print("✅ All required dependencies are available")
    else:
        print("❌ Missing required dependencies:")
        for dep in deps['missing_required']:
            print(f"   - {dep}")
    
    if deps['missing_optional']:
        print("\n⚠️  Missing optional dependencies:")
        for dep in deps['missing_optional']:
            print(f"   - {dep}")
    
    print(f"\nFor more information, visit: {info['url']}")


# Package-level configuration
import logging

# Set up basic logging configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Suppress some noisy third-party loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Package initialization message
if __name__ != '__main__':
    logger = logging.getLogger(__name__)
    logger.debug(f"WiFi-DensePose package v{__version__} initialized")


# Compatibility aliases for backward compatibility
try:
    WifiDensePose = app  # Legacy alias
except NameError:
    WifiDensePose = None  # Will be None if app import failed

try:
    get_config = get_settings  # Legacy alias
except NameError:
    get_config = None  # Will be None if get_settings import failed


def main():
    """Main entry point for the package when run as a module."""
    print_system_info()


if __name__ == '__main__':
    main()