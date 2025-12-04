"""Configuration module for the Graphent agent framework.

This module provides centralized configuration management for the library,
supporting both programmatic configuration and environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphentConfig:
    """Configuration settings for the Graphent framework.

    This class centralizes all configurable parameters for the library.
    Values can be set programmatically or loaded from environment variables.

    Attributes:
        log_level: The logging level for agent activity (default: 25, between INFO and WARNING).
        truncate_limit: Maximum characters to show in log output before truncating.
        max_iterations: Default maximum iterations for agent invoke loops.
        log_file: Optional path to a log file. If None, logs to console.
        debug: Enable debug mode with additional logging.

    Example:
        >>> config = GraphentConfig(log_level=20, truncate_limit=500)
        >>> # Or load from environment
        >>> config = GraphentConfig.from_env()
    """

    log_level: int = 25
    truncate_limit: int = 250
    max_iterations: int = 10
    log_file: Optional[str] = None
    debug: bool = False

    @classmethod
    def from_env(cls) -> "GraphentConfig":
        """Create a configuration from environment variables.

        Environment variables:
            - GRAPHENT_LOG_LEVEL: Logging level (default: 25)
            - GRAPHENT_TRUNCATE_LIMIT: Truncation limit (default: 250)
            - GRAPHENT_MAX_ITERATIONS: Max iterations (default: 10)
            - GRAPHENT_LOG_FILE: Log file path (default: None)
            - GRAPHENT_DEBUG: Enable debug mode (default: false)

        Returns:
            A GraphentConfig instance populated from environment variables.
        """
        return cls(
            log_level=int(os.environ.get("GRAPHENT_LOG_LEVEL", 25)),
            truncate_limit=int(os.environ.get("GRAPHENT_TRUNCATE_LIMIT", 250)),
            max_iterations=int(os.environ.get("GRAPHENT_MAX_ITERATIONS", 10)),
            log_file=os.environ.get("GRAPHENT_LOG_FILE"),
            debug=os.environ.get("GRAPHENT_DEBUG", "").lower() in ("true", "1", "yes"),
        )


# Global default configuration instance
_default_config: Optional[GraphentConfig] = None


def get_config() -> GraphentConfig:
    """Get the current global configuration.

    If no configuration has been set, returns a default configuration
    loaded from environment variables.

    Returns:
        The current GraphentConfig instance.
    """
    global _default_config
    if _default_config is None:
        _default_config = GraphentConfig.from_env()
    return _default_config


def set_config(config: GraphentConfig) -> None:
    """Set the global configuration.

    Args:
        config: The GraphentConfig instance to use globally.
    """
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults.

    This will cause get_config() to reload from environment variables
    on its next call.
    """
    global _default_config
    _default_config = None
