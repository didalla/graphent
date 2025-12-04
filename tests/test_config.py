"""Test suite for the config module.

Tests cover configuration management for the Graphent framework.
"""

import pytest
from lib.config import (
    GraphentConfig,
    get_config,
    set_config,
    reset_config,
)


class TestGraphentConfig:
    """Tests for the GraphentConfig dataclass."""

    def test_default_values(self):
        """GraphentConfig should have sensible defaults."""
        config = GraphentConfig()
        assert config.log_level == 25
        assert config.truncate_limit == 250
        assert config.max_iterations == 10
        assert config.log_file is None
        assert config.debug is False

    def test_custom_values(self):
        """GraphentConfig should accept custom values."""
        config = GraphentConfig(
            log_level=10,
            truncate_limit=500,
            max_iterations=20,
            log_file="/tmp/test.log",
            debug=True,
        )
        assert config.log_level == 10
        assert config.truncate_limit == 500
        assert config.max_iterations == 20
        assert config.log_file == "/tmp/test.log"
        assert config.debug is True

    def test_from_env_defaults(self, monkeypatch):
        """from_env should use defaults when env vars not set."""
        # Clear any existing env vars
        monkeypatch.delenv("GRAPHENT_LOG_LEVEL", raising=False)
        monkeypatch.delenv("GRAPHENT_TRUNCATE_LIMIT", raising=False)
        monkeypatch.delenv("GRAPHENT_MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("GRAPHENT_LOG_FILE", raising=False)
        monkeypatch.delenv("GRAPHENT_DEBUG", raising=False)

        config = GraphentConfig.from_env()
        assert config.log_level == 25
        assert config.truncate_limit == 250
        assert config.max_iterations == 10
        assert config.log_file is None
        assert config.debug is False

    def test_from_env_reads_env_vars(self, monkeypatch):
        """from_env should read from environment variables."""
        monkeypatch.setenv("GRAPHENT_LOG_LEVEL", "10")
        monkeypatch.setenv("GRAPHENT_TRUNCATE_LIMIT", "500")
        monkeypatch.setenv("GRAPHENT_MAX_ITERATIONS", "20")
        monkeypatch.setenv("GRAPHENT_LOG_FILE", "/tmp/test.log")
        monkeypatch.setenv("GRAPHENT_DEBUG", "true")

        config = GraphentConfig.from_env()
        assert config.log_level == 10
        assert config.truncate_limit == 500
        assert config.max_iterations == 20
        assert config.log_file == "/tmp/test.log"
        assert config.debug is True

    def test_from_env_debug_variations(self, monkeypatch):
        """from_env should accept various truthy values for debug."""
        for value in ("true", "True", "TRUE", "1", "yes", "YES"):
            monkeypatch.setenv("GRAPHENT_DEBUG", value)
            config = GraphentConfig.from_env()
            assert config.debug is True, f"Failed for value: {value}"

    def test_from_env_debug_false(self, monkeypatch):
        """from_env should treat other values as False for debug."""
        for value in ("false", "0", "no", "anything", ""):
            monkeypatch.setenv("GRAPHENT_DEBUG", value)
            config = GraphentConfig.from_env()
            assert config.debug is False, f"Failed for value: {value}"


class TestConfigManagement:
    """Tests for global config management functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_config(self):
        """get_config should return a GraphentConfig instance."""
        config = get_config()
        assert isinstance(config, GraphentConfig)

    def test_get_config_is_cached(self):
        """get_config should return the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_changes_global(self):
        """set_config should change the global configuration."""
        custom_config = GraphentConfig(log_level=10, debug=True)
        set_config(custom_config)

        config = get_config()
        assert config is custom_config
        assert config.log_level == 10
        assert config.debug is True

    def test_reset_config_clears_global(self):
        """reset_config should clear the cached configuration."""
        # Set a custom config
        custom_config = GraphentConfig(log_level=10)
        set_config(custom_config)

        # Reset and get new config
        reset_config()
        config = get_config()

        # Should be a new instance with defaults
        assert config is not custom_config
        assert config.log_level == 25  # default value

