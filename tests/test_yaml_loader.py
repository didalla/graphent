"""Tests for YAML configuration loader.

Tests cover agent loading, graph loading, model resolution,
inline agents, and error handling.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

from lib.yaml_loader import (
    load_agent_from_yaml,
    load_graph_from_yaml,
    _resolve_env_vars,
    _create_model_from_config,
    _resolve_model,
)
from lib.exceptions import AgentConfigurationError
from lib.Agent import Agent


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""

    def _create_file(content: str) -> Path:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, content.encode())
        os.close(fd)
        return Path(path)

    return _create_file


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "calculator"
    return tool


@pytest.fixture
def env_with_api_key(monkeypatch):
    """Set up environment with API key."""
    monkeypatch.setenv("TEST_API_KEY", "test-key-12345")
    return "test-key-12345"


# ============================================================================
# Environment Variable Resolution Tests
# ============================================================================


class TestEnvVarResolution:
    """Tests for environment variable resolution."""

    def test_resolve_simple_env_var(self, monkeypatch):
        """Should resolve a simple ${VAR} reference."""
        monkeypatch.setenv("MY_VAR", "my_value")
        result = _resolve_env_vars("${MY_VAR}")
        assert result == "my_value"

    def test_resolve_env_var_in_string(self, monkeypatch):
        """Should resolve env var embedded in string."""
        monkeypatch.setenv("API_KEY", "secret123")
        result = _resolve_env_vars("Bearer ${API_KEY}")
        assert result == "Bearer secret123"

    def test_resolve_multiple_env_vars(self, monkeypatch):
        """Should resolve multiple env vars in one string."""
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = _resolve_env_vars("http://${HOST}:${PORT}")
        assert result == "http://localhost:8080"

    def test_missing_env_var_raises(self):
        """Should raise error for missing env var."""
        with pytest.raises(AgentConfigurationError, match="not set"):
            _resolve_env_vars("${NONEXISTENT_VAR_12345}")

    def test_non_string_passthrough(self):
        """Non-string values should pass through unchanged."""
        assert _resolve_env_vars(123) == 123
        assert _resolve_env_vars(None) is None
        assert _resolve_env_vars(["a", "b"]) == ["a", "b"]


# ============================================================================
# Model Creation Tests
# ============================================================================


class TestModelCreation:
    """Tests for model creation from config."""

    @patch("lib.yaml_loader.ChatOpenAI")
    def test_create_openai_model(self, mock_openai, monkeypatch):
        """Should create ChatOpenAI model."""
        monkeypatch.setenv("OPENAI_KEY", "test-key")
        config = {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5,
            "api_key": "${OPENAI_KEY}",
        }
        _create_model_from_config(config)
        mock_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.5,
            api_key="test-key",
            streaming=False,
        )

    @patch("lib.yaml_loader.ChatOpenAI")
    def test_create_openrouter_model(self, mock_openai, monkeypatch):
        """Should create model with custom base_url."""
        monkeypatch.setenv("OR_KEY", "or-key")
        config = {
            "provider": "openrouter",
            "name": "anthropic/claude-3",
            "temperature": 0.3,
            "api_key": "${OR_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
        }
        _create_model_from_config(config)
        mock_openai.assert_called_once_with(
            model="anthropic/claude-3",
            temperature=0.3,
            api_key="or-key",
            streaming=False,
            base_url="https://openrouter.ai/api/v1",
        )

    def test_unsupported_provider_raises(self):
        """Should raise error for unsupported provider."""
        config = {"provider": "unsupported", "name": "model"}
        with pytest.raises(AgentConfigurationError, match="Unsupported model provider"):
            _create_model_from_config(config)


# ============================================================================
# Model Resolution Tests
# ============================================================================


class TestModelResolution:
    """Tests for model reference resolution."""

    @patch("lib.yaml_loader._create_model_from_config")
    def test_resolve_named_model(self, mock_create, monkeypatch):
        """Should resolve model by name from registry."""
        monkeypatch.setenv("KEY", "k")
        registry = {
            "default": {"provider": "openai", "name": "gpt-4", "api_key": "${KEY}"}
        }
        _resolve_model("default", registry)
        mock_create.assert_called_once_with(registry["default"])

    @patch("lib.yaml_loader._create_model_from_config")
    def test_resolve_inline_model(self, mock_create, monkeypatch):
        """Should create model from inline config."""
        monkeypatch.setenv("KEY", "k")
        inline_config = {"provider": "openai", "name": "gpt-4", "api_key": "${KEY}"}
        _resolve_model(inline_config, {})
        mock_create.assert_called_once_with(inline_config)

    def test_resolve_unknown_model_raises(self):
        """Should raise error for unknown model name."""
        with pytest.raises(
            AgentConfigurationError, match="not found in models registry"
        ):
            _resolve_model("unknown", {"other": {}})


# ============================================================================
# Agent Loading Tests
# ============================================================================


class TestAgentLoading:
    """Tests for loading agents from YAML."""

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_simple_agent(self, mock_create, temp_yaml_file, monkeypatch):
        """Should load a simple agent from YAML."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        mock_create.return_value = mock_model

        yaml_content = """
name: "Test Agent"
description: "A test agent"
system_prompt: "You are a test."
model:
  provider: openai
  name: gpt-4
  api_key: ${API_KEY}
"""
        path = temp_yaml_file(yaml_content)
        try:
            agent = load_agent_from_yaml(path)
            assert agent.name == "Test Agent"
            assert "You are a test" in agent.system_prompt
        finally:
            path.unlink()

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_agent_with_tools(
        self, mock_create, temp_yaml_file, mock_tool, monkeypatch
    ):
        """Should load agent with tools from registry."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        mock_create.return_value = mock_model

        yaml_content = """
name: "Tool Agent"
description: "Agent with tools"
system_prompt: "You have tools."
model:
  provider: openai
  name: gpt-4
  api_key: ${API_KEY}
tools:
  - calculator
"""
        path = temp_yaml_file(yaml_content)
        try:
            agent = load_agent_from_yaml(path, tools_registry={"calculator": mock_tool})
            assert agent.name == "Tool Agent"
        finally:
            path.unlink()

    def test_load_agent_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_agent_from_yaml("/nonexistent/path.yaml")

    def test_load_agent_missing_required_fields(self, temp_yaml_file, monkeypatch):
        """Should raise error for missing required fields."""
        monkeypatch.setenv("API_KEY", "test")
        yaml_content = """
name: "Incomplete"
model:
  provider: openai
  api_key: ${API_KEY}
"""
        path = temp_yaml_file(yaml_content)
        try:
            with pytest.raises(AgentConfigurationError, match="requires"):
                load_agent_from_yaml(path)
        finally:
            path.unlink()


# ============================================================================
# Graph Loading Tests
# ============================================================================


class TestGraphLoading:
    """Tests for loading graphs from YAML."""

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_simple_graph(self, mock_create, temp_yaml_file, monkeypatch):
        """Should load a simple graph with action nodes."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.invoke.return_value = AIMessage(content="test")
        mock_create.return_value = mock_model

        yaml_content = """
models:
  default:
    provider: openai
    name: gpt-4
    api_key: ${API_KEY}

nodes:
  - name: start
    type: action
    system_prompt: "Process input"

entry: start
finish: start
"""
        path = temp_yaml_file(yaml_content)
        try:
            graph = load_graph_from_yaml(path)
            assert "start" in graph.nodes
            assert graph.entry_point == "start"
        finally:
            path.unlink()

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_graph_with_edges(self, mock_create, temp_yaml_file, monkeypatch):
        """Should load graph with edges between nodes."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_create.return_value = mock_model

        yaml_content = """
models:
  default:
    provider: openai
    name: gpt-4
    api_key: ${API_KEY}

nodes:
  - name: first
    type: action
    system_prompt: "First step"
  - name: second
    type: action
    system_prompt: "Second step"

edges:
  - from: first
    to: second

entry: first
finish: second
"""
        path = temp_yaml_file(yaml_content)
        try:
            graph = load_graph_from_yaml(path)
            assert len(graph.edges) == 1
            assert graph.edges[0].source == "first"
            assert graph.edges[0].target == "second"
        finally:
            path.unlink()

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_graph_with_classifier(self, mock_create, temp_yaml_file, monkeypatch):
        """Should load graph with classifier and branches."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_create.return_value = mock_model

        yaml_content = """
models:
  default:
    provider: openai
    name: gpt-4
    api_key: ${API_KEY}

nodes:
  - name: router
    type: classifier
    classes: [tech, creative]
  - name: tech_handler
    type: action
    system_prompt: "Handle tech"
  - name: creative_handler
    type: action
    system_prompt: "Handle creative"

branches:
  router:
    tech: tech_handler
    creative: creative_handler

entry: router
"""
        path = temp_yaml_file(yaml_content)
        try:
            graph = load_graph_from_yaml(path)
            assert "router" in graph.nodes
            assert "tech_handler" in graph.nodes
        finally:
            path.unlink()

    @patch("lib.yaml_loader._create_model_from_config")
    def test_load_graph_with_inline_agents(
        self, mock_create, temp_yaml_file, monkeypatch
    ):
        """Should load graph with inline agent definitions."""
        monkeypatch.setenv("API_KEY", "test")
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        mock_create.return_value = mock_model

        yaml_content = """
models:
  default:
    provider: openai
    name: gpt-4
    api_key: ${API_KEY}

agents:
  helper:
    name: "Helper Agent"
    description: "Helps with tasks"
    system_prompt: "You help."

nodes:
  - name: agent_step
    type: agent
    agent: helper

entry: agent_step
finish: agent_step
"""
        path = temp_yaml_file(yaml_content)
        try:
            graph = load_graph_from_yaml(path)
            assert "agent_step" in graph.nodes
        finally:
            path.unlink()

    def test_load_graph_missing_entry_raises(self, temp_yaml_file, monkeypatch):
        """Should raise error if entry node not specified."""
        monkeypatch.setenv("API_KEY", "test")
        yaml_content = """
models:
  default:
    provider: openai
    api_key: ${API_KEY}
nodes:
  - name: test
    type: action
    system_prompt: "Test"
"""
        path = temp_yaml_file(yaml_content)
        try:
            with pytest.raises(AgentConfigurationError, match="entry"):
                load_graph_from_yaml(path)
        finally:
            path.unlink()

    def test_load_graph_unknown_agent_raises(self, temp_yaml_file, monkeypatch):
        """Should raise error for unknown agent reference."""
        monkeypatch.setenv("API_KEY", "test")
        yaml_content = """
models:
  default:
    provider: openai
    api_key: ${API_KEY}
nodes:
  - name: test
    type: agent
    agent: nonexistent
entry: test
"""
        path = temp_yaml_file(yaml_content)
        try:
            with pytest.raises(AgentConfigurationError, match="not found"):
                load_graph_from_yaml(path)
        finally:
            path.unlink()
