"""Test suite for the exceptions module.

Tests cover all custom exceptions in the Graphent framework.
"""

import pytest
from lib.exceptions import (
    GraphentError,
    AgentConfigurationError,
    ToolExecutionError,
    DelegationError,
    MaxIterationsExceededError,
    HookExecutionError,
)


class TestGraphentError:
    """Tests for the base GraphentError exception."""

    def test_is_exception(self):
        """GraphentError should be an Exception."""
        assert issubclass(GraphentError, Exception)

    def test_can_raise_and_catch(self):
        """GraphentError should be raiseable and catchable."""
        with pytest.raises(GraphentError):
            raise GraphentError("Test error")

    def test_message_preserved(self):
        """GraphentError should preserve the error message."""
        error = GraphentError("Test message")
        assert str(error) == "Test message"


class TestAgentConfigurationError:
    """Tests for AgentConfigurationError."""

    def test_inherits_from_graphent_error(self):
        """AgentConfigurationError should inherit from GraphentError."""
        assert issubclass(AgentConfigurationError, GraphentError)

    def test_can_catch_as_graphent_error(self):
        """AgentConfigurationError should be catchable as GraphentError."""
        with pytest.raises(GraphentError):
            raise AgentConfigurationError("Missing name")


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_inherits_from_graphent_error(self):
        """ToolExecutionError should inherit from GraphentError."""
        assert issubclass(ToolExecutionError, GraphentError)

    def test_stores_tool_name(self):
        """ToolExecutionError should store the tool name."""
        error = ToolExecutionError("my_tool")
        assert error.tool_name == "my_tool"

    def test_stores_original_error(self):
        """ToolExecutionError should store the original error."""
        original = ValueError("Original error")
        error = ToolExecutionError("my_tool", original_error=original)
        assert error.original_error is original

    def test_message_includes_tool_name(self):
        """ToolExecutionError message should include tool name."""
        error = ToolExecutionError("my_tool")
        assert "my_tool" in str(error)

    def test_message_includes_original_error(self):
        """ToolExecutionError message should include original error."""
        original = ValueError("Something went wrong")
        error = ToolExecutionError("my_tool", original_error=original)
        assert "Something went wrong" in str(error)

    def test_custom_message(self):
        """ToolExecutionError should accept custom message."""
        error = ToolExecutionError("my_tool", message="Custom error")
        assert "Custom error" in str(error)


class TestDelegationError:
    """Tests for DelegationError."""

    def test_inherits_from_graphent_error(self):
        """DelegationError should inherit from GraphentError."""
        assert issubclass(DelegationError, GraphentError)

    def test_stores_agent_names(self):
        """DelegationError should store from and to agent names."""
        error = DelegationError("parent", "child")
        assert error.from_agent == "parent"
        assert error.to_agent == "child"

    def test_message_includes_agents(self):
        """DelegationError message should include agent names."""
        error = DelegationError("parent", "child")
        assert "parent" in str(error)
        assert "child" in str(error)

    def test_custom_message(self):
        """DelegationError should accept custom message."""
        error = DelegationError("parent", "child", message="Custom delegation error")
        assert "Custom delegation error" in str(error)


class TestMaxIterationsExceededError:
    """Tests for MaxIterationsExceededError."""

    def test_inherits_from_graphent_error(self):
        """MaxIterationsExceededError should inherit from GraphentError."""
        assert issubclass(MaxIterationsExceededError, GraphentError)

    def test_stores_agent_name_and_max(self):
        """MaxIterationsExceededError should store agent name and max iterations."""
        error = MaxIterationsExceededError("my_agent", 10)
        assert error.agent_name == "my_agent"
        assert error.max_iterations == 10

    def test_message_includes_details(self):
        """MaxIterationsExceededError message should include agent and limit."""
        error = MaxIterationsExceededError("my_agent", 10)
        assert "my_agent" in str(error)
        assert "10" in str(error)


class TestHookExecutionError:
    """Tests for HookExecutionError."""

    def test_inherits_from_graphent_error(self):
        """HookExecutionError should inherit from GraphentError."""
        assert issubclass(HookExecutionError, GraphentError)

    def test_stores_hook_type(self):
        """HookExecutionError should store the hook type."""
        error = HookExecutionError("on_tool_call")
        assert error.hook_type == "on_tool_call"

    def test_stores_original_error(self):
        """HookExecutionError should store the original error."""
        original = RuntimeError("Hook failed")
        error = HookExecutionError("on_tool_call", original_error=original)
        assert error.original_error is original

    def test_message_includes_hook_type(self):
        """HookExecutionError message should include hook type."""
        error = HookExecutionError("on_tool_call")
        assert "on_tool_call" in str(error)

