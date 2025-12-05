"""Tests for the event hooks module."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import asyncio

from lib.hooks import (
    HookType, HookRegistry,
    ToolCallEvent, ToolResultEvent, ResponseEvent,
    ModelCallEvent, ModelResultEvent, DelegationEvent, TodoChangeEvent,
    on_tool_call, after_tool_call, on_response,
    before_model_call, after_model_call, on_delegation, on_todo_change
)


class TestEventDataClasses:
    """Tests for event data classes."""

    def test_tool_call_event(self):
        """Test ToolCallEvent creation and attributes."""
        event = ToolCallEvent(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            tool_call_id="call_123",
            agent_name="TestAgent"
        )
        assert event.tool_name == "test_tool"
        assert event.tool_args == {"arg1": "value1"}
        assert event.tool_call_id == "call_123"
        assert event.agent_name == "TestAgent"

    def test_tool_result_event(self):
        """Test ToolResultEvent creation and attributes."""
        event = ToolResultEvent(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            tool_call_id="call_123",
            agent_name="TestAgent",
            result="Tool result"
        )
        assert event.tool_name == "test_tool"
        assert event.result == "Tool result"

    def test_response_event(self):
        """Test ResponseEvent creation and attributes."""
        event = ResponseEvent(
            content="Hello, world!",
            agent_name="TestAgent",
            has_tool_calls=False
        )
        assert event.content == "Hello, world!"
        assert event.agent_name == "TestAgent"
        assert event.has_tool_calls is False
        assert event.tool_calls == []

    def test_response_event_with_tool_calls(self):
        """Test ResponseEvent with tool calls."""
        event = ResponseEvent(
            content="",
            agent_name="TestAgent",
            has_tool_calls=True,
            tool_calls=[{"id": "1", "name": "tool", "args": {}}]
        )
        assert event.has_tool_calls is True
        assert len(event.tool_calls) == 1

    def test_model_call_event(self):
        """Test ModelCallEvent creation and attributes."""
        event = ModelCallEvent(
            agent_name="TestAgent",
            message_count=5,
            system_prompt="You are a helpful assistant."
        )
        assert event.agent_name == "TestAgent"
        assert event.message_count == 5
        assert event.system_prompt == "You are a helpful assistant."

    def test_model_result_event(self):
        """Test ModelResultEvent creation and attributes."""
        event = ModelResultEvent(
            agent_name="TestAgent",
            response_content="Hello!",
            has_tool_calls=False
        )
        assert event.response_content == "Hello!"
        assert event.has_tool_calls is False

    def test_delegation_event(self):
        """Test DelegationEvent creation and attributes."""
        event = DelegationEvent(
            from_agent="MainAgent",
            to_agent="SubAgent",
            task="Do something"
        )
        assert event.from_agent == "MainAgent"
        assert event.to_agent == "SubAgent"
        assert event.task == "Do something"

    def test_todo_change_event_add(self):
        """Test TodoChangeEvent for add action."""
        event = TodoChangeEvent(
            action="add",
            todo_id=1,
            title="Test Todo",
            description="A test todo item",
            state="pending"
        )
        assert event.action == "add"
        assert event.todo_id == 1
        assert event.title == "Test Todo"
        assert event.description == "A test todo item"
        assert event.state == "pending"
        assert event.old_state is None

    def test_todo_change_event_update(self):
        """Test TodoChangeEvent for update action with state change."""
        event = TodoChangeEvent(
            action="update",
            todo_id=1,
            title="Updated Todo",
            state="in_progress",
            old_state="pending"
        )
        assert event.action == "update"
        assert event.state == "in_progress"
        assert event.old_state == "pending"

    def test_todo_change_event_delete(self):
        """Test TodoChangeEvent for delete action."""
        event = TodoChangeEvent(
            action="delete",
            todo_id=1,
            title="Deleted Todo"
        )
        assert event.action == "delete"
        assert event.todo_id == 1


class TestHookDecorators:
    """Tests for hook decorators."""

    def test_on_tool_call_decorator(self):
        """Test on_tool_call decorator marks function correctly."""
        @on_tool_call
        def my_handler(event):
            pass

        assert hasattr(my_handler, '_is_hook')
        assert my_handler._is_hook is True
        assert my_handler._hook_type == HookType.ON_TOOL_CALL

    def test_after_tool_call_decorator(self):
        """Test after_tool_call decorator marks function correctly."""
        @after_tool_call
        def my_handler(event):
            pass

        assert my_handler._hook_type == HookType.AFTER_TOOL_CALL

    def test_on_response_decorator(self):
        """Test on_response decorator marks function correctly."""
        @on_response
        def my_handler(event):
            pass

        assert my_handler._hook_type == HookType.ON_RESPONSE

    def test_before_model_call_decorator(self):
        """Test before_model_call decorator marks function correctly."""
        @before_model_call
        def my_handler(event):
            pass

        assert my_handler._hook_type == HookType.BEFORE_MODEL_CALL

    def test_after_model_call_decorator(self):
        """Test after_model_call decorator marks function correctly."""
        @after_model_call
        def my_handler(event):
            pass

        assert my_handler._hook_type == HookType.AFTER_MODEL_CALL

    def test_on_delegation_decorator(self):
        """Test on_delegation decorator marks function correctly."""
        @on_delegation
        def my_handler(event):
            pass

        assert my_handler._hook_type == HookType.ON_DELEGATION

    def test_on_todo_change_decorator(self):
        """Test on_todo_change decorator marks function correctly."""
        @on_todo_change
        def my_handler(event):
            pass

        assert hasattr(my_handler, '_is_hook')
        assert my_handler._is_hook is True
        assert my_handler._hook_type == HookType.ON_TODO_CHANGE

    def test_decorated_function_still_callable(self):
        """Test that decorated functions can still be called."""
        results = []

        @on_tool_call
        def my_handler(event):
            results.append(event)
            return "handled"

        event = ToolCallEvent("tool", {}, "id", "agent")
        result = my_handler(event)

        assert result == "handled"
        assert len(results) == 1


class TestHookRegistry:
    """Tests for HookRegistry class."""

    def test_registry_initialization(self):
        """Test HookRegistry initializes with empty hook lists."""
        registry = HookRegistry()
        for hook_type in HookType:
            assert registry._hooks[hook_type] == []

    def test_register_callback(self):
        """Test registering a callback."""
        registry = HookRegistry()
        callback = MagicMock()

        registry.register(HookType.ON_TOOL_CALL, callback)

        assert callback in registry._hooks[HookType.ON_TOOL_CALL]

    def test_register_multiple_callbacks(self):
        """Test registering multiple callbacks for same hook type."""
        registry = HookRegistry()
        callback1 = MagicMock()
        callback2 = MagicMock()

        registry.register(HookType.ON_TOOL_CALL, callback1)
        registry.register(HookType.ON_TOOL_CALL, callback2)

        assert len(registry._hooks[HookType.ON_TOOL_CALL]) == 2

    def test_trigger_hooks(self):
        """Test triggering hooks calls all registered callbacks."""
        registry = HookRegistry()
        callback1 = MagicMock()
        callback2 = MagicMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, callback1)
        registry.register(HookType.ON_TOOL_CALL, callback2)
        registry.trigger(HookType.ON_TOOL_CALL, event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_trigger_empty_hooks(self):
        """Test triggering hooks with no callbacks registered."""
        registry = HookRegistry()
        event = ToolCallEvent("tool", {}, "id", "agent")
        # Should not raise
        registry.trigger(HookType.ON_TOOL_CALL, event)

    def test_trigger_handles_exceptions(self):
        """Test that exceptions in hooks don't break the chain."""
        registry = HookRegistry()
        callback1 = MagicMock(side_effect=ValueError("test error"))
        callback2 = MagicMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, callback1)
        registry.register(HookType.ON_TOOL_CALL, callback2)

        # Should not raise, and second callback should still be called
        registry.trigger(HookType.ON_TOOL_CALL, event)
        callback2.assert_called_once_with(event)

    def test_has_hooks(self):
        """Test has_hooks method."""
        registry = HookRegistry()
        assert registry.has_hooks(HookType.ON_TOOL_CALL) is False

        registry.register(HookType.ON_TOOL_CALL, MagicMock())
        assert registry.has_hooks(HookType.ON_TOOL_CALL) is True

    def test_clear_specific_hook_type(self):
        """Test clearing hooks of a specific type."""
        registry = HookRegistry()
        registry.register(HookType.ON_TOOL_CALL, MagicMock())
        registry.register(HookType.ON_RESPONSE, MagicMock())

        registry.clear(HookType.ON_TOOL_CALL)

        assert registry.has_hooks(HookType.ON_TOOL_CALL) is False
        assert registry.has_hooks(HookType.ON_RESPONSE) is True

    def test_clear_all_hooks(self):
        """Test clearing all hooks."""
        registry = HookRegistry()
        registry.register(HookType.ON_TOOL_CALL, MagicMock())
        registry.register(HookType.ON_RESPONSE, MagicMock())

        registry.clear()

        assert registry.has_hooks(HookType.ON_TOOL_CALL) is False
        assert registry.has_hooks(HookType.ON_RESPONSE) is False

    def test_register_hooks_from_object(self):
        """Test registering hooks from a decorated object."""
        class MyHooks:
            @on_tool_call
            def handle_tool_call(self, event):
                pass

            @on_response
            def handle_response(self, event):
                pass

            def regular_method(self):
                pass

        registry = HookRegistry()
        hooks_obj = MyHooks()
        registry.register_hooks_from_object(hooks_obj)

        assert registry.has_hooks(HookType.ON_TOOL_CALL)
        assert registry.has_hooks(HookType.ON_RESPONSE)
        assert not registry.has_hooks(HookType.ON_DELEGATION)


class TestAsyncHooks:
    """Tests for async hook functionality."""

    @pytest.mark.asyncio
    async def test_atrigger_with_sync_callback(self):
        """Test atrigger works with sync callbacks."""
        registry = HookRegistry()
        callback = MagicMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, callback)
        await registry.atrigger(HookType.ON_TOOL_CALL, event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_atrigger_with_async_callback(self):
        """Test atrigger works with async callbacks."""
        registry = HookRegistry()
        callback = AsyncMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, callback)
        await registry.atrigger(HookType.ON_TOOL_CALL, event)

        callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_atrigger_mixed_callbacks(self):
        """Test atrigger handles mixed sync/async callbacks."""
        registry = HookRegistry()
        sync_callback = MagicMock()
        async_callback = AsyncMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, sync_callback)
        registry.register(HookType.ON_TOOL_CALL, async_callback)
        await registry.atrigger(HookType.ON_TOOL_CALL, event)

        sync_callback.assert_called_once_with(event)
        async_callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_atrigger_handles_exceptions(self):
        """Test that exceptions in async hooks don't break the chain."""
        registry = HookRegistry()
        callback1 = AsyncMock(side_effect=ValueError("test error"))
        callback2 = AsyncMock()
        event = ToolCallEvent("tool", {}, "id", "agent")

        registry.register(HookType.ON_TOOL_CALL, callback1)
        registry.register(HookType.ON_TOOL_CALL, callback2)

        # Should not raise, and second callback should still be called
        await registry.atrigger(HookType.ON_TOOL_CALL, event)
        callback2.assert_awaited_once_with(event)
