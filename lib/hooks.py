"""Event hooks module for the Graphent multi-agent framework.

This module provides decorators and data classes for registering event
callbacks that are triggered at various points during agent execution.

Supported hooks:
- @on_tool_call: Before a tool is called
- @after_tool_call: After a tool returns a result
- @on_response: When the model generates a response
- @before_model_call: Before invoking the model
- @after_model_call: After the model returns
- @on_delegation: When delegating to a sub-agent
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from functools import wraps


class HookType(Enum):
    """Enumeration of available hook types."""
    ON_TOOL_CALL = "on_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_RESPONSE = "on_response"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"
    ON_DELEGATION = "on_delegation"
    ON_TODO_CHANGE = "on_todo_change"


@dataclass
class ToolCallEvent:
    """Event data for tool call hooks.
    
    Attributes:
        tool_name: The name of the tool being called.
        tool_args: The arguments being passed to the tool.
        tool_call_id: The unique identifier for this tool call.
        agent_name: The name of the agent making the tool call.
    """
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str
    agent_name: str


@dataclass
class ToolResultEvent:
    """Event data for after tool call hooks.
    
    Attributes:
        tool_name: The name of the tool that was called.
        tool_args: The arguments that were passed to the tool.
        tool_call_id: The unique identifier for this tool call.
        agent_name: The name of the agent that made the tool call.
        result: The result returned by the tool.
    """
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str
    agent_name: str
    result: Any


@dataclass
class ResponseEvent:
    """Event data for response hooks.
    
    Attributes:
        content: The response content from the model.
        agent_name: The name of the agent generating the response.
        has_tool_calls: Whether the response includes tool calls.
        tool_calls: List of tool calls in the response, if any.
    """
    content: str
    agent_name: str
    has_tool_calls: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelCallEvent:
    """Event data for model call hooks (before/after).
    
    Attributes:
        agent_name: The name of the agent making the model call.
        message_count: The number of messages in the context.
        system_prompt: The system prompt being used.
    """
    agent_name: str
    message_count: int
    system_prompt: str


@dataclass
class ModelResultEvent:
    """Event data for after model call hooks.
    
    Attributes:
        agent_name: The name of the agent that made the model call.
        response_content: The content of the model's response.
        has_tool_calls: Whether the response includes tool calls.
        tool_calls: List of tool calls in the response, if any.
    """
    agent_name: str
    response_content: str
    has_tool_calls: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DelegationEvent:
    """Event data for delegation hooks.
    
    Attributes:
        from_agent: The name of the agent delegating the task.
        to_agent: The name of the agent receiving the delegation.
        task: The task description being delegated.
    """
    from_agent: str
    to_agent: str
    task: str


@dataclass
class TodoChangeEvent:
    """Event data for todo change hooks.

    Attributes:
        action: The type of change (add, update, delete).
        todo_id: The ID of the todo that changed.
        title: The title of the todo (for add/update).
        description: The description of the todo (for add/update).
        state: The state of the todo (for add/update).
        old_state: The previous state (for update, when state changed).
    """
    action: str  # "add", "update", "delete"
    todo_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    state: Optional[str] = None
    old_state: Optional[str] = None


# Type alias for hook callbacks
HookCallback = Callable[[Any], None]


def _create_hook_decorator(hook_type: HookType):
    """Factory function to create hook decorators.
    
    Args:
        hook_type: The type of hook to create a decorator for.
        
    Returns:
        A decorator function that marks methods as hook handlers.
    """
    def decorator(func: Callable) -> Callable:
        """Mark a function as a hook handler.
        
        Args:
            func: The function to mark as a hook handler.
            
        Returns:
            The function with hook metadata attached.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Mark this function as a hook handler
        wrapper._hook_type = hook_type
        wrapper._is_hook = True
        return wrapper
    
    return decorator


# Hook decorators
on_tool_call = _create_hook_decorator(HookType.ON_TOOL_CALL)
"""Decorator to mark a method as an on_tool_call hook handler.

The decorated method will receive a ToolCallEvent object with information
about the tool being called.

Example:
    >>> @on_tool_call
    ... def log_tool_call(event: ToolCallEvent):
    ...     print(f"Calling tool: {event.tool_name}")
"""

after_tool_call = _create_hook_decorator(HookType.AFTER_TOOL_CALL)
"""Decorator to mark a method as an after_tool_call hook handler.

The decorated method will receive a ToolResultEvent object with information
about the tool call and its result.

Example:
    >>> @after_tool_call
    ... def log_tool_result(event: ToolResultEvent):
    ...     print(f"Tool {event.tool_name} returned: {event.result}")
"""

on_response = _create_hook_decorator(HookType.ON_RESPONSE)
"""Decorator to mark a method as an on_response hook handler.

The decorated method will receive a ResponseEvent object with information
about the model's response.

Example:
    >>> @on_response
    ... def log_response(event: ResponseEvent):
    ...     print(f"Agent {event.agent_name} responded: {event.content[:50]}...")
"""

before_model_call = _create_hook_decorator(HookType.BEFORE_MODEL_CALL)
"""Decorator to mark a method as a before_model_call hook handler.

The decorated method will receive a ModelCallEvent object with information
about the upcoming model call.

Example:
    >>> @before_model_call
    ... def log_before_model(event: ModelCallEvent):
    ...     print(f"Agent {event.agent_name} calling model with {event.message_count} messages")
"""

after_model_call = _create_hook_decorator(HookType.AFTER_MODEL_CALL)
"""Decorator to mark a method as an after_model_call hook handler.

The decorated method will receive a ModelResultEvent object with information
about the model's response.

Example:
    >>> @after_model_call
    ... def log_after_model(event: ModelResultEvent):
    ...     print(f"Model returned with tool_calls={event.has_tool_calls}")
"""

on_delegation = _create_hook_decorator(HookType.ON_DELEGATION)
"""Decorator to mark a method as an on_delegation hook handler.

The decorated method will receive a DelegationEvent object with information
about the task delegation.

Example:
    >>> @on_delegation
    ... def log_delegation(event: DelegationEvent):
    ...     print(f"Agent {event.from_agent} delegating to {event.to_agent}")
"""

on_todo_change = _create_hook_decorator(HookType.ON_TODO_CHANGE)
"""Decorator to mark a method as an on_todo_change hook handler.

The decorated method will receive a TodoChangeEvent object with information
about the todo list change.

Example:
    >>> @on_todo_change
    ... def log_todo_change(event: TodoChangeEvent):
    ...     print(f"Todo {event.action}: {event.title}")
"""


class HookRegistry:
    """Registry for managing event hook callbacks.
    
    The HookRegistry collects and organizes hook handlers, allowing
    them to be triggered at appropriate points during agent execution.
    
    Attributes:
        _hooks: Dictionary mapping hook types to lists of callbacks.
    """
    
    def __init__(self):
        """Initialize an empty hook registry."""
        self._hooks: dict[HookType, list[HookCallback]] = {
            hook_type: [] for hook_type in HookType
        }
    
    def register(self, hook_type: HookType, callback: HookCallback) -> None:
        """Register a callback for a specific hook type.
        
        Args:
            hook_type: The type of hook to register for.
            callback: The callback function to invoke when the hook triggers.
        """
        self._hooks[hook_type].append(callback)
    
    def register_hooks_from_object(self, obj: Any) -> None:
        """Register all hook handlers found on an object.
        
        Scans the object for methods decorated with hook decorators
        and registers them automatically.
        
        Args:
            obj: The object to scan for hook handlers.
        """
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name, None)
            if callable(attr) and getattr(attr, '_is_hook', False):
                hook_type = getattr(attr, '_hook_type')
                self.register(hook_type, attr)
    
    def trigger(self, hook_type: HookType, event: Any) -> None:
        """Trigger all callbacks registered for a hook type.
        
        Args:
            hook_type: The type of hook to trigger.
            event: The event data to pass to the callbacks.
        """
        for callback in self._hooks[hook_type]:
            try:
                callback(event)
            except Exception as e:
                # Log but don't raise - hooks shouldn't break the main flow
                import logging
                callback_name = getattr(callback, '__name__', repr(callback))
                logging.warning(f"Hook callback {callback_name} raised exception: {e}")
    
    async def atrigger(self, hook_type: HookType, event: Any) -> None:
        """Asynchronously trigger all callbacks registered for a hook type.
        
        For async callbacks, awaits them. For sync callbacks, calls them directly.
        
        Args:
            hook_type: The type of hook to trigger.
            event: The event data to pass to the callbacks.
        """
        import asyncio
        for callback in self._hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                # Log but don't raise - hooks shouldn't break the main flow
                import logging
                callback_name = getattr(callback, '__name__', repr(callback))
                logging.warning(f"Hook callback {callback_name} raised exception: {e}")
    
    def has_hooks(self, hook_type: HookType) -> bool:
        """Check if any hooks are registered for a given type.
        
        Args:
            hook_type: The type of hook to check.
            
        Returns:
            True if any callbacks are registered, False otherwise.
        """
        return len(self._hooks[hook_type]) > 0
    
    def clear(self, hook_type: Optional[HookType] = None) -> None:
        """Clear registered hooks.
        
        Args:
            hook_type: If provided, only clear hooks of this type.
                If None, clear all hooks.
        """
        if hook_type is not None:
            self._hooks[hook_type] = []
        else:
            for ht in HookType:
                self._hooks[ht] = []
