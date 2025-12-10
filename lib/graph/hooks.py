"""Graph-specific hooks for node and edge events.

This module provides hook types and event data classes for monitoring
and reacting to graph execution events.

Supported hooks:
- @on_node_enter: Before a node is executed
- @on_node_exit: After a node completes execution
- @on_edge_traverse: When traversing an edge between nodes
- @on_classification: When a classifier node makes a decision
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
from functools import wraps

from lib.Context import Context


class GraphHookType(Enum):
    """Enumeration of available graph hook types."""

    ON_NODE_ENTER = "on_node_enter"
    ON_NODE_EXIT = "on_node_exit"
    ON_EDGE_TRAVERSE = "on_edge_traverse"
    ON_CLASSIFICATION = "on_classification"


@dataclass
class NodeEnterEvent:
    """Event data for node entry hooks.

    Attributes:
        node_name: The name of the node being entered.
        node_type: The type/class name of the node.
        context: The context being passed to the node.
    """

    node_name: str
    node_type: str
    context: Context


@dataclass
class NodeExitEvent:
    """Event data for node exit hooks.

    Attributes:
        node_name: The name of the node that completed.
        node_type: The type/class name of the node.
        context: The context after node processing.
    """

    node_name: str
    node_type: str
    context: Context


@dataclass
class EdgeTraverseEvent:
    """Event data for edge traversal hooks.

    Attributes:
        source_node: The name of the source node.
        target_node: The name of the target node.
        edge_type: Type of edge (simple or conditional).
    """

    source_node: str
    target_node: str
    edge_type: str


@dataclass
class ClassificationEvent:
    """Event data for classification hooks.

    Attributes:
        node_name: The name of the classifier node.
        classification: The classification result.
        target_node: The target node selected based on classification.
        confidence: Optional confidence score for the classification.
    """

    node_name: str
    classification: str
    target_node: str
    confidence: Optional[float] = None


# Type alias for hook callbacks
GraphHookCallback = Callable[[Any], None]


def _create_graph_hook_decorator(hook_type: GraphHookType):
    """Factory function to create graph hook decorators.

    Args:
        hook_type: The type of hook to create a decorator for.

    Returns:
        A decorator function that marks methods as hook handlers.
    """

    def decorator(func: Callable) -> Callable:
        """Mark a function as a graph hook handler.

        Args:
            func: The function to mark as a hook handler.

        Returns:
            The function with hook metadata attached.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Mark this function as a graph hook handler
        wrapper._graph_hook_type = hook_type
        wrapper._is_graph_hook = True
        return wrapper

    return decorator


# Hook decorators
on_node_enter = _create_graph_hook_decorator(GraphHookType.ON_NODE_ENTER)
"""Decorator to mark a method as an on_node_enter hook handler.

The decorated method will receive a NodeEnterEvent object.

Example:
    >>> @on_node_enter
    ... def log_entry(event: NodeEnterEvent):
    ...     print(f"Entering node: {event.node_name}")
"""

on_node_exit = _create_graph_hook_decorator(GraphHookType.ON_NODE_EXIT)
"""Decorator to mark a method as an on_node_exit hook handler.

The decorated method will receive a NodeExitEvent object.

Example:
    >>> @on_node_exit
    ... def log_exit(event: NodeExitEvent):
    ...     print(f"Exiting node: {event.node_name}")
"""

on_edge_traverse = _create_graph_hook_decorator(GraphHookType.ON_EDGE_TRAVERSE)
"""Decorator to mark a method as an on_edge_traverse hook handler.

The decorated method will receive an EdgeTraverseEvent object.

Example:
    >>> @on_edge_traverse
    ... def log_edge(event: EdgeTraverseEvent):
    ...     print(f"Traversing: {event.source_node} -> {event.target_node}")
"""

on_classification = _create_graph_hook_decorator(GraphHookType.ON_CLASSIFICATION)
"""Decorator to mark a method as an on_classification hook handler.

The decorated method will receive a ClassificationEvent object.

Example:
    >>> @on_classification
    ... def log_classification(event: ClassificationEvent):
    ...     print(f"Classified as: {event.classification}")
"""


class GraphHookRegistry:
    """Registry for managing graph event hook callbacks.

    The GraphHookRegistry collects and organizes hook handlers,
    allowing them to be triggered at appropriate points during
    graph execution.

    Attributes:
        _hooks: Dictionary mapping hook types to lists of callbacks.
    """

    def __init__(self):
        """Initialize an empty graph hook registry."""
        self._hooks: dict[GraphHookType, list[GraphHookCallback]] = {
            hook_type: [] for hook_type in GraphHookType
        }

    def register(self, hook_type: GraphHookType, callback: GraphHookCallback) -> None:
        """Register a callback for a specific hook type.

        Args:
            hook_type: The type of hook to register for.
            callback: The callback function to invoke when the hook triggers.
        """
        self._hooks[hook_type].append(callback)

    def register_hooks_from_object(self, obj: Any) -> None:
        """Register all graph hook handlers found on an object.

        Scans the object for methods decorated with graph hook decorators
        and registers them automatically.

        Args:
            obj: The object to scan for hook handlers.
        """
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name, None)
            if callable(attr) and getattr(attr, "_is_graph_hook", False):
                hook_type = getattr(attr, "_graph_hook_type")
                self.register(hook_type, attr)

    def trigger(self, hook_type: GraphHookType, event: Any) -> None:
        """Trigger all callbacks registered for a hook type.

        Args:
            hook_type: The type of hook to trigger.
            event: The event data to pass to the callbacks.
        """
        for callback in self._hooks[hook_type]:
            try:
                callback(event)
            except Exception as e:
                import logging

                callback_name = getattr(callback, "__name__", repr(callback))
                logging.warning(
                    f"Graph hook callback {callback_name} raised exception: {e}"
                )

    async def atrigger(self, hook_type: GraphHookType, event: Any) -> None:
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
                import logging

                callback_name = getattr(callback, "__name__", repr(callback))
                logging.warning(
                    f"Graph hook callback {callback_name} raised exception: {e}"
                )

    def has_hooks(self, hook_type: GraphHookType) -> bool:
        """Check if any hooks are registered for a given type.

        Args:
            hook_type: The type of hook to check.

        Returns:
            True if any callbacks are registered, False otherwise.
        """
        return len(self._hooks[hook_type]) > 0

    def clear(self, hook_type: Optional[GraphHookType] = None) -> None:
        """Clear registered hooks.

        Args:
            hook_type: If provided, only clear hooks of this type.
                If None, clear all hooks.
        """
        if hook_type is not None:
            self._hooks[hook_type] = []
        else:
            for ht in GraphHookType:
                self._hooks[ht] = []
