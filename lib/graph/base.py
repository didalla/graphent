"""Base classes for graph nodes.

This module provides the abstract base class for all node types
in the graph-based workflow system.
"""

from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator, Optional, TYPE_CHECKING

from lib.Context import Context

if TYPE_CHECKING:
    from lib.graph.hooks import GraphHookRegistry


class BaseNode(ABC):
    """Abstract base node for graph execution.

    All node types (ActionNode, AgentNode, ClassifierNode) inherit from
    this base class and implement the required invoke methods.

    Attributes:
        name: Unique identifier for this node in the graph.
        _hooks: Optional hook registry for node events.

    Example:
        >>> class MyNode(BaseNode):
        ...     def invoke(self, context: Context) -> Context:
        ...         # Process context
        ...         return context
    """

    def __init__(self, name: str, hooks: Optional["GraphHookRegistry"] = None):
        """Initialize a base node.

        Args:
            name: Unique identifier for this node.
            hooks: Optional hook registry for events.
        """
        self.name = name
        self._hooks = hooks

    def set_hooks(self, hooks: "GraphHookRegistry") -> None:
        """Set the hook registry for this node.

        Args:
            hooks: The hook registry to use.
        """
        self._hooks = hooks

    @abstractmethod
    def invoke(self, context: Context) -> Context:
        """Execute the node with the given context.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context after processing.
        """
        pass

    @abstractmethod
    async def ainvoke(self, context: Context) -> Context:
        """Asynchronously execute the node with the given context.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context after processing.
        """
        pass

    def stream(self, context: Context) -> Generator[str, None, Context]:
        """Stream the node's response chunk by chunk.

        Default implementation falls back to invoke().
        Subclasses can override for true streaming support.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.

        Returns:
            The updated context after processing.
        """
        result = self.invoke(context)
        # Yield the last message content if available
        messages = result.get_messages(last_n=1)
        if messages and hasattr(messages[0], "content"):
            content = messages[0].content
            if isinstance(content, str):
                yield content
        return result

    async def astream(self, context: Context) -> AsyncGenerator[str, None]:
        """Asynchronously stream the node's response.

        Default implementation falls back to ainvoke().
        Subclasses can override for true streaming support.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.
        """
        result = await self.ainvoke(context)
        # Yield the last message content if available
        messages = result.get_messages(last_n=1)
        if messages and hasattr(messages[0], "content"):
            content = messages[0].content
            if isinstance(content, str):
                yield content

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return f"{self.__class__.__name__}(name='{self.name}')"
