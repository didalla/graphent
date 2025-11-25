"""Context module for managing conversation state.

This module provides the Context class for tracking message history
in agent conversations.
"""

from typing import Optional

from langchain_core.messages import BaseMessage


class Context:
    """A container for conversation messages.
    
    The Context class maintains an ordered list of messages exchanged
    during an agent conversation. It provides methods for adding and
    retrieving messages.
    
    Attributes:
        _messages: Internal list of BaseMessage objects.
    
    Example:
        >>> context = Context()
        >>> context.add_message(HumanMessage(content="Hello"))
        >>> context.add_message(AIMessage(content="Hi there!"))
        >>> len(context.get_messages())
        2
    """
    
    def __init__(self):
        """Initialize an empty Context."""
        self._messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> "Context":
        """Add a message to the context.
        
        Args:
            message: A LangChain BaseMessage to append.
            
        Returns:
            Self for method chaining.
        """
        self._messages.append(message)
        return self

    def get_messages(self, last_n: Optional[int] = None) -> list[BaseMessage]:
        """Retrieve messages from the context.
        
        Args:
            last_n: If provided, return only the last N messages.
                If None, return all messages.
                
        Returns:
            A list of messages from the context.
        """
        if last_n:
            return self._messages[-last_n:]
        return self._messages

    def __str__(self) -> str:
        """Return a string representation of the context."""
        return str(self._messages)
