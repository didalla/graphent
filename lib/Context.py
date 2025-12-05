"""Context module for managing conversation state.

This module provides the Context class for tracking message history
in agent conversations.
"""

import itertools
from typing import Optional, TYPE_CHECKING

from langchain_core.messages import BaseMessage
from typing import Literal

if TYPE_CHECKING:
    from lib.hooks import HookRegistry


class Todo:
    """A simple class to store a todo item.

    Attributes:
        id: Unique identifier for the todo item.
        title: The title of the todo item.
        description: A detailed description of the todo item.
        state: Current state of the Todo item (pending, in_progress, done).

    """
    _id_generator = itertools.count(1)

    def __init__(self, title: str, description: str = ""):
        self.id: int = next(Todo._id_generator)
        self.title: str = title
        self.description: str = description
        self.state: Literal["pending", "in_progress", "done"] = "pending"

    @classmethod
    def reset_id_counter(cls, start: int = 1) -> None:
        """Reset the ID counter. Useful for testing.

        Args:
            start: The starting value for the counter (default: 1).
        """
        cls._id_generator = itertools.count(start)

    def to_dict(self) -> dict:
        """Convert the todo to a dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "state": self.state
        }

    def __str__(self) -> str:
        """Return a string representation of the todo."""
        return f"[{self.id}] ({self.state}) {self.title}: {self.description}"


class Context:
    """A container for conversation messages.
    
    The Context class maintains an ordered list of messages exchanged
    during an agent conversation. It provides methods for adding and
    retrieving messages.
    
    Attributes:
        _messages: Internal list of BaseMessage objects.
        _hooks: Optional HookRegistry for event callbacks.

    Example:
        >>> context = Context()
        >>> context.add_message(HumanMessage(content="Hello"))
        >>> context.add_message(AIMessage(content="Hi there!"))
        >>> len(context.get_messages())
        2
    """
    
    def __init__(self, hooks: Optional["HookRegistry"] = None):
        """Initialize an empty Context.

        Args:
            hooks: Optional HookRegistry for triggering todo change events.
        """
        self._messages: list[BaseMessage] = []
        self.todos: list[Todo] = []
        self._hooks = hooks

    def add_todo(self, title: str, description: str = "") -> Todo:
        """Add a new todo item to the context.

        Args:
            title: The title of the todo item.
            description: Optional detailed description of the todo item.

        Returns:
            The created Todo object.
        """
        todo = Todo(title, description)
        self.todos.append(todo)

        # Trigger todo change hook
        if self._hooks:
            from lib.hooks import HookType, TodoChangeEvent
            self._hooks.trigger(HookType.ON_TODO_CHANGE, TodoChangeEvent(
                action="add",
                todo_id=todo.id,
                title=todo.title,
                description=todo.description,
                state=todo.state
            ))

        return todo

    def get_todos(self, state: Optional[Literal["pending", "in_progress", "done"]] = None) -> list[Todo]:
        """Get all todos, optionally filtered by state.

        Args:
            state: If provided, filter todos by this state.

        Returns:
            A list of Todo objects.
        """
        if state:
            return [todo for todo in self.todos if todo.state == state]
        return self.todos

    def get_todo_by_id(self, todo_id: int) -> Optional[Todo]:
        """Get a specific todo by its ID.

        Args:
            todo_id: The unique identifier of the todo.

        Returns:
            The Todo object if found, None otherwise.
        """
        for todo in self.todos:
            if todo.id == todo_id:
                return todo
        return None

    def update_todo(self, todo_id: int,
                    title: Optional[str] = None,
                    description: Optional[str] = None,
                    state: Optional[Literal["pending", "in_progress", "done"]] = None) -> Optional[Todo]:
        """Update an existing todo item.

        Args:
            todo_id: The unique identifier of the todo to update.
            title: New title (optional).
            description: New description (optional).
            state: New state (optional).

        Returns:
            The updated Todo object if found, None otherwise.
        """
        todo = self.get_todo_by_id(todo_id)
        if todo:
            old_state = todo.state
            if title is not None:
                todo.title = title
            if description is not None:
                todo.description = description
            if state is not None:
                todo.state = state

            # Trigger todo change hook
            if self._hooks:
                from lib.hooks import HookType, TodoChangeEvent
                self._hooks.trigger(HookType.ON_TODO_CHANGE, TodoChangeEvent(
                    action="update",
                    todo_id=todo.id,
                    title=todo.title,
                    description=todo.description,
                    state=todo.state,
                    old_state=old_state if old_state != todo.state else None
                ))

            return todo
        return None

    def delete_todo(self, todo_id: int) -> bool:
        """Delete a todo item by its ID.

        Args:
            todo_id: The unique identifier of the todo to delete.

        Returns:
            True if the todo was deleted, False if not found.
        """
        todo = self.get_todo_by_id(todo_id)
        if todo:
            # Trigger todo change hook before deletion
            if self._hooks:
                from lib.hooks import HookType, TodoChangeEvent
                self._hooks.trigger(HookType.ON_TODO_CHANGE, TodoChangeEvent(
                    action="delete",
                    todo_id=todo.id,
                    title=todo.title,
                    description=todo.description,
                    state=todo.state
                ))

            self.todos.remove(todo)
            return True
        return False

    def get_todos_summary(self) -> str:
        """Get a formatted summary of all todos for injection into prompts.

        Returns:
            A formatted string summarizing all todos.
        """
        if not self.todos:
            return "No todos in the list."

        lines = ["Current Todo List:"]
        for todo in self.todos:
            lines.append(f"  {todo}")

        pending = len([t for t in self.todos if t.state == "pending"])
        in_progress = len([t for t in self.todos if t.state == "in_progress"])
        done = len([t for t in self.todos if t.state == "done"])
        lines.append(f"\nSummary: {pending} pending, {in_progress} in progress, {done} done")

        return "\n".join(lines)

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
