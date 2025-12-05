"""Todo tools for managing task lists within agent conversations.

This module provides tools that allow agents to manage a todo list
for planning and task tracking purposes.
"""

from typing import Literal, Optional, Callable
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class AddTodoInput(BaseModel):
    """Input schema for adding a todo item."""
    title: str = Field(..., description="The title of the todo item")
    description: str = Field(default="", description="A detailed description of the todo item")


class UpdateTodoInput(BaseModel):
    """Input schema for updating a todo item."""
    todo_id: int = Field(..., description="The ID of the todo item to update")
    title: Optional[str] = Field(default=None, description="New title for the todo item")
    description: Optional[str] = Field(default=None, description="New description for the todo item")
    state: Optional[Literal["pending", "in_progress", "done"]] = Field(
        default=None,
        description="New state for the todo item (pending, in_progress, or done)"
    )


class DeleteTodoInput(BaseModel):
    """Input schema for deleting a todo item."""
    todo_id: int = Field(..., description="The ID of the todo item to delete")


class GetTodosInput(BaseModel):
    """Input schema for getting todos."""
    state: Optional[Literal["pending", "in_progress", "done"]] = Field(
        default=None,
        description="Filter todos by state (pending, in_progress, or done). Leave empty to get all todos."
    )


def create_todo_tools(context_getter: Callable):
    """Create todo management tools bound to a specific context.

    This function creates tools that operate on the context's todo list.
    The context_getter is a callable that returns the current Context object,
    allowing the tools to access and modify the shared todo list.

    Args:
        context_getter: A callable that returns the current Context object.

    Returns:
        A list of StructuredTool objects for todo management.
    """

    def add_todo(title: str, description: str = "") -> str:
        """Add a new todo item to the list."""
        context = context_getter()
        todo = context.add_todo(title, description)
        return f"Added todo: {todo}"

    def get_todos(state: Optional[Literal["pending", "in_progress", "done"]] = None) -> str:
        """Get all todos, optionally filtered by state."""
        context = context_getter()
        todos = context.get_todos(state)
        if not todos:
            if state:
                return f"No todos with state '{state}' found."
            return "No todos in the list."

        lines = []
        for todo in todos:
            lines.append(str(todo))
        return "\n".join(lines)

    def update_todo(
        todo_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        state: Optional[Literal["pending", "in_progress", "done"]] = None
    ) -> str:
        """Update an existing todo item."""
        context = context_getter()
        todo = context.update_todo(todo_id, title, description, state)
        if todo:
            return f"Updated todo: {todo}"
        return f"Todo with ID {todo_id} not found."

    def delete_todo(todo_id: int) -> str:
        """Delete a todo item by its ID."""
        context = context_getter()
        if context.delete_todo(todo_id):
            return f"Deleted todo with ID {todo_id}."
        return f"Todo with ID {todo_id} not found."

    add_todo_tool = StructuredTool.from_function(
        func=add_todo,
        name="add_todo",
        description="Add a new todo item to your planning list. Use this to break down tasks and track progress.",
        args_schema=AddTodoInput
    )

    get_todos_tool = StructuredTool.from_function(
        func=get_todos,
        name="get_todos",
        description="Get all todos from your planning list. Optionally filter by state (pending, in_progress, done).",
        args_schema=GetTodosInput
    )

    update_todo_tool = StructuredTool.from_function(
        func=update_todo,
        name="update_todo",
        description="Update an existing todo item. You can change the title, description, or state (pending, in_progress, done).",
        args_schema=UpdateTodoInput
    )

    delete_todo_tool = StructuredTool.from_function(
        func=delete_todo,
        name="delete_todo",
        description="Delete a todo item from your planning list by its ID.",
        args_schema=DeleteTodoInput
    )

    return [add_todo_tool, get_todos_tool, update_todo_tool, delete_todo_tool]

