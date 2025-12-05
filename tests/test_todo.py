"""Tests for the todo functionality in Context and todo tools."""

from lib.Context import Context, Todo
from lib.tools.todo_tool import create_todo_tools


class TestTodoClass:
    """Tests for the Todo class."""

    def test_todo_creation(self):
        """Test basic todo creation."""
        todo = Todo("Test task", "Test description")
        assert todo.title == "Test task"
        assert todo.description == "Test description"
        assert todo.state == "pending"
        assert todo.id > 0

    def test_todo_creation_without_description(self):
        """Test todo creation without description."""
        todo = Todo("Test task")
        assert todo.title == "Test task"
        assert todo.description == ""
        assert todo.state == "pending"

    def test_todo_unique_ids(self):
        """Test that each todo gets a unique ID."""
        todo1 = Todo("Task 1")
        todo2 = Todo("Task 2")
        assert todo1.id != todo2.id

    def test_todo_to_dict(self):
        """Test todo conversion to dictionary."""
        todo = Todo("Test task", "Test description")
        d = todo.to_dict()
        assert d["title"] == "Test task"
        assert d["description"] == "Test description"
        assert d["state"] == "pending"
        assert "id" in d

    def test_todo_str(self):
        """Test todo string representation."""
        todo = Todo("Test task", "Test description")
        s = str(todo)
        assert "Test task" in s
        assert "Test description" in s
        assert "pending" in s


class TestContextTodoMethods:
    """Tests for todo methods in Context class."""

    def test_add_todo(self):
        """Test adding a todo to context."""
        context = Context()
        todo = context.add_todo("Test task", "Description")
        assert len(context.todos) == 1
        assert context.todos[0] == todo
        assert todo.title == "Test task"

    def test_add_multiple_todos(self):
        """Test adding multiple todos."""
        context = Context()
        todo1 = context.add_todo("Task 1")
        todo2 = context.add_todo("Task 2")
        assert len(context.todos) == 2
        assert context.todos[0] == todo1
        assert context.todos[1] == todo2

    def test_get_todos_all(self):
        """Test getting all todos."""
        context = Context()
        context.add_todo("Task 1")
        context.add_todo("Task 2")
        todos = context.get_todos()
        assert len(todos) == 2

    def test_get_todos_by_state(self):
        """Test filtering todos by state."""
        context = Context()
        todo1 = context.add_todo("Task 1")
        todo2 = context.add_todo("Task 2")
        todo2.state = "done"

        pending = context.get_todos("pending")
        done = context.get_todos("done")

        assert len(pending) == 1
        assert pending[0] == todo1
        assert len(done) == 1
        assert done[0] == todo2

    def test_get_todo_by_id(self):
        """Test getting a specific todo by ID."""
        context = Context()
        todo = context.add_todo("Test task")
        found = context.get_todo_by_id(todo.id)
        assert found == todo

    def test_get_todo_by_id_not_found(self):
        """Test getting a non-existent todo."""
        context = Context()
        found = context.get_todo_by_id(99999)
        assert found is None

    def test_update_todo_title(self):
        """Test updating todo title."""
        context = Context()
        todo = context.add_todo("Original title")
        updated = context.update_todo(todo.id, title="New title")
        assert updated.title == "New title"

    def test_update_todo_description(self):
        """Test updating todo description."""
        context = Context()
        todo = context.add_todo("Task", "Old description")
        updated = context.update_todo(todo.id, description="New description")
        assert updated.description == "New description"

    def test_update_todo_state(self):
        """Test updating todo state."""
        context = Context()
        todo = context.add_todo("Task")
        updated = context.update_todo(todo.id, state="in_progress")
        assert updated.state == "in_progress"

    def test_update_todo_multiple_fields(self):
        """Test updating multiple todo fields at once."""
        context = Context()
        todo = context.add_todo("Task", "Description")
        updated = context.update_todo(
            todo.id,
            title="New title",
            description="New description",
            state="done"
        )
        assert updated.title == "New title"
        assert updated.description == "New description"
        assert updated.state == "done"

    def test_update_todo_not_found(self):
        """Test updating a non-existent todo."""
        context = Context()
        result = context.update_todo(99999, title="New title")
        assert result is None

    def test_delete_todo(self):
        """Test deleting a todo."""
        context = Context()
        todo = context.add_todo("Task")
        result = context.delete_todo(todo.id)
        assert result is True
        assert len(context.todos) == 0

    def test_delete_todo_not_found(self):
        """Test deleting a non-existent todo."""
        context = Context()
        result = context.delete_todo(99999)
        assert result is False

    def test_get_todos_summary_empty(self):
        """Test summary with no todos."""
        context = Context()
        summary = context.get_todos_summary()
        assert "No todos" in summary

    def test_get_todos_summary_with_todos(self):
        """Test summary with todos."""
        context = Context()
        context.add_todo("Task 1")
        todo2 = context.add_todo("Task 2")
        todo2.state = "in_progress"
        todo3 = context.add_todo("Task 3")
        todo3.state = "done"

        summary = context.get_todos_summary()
        assert "Task 1" in summary
        assert "Task 2" in summary
        assert "Task 3" in summary
        assert "1 pending" in summary
        assert "1 in progress" in summary
        assert "1 done" in summary


class TestTodoTools:
    """Tests for the todo tools."""

    def test_create_todo_tools(self):
        """Test that create_todo_tools returns the expected tools."""
        context = Context()
        tools = create_todo_tools(lambda: context)

        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "add_todo" in tool_names
        assert "get_todos" in tool_names
        assert "update_todo" in tool_names
        assert "delete_todo" in tool_names

    def test_add_todo_tool(self):
        """Test the add_todo tool."""
        context = Context()
        tools = create_todo_tools(lambda: context)
        add_tool = next(t for t in tools if t.name == "add_todo")

        result = add_tool.invoke({"title": "Test task", "description": "Test desc"})
        assert "Test task" in result
        assert len(context.todos) == 1

    def test_get_todos_tool(self):
        """Test the get_todos tool."""
        context = Context()
        context.add_todo("Task 1")
        context.add_todo("Task 2")

        tools = create_todo_tools(lambda: context)
        get_tool = next(t for t in tools if t.name == "get_todos")

        result = get_tool.invoke({})
        assert "Task 1" in result
        assert "Task 2" in result

    def test_get_todos_tool_filtered(self):
        """Test the get_todos tool with state filter."""
        context = Context()
        context.add_todo("Task 1")
        todo2 = context.add_todo("Task 2")
        todo2.state = "done"

        tools = create_todo_tools(lambda: context)
        get_tool = next(t for t in tools if t.name == "get_todos")

        result = get_tool.invoke({"state": "done"})
        assert "Task 2" in result
        assert "Task 1" not in result

    def test_get_todos_tool_empty(self):
        """Test the get_todos tool with no todos."""
        context = Context()
        tools = create_todo_tools(lambda: context)
        get_tool = next(t for t in tools if t.name == "get_todos")

        result = get_tool.invoke({})
        assert "No todos" in result

    def test_update_todo_tool(self):
        """Test the update_todo tool."""
        context = Context()
        todo = context.add_todo("Original")

        tools = create_todo_tools(lambda: context)
        update_tool = next(t for t in tools if t.name == "update_todo")

        result = update_tool.invoke({
            "todo_id": todo.id,
            "title": "Updated",
            "state": "in_progress"
        })
        assert "Updated" in result
        assert context.todos[0].title == "Updated"
        assert context.todos[0].state == "in_progress"

    def test_update_todo_tool_not_found(self):
        """Test the update_todo tool with invalid ID."""
        context = Context()
        tools = create_todo_tools(lambda: context)
        update_tool = next(t for t in tools if t.name == "update_todo")

        result = update_tool.invoke({"todo_id": 99999, "title": "New"})
        assert "not found" in result

    def test_delete_todo_tool(self):
        """Test the delete_todo tool."""
        context = Context()
        todo = context.add_todo("To be deleted")

        tools = create_todo_tools(lambda: context)
        delete_tool = next(t for t in tools if t.name == "delete_todo")

        result = delete_tool.invoke({"todo_id": todo.id})
        assert "Deleted" in result
        assert len(context.todos) == 0

    def test_delete_todo_tool_not_found(self):
        """Test the delete_todo tool with invalid ID."""
        context = Context()
        tools = create_todo_tools(lambda: context)
        delete_tool = next(t for t in tools if t.name == "delete_todo")

        result = delete_tool.invoke({"todo_id": 99999})
        assert "not found" in result


class TestTodoHooks:
    """Tests for todo change hooks."""

    def test_add_todo_triggers_hook(self):
        """Test that add_todo triggers the ON_TODO_CHANGE hook."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType, TodoChangeEvent

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        todo = context.add_todo("Test task", "Test description")

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, TodoChangeEvent)
        assert event.action == "add"
        assert event.todo_id == todo.id
        assert event.title == "Test task"
        assert event.description == "Test description"
        assert event.state == "pending"

    def test_update_todo_triggers_hook(self):
        """Test that update_todo triggers the ON_TODO_CHANGE hook."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType, TodoChangeEvent

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        todo = context.add_todo("Test task")
        callback.reset_mock()  # Reset to ignore add call

        context.update_todo(todo.id, title="Updated", state="in_progress")

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, TodoChangeEvent)
        assert event.action == "update"
        assert event.todo_id == todo.id
        assert event.title == "Updated"
        assert event.state == "in_progress"
        assert event.old_state == "pending"

    def test_update_todo_no_state_change(self):
        """Test that update_todo with no state change has old_state as None."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType, TodoChangeEvent

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        todo = context.add_todo("Test task")
        callback.reset_mock()

        context.update_todo(todo.id, title="Updated")  # Only title change

        event = callback.call_args[0][0]
        assert event.old_state is None  # No state change

    def test_delete_todo_triggers_hook(self):
        """Test that delete_todo triggers the ON_TODO_CHANGE hook."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType, TodoChangeEvent

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        todo = context.add_todo("Test task", "Description")
        todo_id = todo.id
        callback.reset_mock()

        context.delete_todo(todo_id)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, TodoChangeEvent)
        assert event.action == "delete"
        assert event.todo_id == todo_id
        assert event.title == "Test task"

    def test_no_hook_without_registry(self):
        """Test that todo operations work without a hook registry."""
        context = Context()  # No hooks passed
        todo = context.add_todo("Test task")
        context.update_todo(todo.id, title="Updated")
        context.delete_todo(todo.id)
        # Should not raise any errors

    def test_update_nonexistent_todo_no_hook(self):
        """Test that updating a non-existent todo doesn't trigger hook."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        context.update_todo(99999, title="New")  # Non-existent

        callback.assert_not_called()

    def test_delete_nonexistent_todo_no_hook(self):
        """Test that deleting a non-existent todo doesn't trigger hook."""
        from unittest.mock import MagicMock
        from lib.hooks import HookRegistry, HookType

        hooks = HookRegistry()
        callback = MagicMock()
        hooks.register(HookType.ON_TODO_CHANGE, callback)

        context = Context(hooks=hooks)
        context.delete_todo(99999)  # Non-existent

        callback.assert_not_called()


