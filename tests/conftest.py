"""Pytest configuration for graphent tests."""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def reset_todo_counter():
    """Reset the Todo ID counter before each test for isolation."""
    from lib.Context import Todo
    Todo.reset_id_counter(1)
    yield
    # Optionally reset after test as well
    Todo.reset_id_counter(1)

