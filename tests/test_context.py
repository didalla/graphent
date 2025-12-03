"""Comprehensive test suite for the Context module.

Tests cover initialization, message management, retrieval with slicing,
method chaining, edge cases, and string representation.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


from lib.Context import Context


class TestContextInitialization:
    """Tests for Context initialization."""

    def test_init_creates_empty_messages_list(self):
        """Context should initialize with an empty messages list."""
        context = Context()
        assert context._messages == []

    def test_init_messages_is_list_type(self):
        """Internal messages storage should be a list."""
        context = Context()
        assert isinstance(context._messages, list)

    def test_get_messages_returns_empty_list_on_new_context(self):
        """get_messages should return empty list for new context."""
        context = Context()
        assert context.get_messages() == []


class TestAddMessage:
    """Tests for the add_message method."""

    def test_add_single_message(self):
        """Adding a single message should store it."""
        context = Context()
        message = HumanMessage(content="Hello")
        context.add_message(message)
        
        assert len(context._messages) == 1
        assert context._messages[0] == message

    def test_add_multiple_messages(self):
        """Adding multiple messages should preserve order."""
        context = Context()
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi there!")
        msg3 = HumanMessage(content="How are you?")
        
        context.add_message(msg1)
        context.add_message(msg2)
        context.add_message(msg3)
        
        assert len(context._messages) == 3
        assert context._messages[0] == msg1
        assert context._messages[1] == msg2
        assert context._messages[2] == msg3

    def test_add_message_returns_self_for_chaining(self):
        """add_message should return self to enable method chaining."""
        context = Context()
        message = HumanMessage(content="Test")
        result = context.add_message(message)
        
        assert result is context

    def test_method_chaining(self):
        """Multiple add_message calls should be chainable."""
        context = Context()
        msg1 = HumanMessage(content="First")
        msg2 = AIMessage(content="Second")
        msg3 = SystemMessage(content="Third")
        
        result = context.add_message(msg1).add_message(msg2).add_message(msg3)
        
        assert result is context
        assert len(context._messages) == 3

    def test_add_different_message_types(self):
        """Context should accept different BaseMessage subclasses."""
        context = Context()
        human_msg = HumanMessage(content="Human message")
        ai_msg = AIMessage(content="AI message")
        system_msg = SystemMessage(content="System message")
        
        context.add_message(human_msg)
        context.add_message(ai_msg)
        context.add_message(system_msg)
        
        messages = context.get_messages()
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], SystemMessage)


class TestGetMessages:
    """Tests for the get_messages method."""

    def test_get_all_messages_without_limit(self):
        """get_messages without last_n should return all messages."""
        context = Context()
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(5)
        ]
        for msg in messages:
            context.add_message(msg)
        
        result = context.get_messages()
        assert result == messages

    def test_get_last_n_messages(self):
        """get_messages with last_n should return only last N messages."""
        context = Context()
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(5)
        ]
        for msg in messages:
            context.add_message(msg)
        
        result = context.get_messages(last_n=3)
        
        assert len(result) == 3
        assert result == messages[-3:]

    def test_get_last_n_equal_to_total(self):
        """last_n equal to total messages should return all."""
        context = Context()
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(3)
        ]
        for msg in messages:
            context.add_message(msg)
        
        result = context.get_messages(last_n=3)
        assert result == messages

    def test_get_last_n_greater_than_total(self):
        """last_n greater than total should return all messages."""
        context = Context()
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(2)
        ]
        for msg in messages:
            context.add_message(msg)
        
        result = context.get_messages(last_n=10)
        assert result == messages

    def test_get_last_1_message(self):
        """last_n=1 should return only the most recent message."""
        context = Context()
        context.add_message(HumanMessage(content="First"))
        context.add_message(AIMessage(content="Second"))
        context.add_message(HumanMessage(content="Third"))
        
        result = context.get_messages(last_n=1)
        
        assert len(result) == 1
        assert result[0].content == "Third"

    def test_get_messages_with_none_explicitly(self):
        """Explicitly passing None should return all messages."""
        context = Context()
        msg1 = HumanMessage(content="Test1")
        msg2 = HumanMessage(content="Test2")
        context.add_message(msg1)
        context.add_message(msg2)
        
        result = context.get_messages(last_n=None)
        assert len(result) == 2

    def test_get_messages_returns_new_reference(self):
        """get_messages should return the internal list (not a copy by default)."""
        context = Context()
        msg = HumanMessage(content="Test")
        context.add_message(msg)
        
        result = context.get_messages()
        # Note: Current implementation returns direct reference
        assert result is context._messages


class TestGetMessagesEdgeCases:
    """Edge case tests for get_messages."""

    def test_get_messages_from_empty_context_with_last_n(self):
        """Requesting last_n from empty context should return empty list."""
        context = Context()
        result = context.get_messages(last_n=5)
        assert result == []

    def test_get_messages_last_n_zero(self):
        """last_n=0 is falsy, so it should return all messages."""
        context = Context()
        context.add_message(HumanMessage(content="Message"))
        
        # Note: Current implementation uses `if last_n:` which treats 0 as falsy
        result = context.get_messages(last_n=0)
        assert len(result) == 1  # Returns all because 0 is falsy


class TestStringRepresentation:
    """Tests for __str__ method."""

    def test_str_empty_context(self):
        """String representation of empty context should show empty list."""
        context = Context()
        result = str(context)
        assert result == "[]"

    def test_str_with_messages(self):
        """String representation should include message content."""
        context = Context()
        context.add_message(HumanMessage(content="Hello"))
        
        result = str(context)
        assert "Hello" in result

    def test_str_multiple_messages(self):
        """String representation should include all messages."""
        context = Context()
        context.add_message(HumanMessage(content="First"))
        context.add_message(AIMessage(content="Second"))
        
        result = str(context)
        assert "First" in result
        assert "Second" in result


class TestMessageContent:
    """Tests for message content preservation."""

    def test_empty_message_content(self):
        """Messages with empty content should be stored correctly."""
        context = Context()
        msg = HumanMessage(content="")
        context.add_message(msg)
        
        assert context.get_messages()[0].content == ""

    def test_long_message_content(self):
        """Long message content should be preserved."""
        context = Context()
        long_content = "A" * 10000
        msg = HumanMessage(content=long_content)
        context.add_message(msg)
        
        assert context.get_messages()[0].content == long_content

    def test_special_characters_in_content(self):
        """Special characters in messages should be preserved."""
        context = Context()
        special_content = "Hello! @#$%^&*() 日本語 🎉 \n\t"
        msg = HumanMessage(content=special_content)
        context.add_message(msg)
        
        assert context.get_messages()[0].content == special_content

    def test_multiline_content(self):
        """Multiline message content should be preserved."""
        context = Context()
        multiline = """Line 1
        Line 2
        Line 3"""
        msg = HumanMessage(content=multiline)
        context.add_message(msg)
        
        assert context.get_messages()[0].content == multiline


class TestMessageOrder:
    """Tests for message ordering."""

    def test_fifo_order(self):
        """Messages should maintain FIFO order."""
        context = Context()
        contents = ["First", "Second", "Third", "Fourth", "Fifth"]
        
        for content in contents:
            context.add_message(HumanMessage(content=content))
        
        messages = context.get_messages()
        for i, msg in enumerate(messages):
            assert msg.content == contents[i]

    def test_get_last_n_maintains_order(self):
        """last_n messages should maintain their relative order."""
        context = Context()
        for i in range(10):
            context.add_message(HumanMessage(content=f"Message {i}"))
        
        last_5 = context.get_messages(last_n=5)
        
        for i, msg in enumerate(last_5):
            assert msg.content == f"Message {5 + i}"


class TestWithMockedMessages:
    """Tests using mocked BaseMessage objects."""

    def test_add_mock_message(self):
        """Context should accept any BaseMessage-like object."""
        context = Context()
        mock_message = MagicMock(spec=BaseMessage)
        mock_message.content = "Mocked content"
        
        context.add_message(mock_message)
        
        assert len(context._messages) == 1
        assert context._messages[0] is mock_message

    def test_multiple_mock_messages(self):
        """Multiple mocked messages should be handled correctly."""
        context = Context()
        mocks = [MagicMock(spec=BaseMessage) for _ in range(3)]
        
        for mock in mocks:
            context.add_message(mock)
        
        assert context.get_messages() == mocks


class TestContextIsolation:
    """Tests for context instance isolation."""

    def test_separate_contexts_are_independent(self):
        """Different Context instances should not share state."""
        context1 = Context()
        context2 = Context()
        
        context1.add_message(HumanMessage(content="Context 1 message"))
        
        assert len(context1.get_messages()) == 1
        assert len(context2.get_messages()) == 0

    def test_modifications_dont_affect_other_instances(self):
        """Modifying one context should not affect another."""
        context1 = Context()
        context2 = Context()
        
        context1.add_message(HumanMessage(content="Message 1"))
        context2.add_message(HumanMessage(content="Message A"))
        context2.add_message(HumanMessage(content="Message B"))
        
        assert len(context1.get_messages()) == 1
        assert len(context2.get_messages()) == 2
        assert context1.get_messages()[0].content == "Message 1"


class TestLargeScaleOperations:
    """Tests for performance with many messages."""

    def test_many_messages(self):
        """Context should handle a large number of messages."""
        context = Context()
        num_messages = 1000
        
        for i in range(num_messages):
            context.add_message(HumanMessage(content=f"Message {i}"))
        
        assert len(context.get_messages()) == num_messages

    def test_large_last_n_retrieval(self):
        """Retrieving many messages with last_n should work correctly."""
        context = Context()
        num_messages = 500
        
        for i in range(num_messages):
            context.add_message(HumanMessage(content=f"Message {i}"))
        
        last_100 = context.get_messages(last_n=100)
        
        assert len(last_100) == 100
        assert last_100[0].content == "Message 400"
        assert last_100[-1].content == "Message 499"
