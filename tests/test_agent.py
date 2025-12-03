"""Comprehensive test suite for the Agent module.

Tests cover initialization, tool binding, system prompt setup,
sub-agent delegation, invoke/ainvoke methods, and edge cases.
All model invocations are mocked.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from lib.Agent import (
    Agent,
    AgentHandOff,
    AGENT_DELEGATION_PROMPT_HEADER,
    AGENT_DELEGATION_AGENT_TEMPLATE,
)
from lib.Context import Context

# Configure pytest-asyncio
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock BaseChatModel."""
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_response_no_tools():
    """Create a mock AI response with no tool calls."""
    response = MagicMock(spec=AIMessage)
    response.content = "Hello! How can I help you?"
    response.tool_calls = []
    return response


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "test_tool"
    tool.description = "A test tool"
    return tool


@pytest.fixture
def basic_agent(mock_model, mock_response_no_tools):
    """Create a basic agent without tools or sub-agents."""
    mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
    return Agent(
        name="TestAgent",
        model=mock_model,
        system_prompt="You are a test agent.",
        description="A test agent for unit testing"
    )


@pytest.fixture
def agent_with_tools(mock_model, mock_tool, mock_response_no_tools):
    """Create an agent with tools."""
    mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
    return Agent(
        name="ToolAgent",
        model=mock_model,
        system_prompt="You are an agent with tools.",
        description="An agent with tools for testing",
        tools=[mock_tool]
    )


# ============================================================================
# Test Classes
# ============================================================================

class TestAgentHandOffSchema:
    """Tests for the AgentHandOff Pydantic model."""

    def test_valid_handoff_schema(self):
        """AgentHandOff should accept valid agent_name and task."""
        handoff = AgentHandOff(agent_name="SubAgent", task="Do something")
        assert handoff.agent_name == "SubAgent"
        assert handoff.task == "Do something"

    def test_handoff_requires_agent_name(self):
        """AgentHandOff should require agent_name field."""
        # Test that the field is defined as required (has no default)
        assert AgentHandOff.model_fields['agent_name'].is_required()

    def test_handoff_requires_task(self):
        """AgentHandOff should require task field."""
        # Test that the field is defined as required (has no default)
        assert AgentHandOff.model_fields['task'].is_required()


class TestAgentInitialization:
    """Tests for Agent initialization."""

    def test_basic_initialization(self, mock_model):
        """Agent should initialize with required parameters."""
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Test description"
        )
        
        assert agent.name == "TestAgent"
        assert agent.description == "Test description"
        assert "Test prompt" in agent.system_prompt
        assert agent.tools is None
        assert agent.callable_agents is None

    def test_initialization_with_tools(self, mock_model, mock_tool):
        """Agent should initialize with tools and bind them to model."""
        agent = Agent(
            name="ToolAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Test description",
            tools=[mock_tool]
        )
        
        assert agent.tools == [mock_tool]
        mock_model.bind_tools.assert_called_once_with([mock_tool])

    def test_initialization_without_tools_no_binding(self, mock_model):
        """Agent without tools should not call bind_tools."""
        agent = Agent(
            name="NoToolAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Test description",
            tools=None
        )
        
        assert agent.tools is None
        mock_model.bind_tools.assert_not_called()

    def test_initialization_with_callable_agents(self, mock_model):
        """Agent with callable_agents should add hand_off_to_subagent tool."""
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "SubAgent"
        sub_agent.description = "A sub-agent"
        
        agent = Agent(
            name="ParentAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Parent agent",
            callable_agents=[sub_agent]
        )
        
        assert agent.callable_agents == [sub_agent]
        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "hand_off_to_subagent"

    def test_initialization_with_tools_and_callable_agents(self, mock_model, mock_tool):
        """Agent with both tools and callable_agents should have all tools."""
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "SubAgent"
        sub_agent.description = "A sub-agent"
        
        agent = Agent(
            name="ParentAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Parent agent",
            tools=[mock_tool],
            callable_agents=[sub_agent]
        )
        
        assert agent.tools is not None
        assert len(agent.tools) == 2
        tool_names = [t.name for t in agent.tools]
        assert "test_tool" in tool_names
        assert "hand_off_to_subagent" in tool_names


class TestSystemPromptSetup:
    """Tests for system prompt configuration."""

    def test_system_prompt_without_callable_agents(self, mock_model):
        """System prompt should remain unchanged without callable_agents."""
        original_prompt = "You are a helpful assistant."
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt=original_prompt,
            description="Test"
        )
        
        assert agent.system_prompt == original_prompt

    def test_system_prompt_with_callable_agents(self, mock_model):
        """System prompt should include delegation instructions with callable_agents."""
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helps with tasks"
        
        original_prompt = "You are a helpful assistant."
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt=original_prompt,
            description="Test",
            callable_agents=[sub_agent]
        )
        
        assert original_prompt in agent.system_prompt
        assert AGENT_DELEGATION_PROMPT_HEADER in agent.system_prompt
        assert "HelperAgent" in agent.system_prompt
        assert "Helps with tasks" in agent.system_prompt

    def test_system_prompt_with_multiple_callable_agents(self, mock_model):
        """System prompt should list all callable agents."""
        sub_agent1 = MagicMock(spec=Agent)
        sub_agent1.name = "Agent1"
        sub_agent1.description = "First agent"
        
        sub_agent2 = MagicMock(spec=Agent)
        sub_agent2.name = "Agent2"
        sub_agent2.description = "Second agent"
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Base prompt",
            description="Test",
            callable_agents=[sub_agent1, sub_agent2]
        )
        
        assert "Agent1" in agent.system_prompt
        assert "First agent" in agent.system_prompt
        assert "Agent2" in agent.system_prompt
        assert "Second agent" in agent.system_prompt

    def test_set_up_system_prompt_static_method(self):
        """_set_up_system_prompt should work as static method."""
        result = Agent._set_up_system_prompt("Base prompt", None)
        assert result == "Base prompt"

    def test_set_up_system_prompt_with_agents(self):
        """_set_up_system_prompt should append agent info."""
        sub_agent = MagicMock()
        sub_agent.name = "TestSub"
        sub_agent.description = "Test description"
        
        result = Agent._set_up_system_prompt("Base prompt", [sub_agent])
        
        assert "Base prompt" in result
        assert "TestSub" in result
        assert "Test description" in result


class TestHandOffToSubagent:
    """Tests for sub-agent delegation."""

    def test_hand_off_no_callable_agents(self, mock_model):
        """Hand off should return error when no callable_agents."""
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        result = agent._hand_off_to_subagent("SomeAgent", "Do task")
        assert "not available" in result

    def test_hand_off_agent_not_found(self, mock_model):
        """Hand off should return error when agent name not found."""
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helper"
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent]
        )
        
        result = agent._hand_off_to_subagent("NonExistentAgent", "Do task")
        assert "not available" in result

    def test_hand_off_duplicate_agent_names(self, mock_model):
        """Hand off should return error when multiple agents have same name."""
        sub_agent1 = MagicMock(spec=Agent)
        sub_agent1.name = "DuplicateName"
        sub_agent1.description = "First"
        
        sub_agent2 = MagicMock(spec=Agent)
        sub_agent2.name = "DuplicateName"
        sub_agent2.description = "Second"
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent1, sub_agent2]
        )
        
        result = agent._hand_off_to_subagent("DuplicateName", "Do task")
        assert "More than one agent" in result

    def test_hand_off_successful(self, mock_model):
        """Hand off should invoke sub-agent and return result."""
        # Create a mock response for the sub-agent
        sub_response = MagicMock()
        sub_response.content = "Task completed successfully"
        
        sub_context = MagicMock(spec=Context)
        sub_context.get_messages = MagicMock(return_value=[sub_response])
        
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helper"
        sub_agent.invoke = MagicMock(return_value=sub_context)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent]
        )
        
        result = agent._hand_off_to_subagent("HelperAgent", "Do task")
        
        assert result == "Task completed successfully"
        sub_agent.invoke.assert_called_once()

    def test_hand_off_result_is_list(self, mock_model):
        """Hand off should handle when content is a list."""
        sub_response = MagicMock()
        sub_response.content = ["item1", "item2"]
        
        sub_context = MagicMock(spec=Context)
        sub_context.get_messages = MagicMock(return_value=[sub_response])
        
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helper"
        sub_agent.invoke = MagicMock(return_value=sub_context)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent]
        )
        
        result = agent._hand_off_to_subagent("HelperAgent", "Do task")
        
        assert result == "['item1', 'item2']"


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncHandOffToSubagent:
    """Tests for async sub-agent delegation."""

    @pytest.mark.asyncio
    async def test_async_hand_off_no_callable_agents(self, mock_model):
        """Async hand off should return error when no callable_agents."""
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        result = await agent._ahand_off_to_subagent("SomeAgent", "Do task")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_async_hand_off_agent_not_found(self, mock_model):
        """Async hand off should return error when agent name not found."""
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helper"
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent]
        )
        
        result = await agent._ahand_off_to_subagent("NonExistentAgent", "Do task")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_async_hand_off_successful(self, mock_model):
        """Async hand off should invoke sub-agent and return result."""
        sub_response = MagicMock()
        sub_response.content = "Async task completed"
        
        sub_context = MagicMock(spec=Context)
        sub_context.get_messages = MagicMock(return_value=[sub_response])
        
        sub_agent = MagicMock(spec=Agent)
        sub_agent.name = "HelperAgent"
        sub_agent.description = "Helper"
        sub_agent.ainvoke = AsyncMock(return_value=sub_context)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            callable_agents=[sub_agent]
        )
        
        result = await agent._ahand_off_to_subagent("HelperAgent", "Do task")
        
        assert result == "Async task completed"
        sub_agent.ainvoke.assert_called_once()


class TestInvoke:
    """Tests for the invoke method."""

    def test_invoke_simple_response(self, mock_model, mock_response_no_tools):
        """Invoke should return context with AI response."""
        mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        result = agent.invoke(context)
        
        assert len(result.get_messages()) == 2  # Human + AI
        mock_model.invoke.assert_called_once()

    def test_invoke_includes_system_message(self, mock_model, mock_response_no_tools):
        """Invoke should include system message in chat history."""
        mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Be helpful",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        agent.invoke(context)
        
        call_args = mock_model.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "Be helpful"

    def test_invoke_max_iterations_reached(self, mock_model):
        """Invoke should stop and add message when max iterations reached."""
        # Create a response that always has tool calls to force iteration
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "1", "name": "some_tool", "args": {}}]
        
        mock_model.invoke = MagicMock(return_value=response_with_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        result = agent.invoke(context, max_iterations=0)
        
        messages = result.get_messages()
        last_message = messages[-1]
        assert "maximum number of steps" in last_message.content

    def test_invoke_with_tool_call(self, mock_model, mock_tool):
        """Invoke should execute tool calls and recurse."""
        # First response has tool call
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "call_1", "name": "test_tool", "args": {"input": "test"}}]
        
        # Second response has no tool calls (end recursion)
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Here is the result"
        final_response.tool_calls = []
        
        mock_model.invoke = MagicMock(side_effect=[response_with_tools, final_response])
        
        # Mock tool result
        tool_result = ToolMessage(content="Tool output", tool_call_id="call_1")
        mock_tool.invoke = MagicMock(return_value=tool_result)
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool]
        )
        
        context = Context().add_message(HumanMessage(content="Use the tool"))
        result = agent.invoke(context)
        
        # Should have: Human + AI(tool_call) + ToolMessage + AI(final)
        assert mock_model.invoke.call_count == 2
        mock_tool.invoke.assert_called_once()

    def test_invoke_tool_not_found(self, mock_model, mock_tool):
        """Invoke should add error message when tool not found."""
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "call_1", "name": "nonexistent_tool", "args": {}}]
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Done"
        final_response.tool_calls = []
        
        mock_model.invoke = MagicMock(side_effect=[response_with_tools, final_response])
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool]
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        result = agent.invoke(context)
        
        messages = result.get_messages()
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert any("not available" in m.content for m in tool_messages)

    def test_invoke_duplicate_tool_names(self, mock_model):
        """Invoke should add error message when multiple tools have same name."""
        mock_tool1 = MagicMock(spec=BaseTool)
        mock_tool1.name = "duplicate_tool"
        mock_tool2 = MagicMock(spec=BaseTool)
        mock_tool2.name = "duplicate_tool"
        
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "call_1", "name": "duplicate_tool", "args": {}}]
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Done"
        final_response.tool_calls = []
        
        mock_model.invoke = MagicMock(side_effect=[response_with_tools, final_response])
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool1, mock_tool2]
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        result = agent.invoke(context)
        
        messages = result.get_messages()
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert any("More than one tool" in m.content for m in tool_messages)

    def test_invoke_stores_last_response(self, mock_model, mock_response_no_tools):
        """Invoke should store the last response in _last_response."""
        mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        agent.invoke(context)
        
        assert agent._last_response is mock_response_no_tools


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncInvoke:
    """Tests for the ainvoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_simple_response(self, mock_model, mock_response_no_tools):
        """Async invoke should return context with AI response."""
        mock_model.ainvoke = AsyncMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test prompt",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        result = await agent.ainvoke(context)
        
        assert len(result.get_messages()) == 2
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_max_iterations_reached(self, mock_model):
        """Async invoke should stop when max iterations reached."""
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "1", "name": "some_tool", "args": {}}]
        
        mock_model.ainvoke = AsyncMock(return_value=response_with_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        context = Context().add_message(HumanMessage(content="Hello"))
        result = await agent.ainvoke(context, max_iterations=0)
        
        messages = result.get_messages()
        last_message = messages[-1]
        assert "maximum number of steps" in last_message.content

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_tool(self, mock_model, mock_tool):
        """Async invoke should use ainvoke on tools that support it."""
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "call_1", "name": "test_tool", "args": {}}]
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Done"
        final_response.tool_calls = []
        
        mock_model.ainvoke = AsyncMock(side_effect=[response_with_tools, final_response])
        
        tool_result = ToolMessage(content="Async tool output", tool_call_id="call_1")
        mock_tool.ainvoke = AsyncMock(return_value=tool_result)
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool]
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        await agent.ainvoke(context)
        
        mock_tool.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_fallback_to_sync_tool(self, mock_model):
        """Async invoke should fall back to sync invoke for tools without ainvoke."""
        mock_tool_sync = MagicMock(spec=BaseTool)
        mock_tool_sync.name = "sync_tool"
        # Remove ainvoke to simulate sync-only tool
        del mock_tool_sync.ainvoke
        
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"id": "call_1", "name": "sync_tool", "args": {}}]
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Done"
        final_response.tool_calls = []
        
        mock_model.ainvoke = AsyncMock(side_effect=[response_with_tools, final_response])
        
        tool_result = ToolMessage(content="Sync tool output", tool_call_id="call_1")
        mock_tool_sync.invoke = MagicMock(return_value=tool_result)
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool_sync]
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        await agent.ainvoke(context)
        
        mock_tool_sync.invoke.assert_called_once()


class TestStringRepresentation:
    """Tests for __str__ method."""

    def test_str_basic_agent(self, mock_model):
        """String representation should include name and model."""
        agent = Agent(
            name="MyAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        result = str(agent)
        
        assert "Agent:" in result
        assert "MyAgent" in result

    def test_str_agent_with_tools(self, mock_model, mock_tool):
        """String representation should include tools."""
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        agent = Agent(
            name="ToolAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool]
        )
        
        result = str(agent)
        
        assert "tools:" in result


class TestMultipleToolCalls:
    """Tests for handling multiple tool calls in one response."""

    def test_invoke_multiple_tool_calls(self, mock_model):
        """Invoke should handle multiple tool calls in one response."""
        mock_tool1 = MagicMock(spec=BaseTool)
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock(spec=BaseTool)
        mock_tool2.name = "tool2"
        
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [
            {"id": "call_1", "name": "tool1", "args": {}},
            {"id": "call_2", "name": "tool2", "args": {}}
        ]
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Both tools used"
        final_response.tool_calls = []
        
        mock_model.invoke = MagicMock(side_effect=[response_with_tools, final_response])
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        tool1_result = ToolMessage(content="Tool 1 result", tool_call_id="call_1")
        tool2_result = ToolMessage(content="Tool 2 result", tool_call_id="call_2")
        mock_tool1.invoke = MagicMock(return_value=tool1_result)
        mock_tool2.invoke = MagicMock(return_value=tool2_result)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool1, mock_tool2]
        )
        
        context = Context().add_message(HumanMessage(content="Use both tools"))
        result = agent.invoke(context)
        
        mock_tool1.invoke.assert_called_once()
        mock_tool2.invoke.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_tool_list(self, mock_model, mock_response_no_tools):
        """Agent with empty tool list should work like no tools."""
        mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[]
        )
        
        # Empty list is still bound
        mock_model.bind_tools.assert_called_once_with([])
        
        context = Context().add_message(HumanMessage(content="Hello"))
        result = agent.invoke(context)
        assert len(result.get_messages()) == 2

    def test_tool_call_missing_id(self, mock_model, mock_tool):
        """Invoke should handle tool calls with missing id."""
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = ""
        response_with_tools.tool_calls = [{"name": "test_tool", "args": {}}]  # No 'id' key
        
        final_response = MagicMock(spec=AIMessage)
        final_response.content = "Done"
        final_response.tool_calls = []
        
        mock_model.invoke = MagicMock(side_effect=[response_with_tools, final_response])
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        
        tool_result = ToolMessage(content="Output", tool_call_id="")
        mock_tool.invoke = MagicMock(return_value=tool_result)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=[mock_tool]
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        # Should not raise an error
        result = agent.invoke(context)
        assert result is not None

    def test_invoke_preserves_context_messages(self, mock_model, mock_response_no_tools):
        """Invoke should preserve existing context messages."""
        mock_model.invoke = MagicMock(return_value=mock_response_no_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test"
        )
        
        context = Context()
        context.add_message(HumanMessage(content="First message"))
        context.add_message(AIMessage(content="First response"))
        context.add_message(HumanMessage(content="Second message"))
        
        result = agent.invoke(context)
        
        messages = result.get_messages()
        assert len(messages) == 4  # 3 original + 1 new AI response
        assert messages[0].content == "First message"
        assert messages[1].content == "First response"
        assert messages[2].content == "Second message"


class TestToolCallGuard:
    """Tests for the None tools guard in invoke/ainvoke."""

    def test_invoke_with_tool_calls_but_no_tools(self, mock_model):
        """Invoke should return early if tool_calls exist but tools is None."""
        response_with_tools = MagicMock(spec=AIMessage)
        response_with_tools.content = "I'll use a tool"
        response_with_tools.tool_calls = [{"id": "1", "name": "some_tool", "args": {}}]
        
        mock_model.invoke = MagicMock(return_value=response_with_tools)
        
        agent = Agent(
            name="TestAgent",
            model=mock_model,
            system_prompt="Test",
            description="Test",
            tools=None
        )
        
        context = Context().add_message(HumanMessage(content="Test"))
        result = agent.invoke(context)
        
        # Should return without processing tool calls or recursing
        assert mock_model.invoke.call_count == 1
