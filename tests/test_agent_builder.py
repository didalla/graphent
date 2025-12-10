"""Comprehensive test suite for the AgentBuilder module.

Tests cover initialization, fluent API methods, builder pattern,
validation, and edge cases. All model invocations are mocked.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from lib.AgentBuilder import AgentBuilder
from lib.Agent import Agent
from lib.exceptions import AgentConfigurationError


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
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "test_tool"
    tool.description = "A test tool"
    return tool


@pytest.fixture
def mock_agent():
    """Create a mock Agent for sub-agent testing."""
    agent = MagicMock(spec=Agent)
    agent.name = "SubAgent"
    agent.description = "A sub-agent"
    return agent


@pytest.fixture
def complete_builder(mock_model):
    """Create a builder with all required fields set."""
    return (
        AgentBuilder()
        .with_name("TestAgent")
        .with_model(mock_model)
        .with_system_prompt("You are a test agent.")
        .with_description("A test agent for testing")
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestAgentBuilderInitialization:
    """Tests for AgentBuilder initialization."""

    def test_init_creates_empty_builder(self):
        """AgentBuilder should initialize with None/empty values."""
        builder = AgentBuilder()

        assert builder._name is None
        assert builder._model is None
        assert builder._system_prompt is None
        assert builder._description is None
        assert builder._tools == []
        assert builder._callable_agents == []

    def test_init_tools_is_empty_list(self):
        """Tools should be an empty list, not None."""
        builder = AgentBuilder()
        assert isinstance(builder._tools, list)
        assert len(builder._tools) == 0

    def test_init_callable_agents_is_empty_list(self):
        """Callable agents should be an empty list, not None."""
        builder = AgentBuilder()
        assert isinstance(builder._callable_agents, list)
        assert len(builder._callable_agents) == 0


class TestWithName:
    """Tests for the with_name method."""

    def test_with_name_sets_name(self):
        """with_name should set the agent name."""
        builder = AgentBuilder().with_name("MyAgent")
        assert builder._name == "MyAgent"

    def test_with_name_returns_self(self):
        """with_name should return self for chaining."""
        builder = AgentBuilder()
        result = builder.with_name("MyAgent")
        assert result is builder

    def test_with_name_empty_string(self):
        """with_name should accept empty string."""
        builder = AgentBuilder().with_name("")
        assert builder._name == ""

    def test_with_name_overwrites_previous(self):
        """Calling with_name again should overwrite previous value."""
        builder = AgentBuilder().with_name("First").with_name("Second")
        assert builder._name == "Second"


class TestWithModel:
    """Tests for the with_model method."""

    def test_with_model_sets_model(self, mock_model):
        """with_model should set the model."""
        builder = AgentBuilder().with_model(mock_model)
        assert builder._model is mock_model

    def test_with_model_returns_self(self, mock_model):
        """with_model should return self for chaining."""
        builder = AgentBuilder()
        result = builder.with_model(mock_model)
        assert result is builder

    def test_with_model_overwrites_previous(self, mock_model):
        """Calling with_model again should overwrite previous value."""
        model2 = MagicMock(spec=BaseChatModel)
        builder = AgentBuilder().with_model(mock_model).with_model(model2)
        assert builder._model is model2


class TestWithSystemPrompt:
    """Tests for the with_system_prompt method."""

    def test_with_system_prompt_sets_prompt(self):
        """with_system_prompt should set the system prompt."""
        builder = AgentBuilder().with_system_prompt("Be helpful.")
        assert builder._system_prompt == "Be helpful."

    def test_with_system_prompt_returns_self(self):
        """with_system_prompt should return self for chaining."""
        builder = AgentBuilder()
        result = builder.with_system_prompt("Test")
        assert result is builder

    def test_with_system_prompt_empty_string(self):
        """with_system_prompt should accept empty string."""
        builder = AgentBuilder().with_system_prompt("")
        assert builder._system_prompt == ""

    def test_with_system_prompt_multiline(self):
        """with_system_prompt should accept multiline strings."""
        prompt = """You are a helpful assistant.
        You should be polite.
        Always provide accurate information."""
        builder = AgentBuilder().with_system_prompt(prompt)
        assert builder._system_prompt == prompt

    def test_with_system_prompt_overwrites_previous(self):
        """Calling with_system_prompt again should overwrite previous value."""
        builder = (
            AgentBuilder().with_system_prompt("First").with_system_prompt("Second")
        )
        assert builder._system_prompt == "Second"


class TestWithDescription:
    """Tests for the with_description method."""

    def test_with_description_sets_description(self):
        """with_description should set the description."""
        builder = AgentBuilder().with_description("A helpful agent")
        assert builder._description == "A helpful agent"

    def test_with_description_returns_self(self):
        """with_description should return self for chaining."""
        builder = AgentBuilder()
        result = builder.with_description("Test")
        assert result is builder

    def test_with_description_empty_string(self):
        """with_description should accept empty string."""
        builder = AgentBuilder().with_description("")
        assert builder._description == ""

    def test_with_description_overwrites_previous(self):
        """Calling with_description again should overwrite previous value."""
        builder = AgentBuilder().with_description("First").with_description("Second")
        assert builder._description == "Second"


class TestAddTool:
    """Tests for the add_tool method."""

    def test_add_tool_adds_single_tool(self, mock_tool):
        """add_tool should add a tool to the list."""
        builder = AgentBuilder().add_tool(mock_tool)
        assert len(builder._tools) == 1
        assert builder._tools[0] is mock_tool

    def test_add_tool_returns_self(self, mock_tool):
        """add_tool should return self for chaining."""
        builder = AgentBuilder()
        result = builder.add_tool(mock_tool)
        assert result is builder

    def test_add_tool_multiple_times(self):
        """add_tool can be called multiple times."""
        tool1 = MagicMock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = MagicMock(spec=BaseTool)
        tool2.name = "tool2"
        tool3 = MagicMock(spec=BaseTool)
        tool3.name = "tool3"

        builder = AgentBuilder().add_tool(tool1).add_tool(tool2).add_tool(tool3)

        assert len(builder._tools) == 3
        assert builder._tools[0] is tool1
        assert builder._tools[1] is tool2
        assert builder._tools[2] is tool3

    def test_add_tool_preserves_order(self):
        """Tools should be added in order."""
        tools = [MagicMock(spec=BaseTool) for _ in range(5)]
        builder = AgentBuilder()

        for tool in tools:
            builder.add_tool(tool)

        for i, tool in enumerate(tools):
            assert builder._tools[i] is tool


class TestAddTools:
    """Tests for the add_tools method."""

    def test_add_tools_adds_multiple_tools(self):
        """add_tools should add all tools from the list."""
        tools = [MagicMock(spec=BaseTool) for _ in range(3)]
        builder = AgentBuilder().add_tools(tools)

        assert len(builder._tools) == 3
        for i, tool in enumerate(tools):
            assert builder._tools[i] is tool

    def test_add_tools_returns_self(self):
        """add_tools should return self for chaining."""
        builder = AgentBuilder()
        result = builder.add_tools([])
        assert result is builder

    def test_add_tools_empty_list(self):
        """add_tools with empty list should not change tools."""
        builder = AgentBuilder().add_tools([])
        assert len(builder._tools) == 0

    def test_add_tools_extends_existing(self, mock_tool):
        """add_tools should extend existing tools, not replace."""
        tool2 = MagicMock(spec=BaseTool)
        tool3 = MagicMock(spec=BaseTool)

        builder = AgentBuilder().add_tool(mock_tool).add_tools([tool2, tool3])

        assert len(builder._tools) == 3
        assert builder._tools[0] is mock_tool
        assert builder._tools[1] is tool2
        assert builder._tools[2] is tool3

    def test_add_tools_multiple_calls(self):
        """Multiple add_tools calls should accumulate."""
        tools1 = [MagicMock(spec=BaseTool) for _ in range(2)]
        tools2 = [MagicMock(spec=BaseTool) for _ in range(3)]

        builder = AgentBuilder().add_tools(tools1).add_tools(tools2)

        assert len(builder._tools) == 5


class TestAddAgent:
    """Tests for the add_agent method."""

    def test_add_agent_adds_single_agent(self, mock_agent):
        """add_agent should add a sub-agent to the list."""
        builder = AgentBuilder().add_agent(mock_agent)
        assert len(builder._callable_agents) == 1
        assert builder._callable_agents[0] is mock_agent

    def test_add_agent_returns_self(self, mock_agent):
        """add_agent should return self for chaining."""
        builder = AgentBuilder()
        result = builder.add_agent(mock_agent)
        assert result is builder

    def test_add_agent_multiple_times(self):
        """add_agent can be called multiple times."""
        agent1 = MagicMock(spec=Agent)
        agent2 = MagicMock(spec=Agent)

        builder = AgentBuilder().add_agent(agent1).add_agent(agent2)

        assert len(builder._callable_agents) == 2
        assert builder._callable_agents[0] is agent1
        assert builder._callable_agents[1] is agent2


class TestAddAgents:
    """Tests for the add_agents method."""

    def test_add_agents_adds_multiple_agents(self):
        """add_agents should add all agents from the list."""
        agents = [MagicMock(spec=Agent) for _ in range(3)]
        builder = AgentBuilder().add_agents(agents)

        assert len(builder._callable_agents) == 3

    def test_add_agents_returns_self(self):
        """add_agents should return self for chaining."""
        builder = AgentBuilder()
        result = builder.add_agents([])
        assert result is builder

    def test_add_agents_empty_list(self):
        """add_agents with empty list should not change agents."""
        builder = AgentBuilder().add_agents([])
        assert len(builder._callable_agents) == 0

    def test_add_agents_extends_existing(self, mock_agent):
        """add_agents should extend existing agents, not replace."""
        agent2 = MagicMock(spec=Agent)

        builder = AgentBuilder().add_agent(mock_agent).add_agents([agent2])

        assert len(builder._callable_agents) == 2


class TestBuild:
    """Tests for the build method."""

    def test_build_creates_agent(self, complete_builder):
        """build should create an Agent with configured values."""
        agent = complete_builder.build()

        assert isinstance(agent, Agent)
        assert agent.name == "TestAgent"
        assert "You are a test agent." in agent.system_prompt
        assert agent.description == "A test agent for testing"

    def test_build_without_name_raises(self, mock_model):
        """build without name should raise AgentConfigurationError."""
        builder = (
            AgentBuilder()
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
        )

        with pytest.raises(AgentConfigurationError) as exc_info:
            builder.build()
        assert "name" in str(exc_info.value).lower()

    def test_build_without_model_raises(self):
        """build without model should raise AgentConfigurationError."""
        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_system_prompt("Test")
            .with_description("Test")
        )

        with pytest.raises(AgentConfigurationError) as exc_info:
            builder.build()
        assert "model" in str(exc_info.value).lower()

    def test_build_without_system_prompt_raises(self, mock_model):
        """build without system_prompt should raise AgentConfigurationError."""
        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_description("Test")
        )

        with pytest.raises(AgentConfigurationError) as exc_info:
            builder.build()
        assert "system prompt" in str(exc_info.value).lower()

    def test_build_without_description_raises(self, mock_model):
        """build without description should raise AgentConfigurationError."""
        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_system_prompt("Test")
        )

        with pytest.raises(AgentConfigurationError) as exc_info:
            builder.build()
        assert "description" in str(exc_info.value).lower()

    def test_build_with_empty_name_succeeds(self, mock_model):
        """build with empty string name should succeed (not None)."""
        builder = (
            AgentBuilder()
            .with_name("")
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
        )

        # Empty string is not None, so this should work
        agent = builder.build()
        assert agent.name == ""

    def test_build_with_tools(self, complete_builder, mock_tool):
        """build should include tools when added."""
        complete_builder.add_tool(mock_tool)
        agent = complete_builder.build()

        assert agent.tools is not None
        assert len(agent.tools) == 1

    def test_build_without_tools_sets_none(self, complete_builder):
        """build without tools should set tools to None."""
        agent = complete_builder.build()
        assert agent.tools is None

    def test_build_with_callable_agents(self, complete_builder, mock_agent):
        """build should include callable_agents when added."""
        complete_builder.add_agent(mock_agent)
        agent = complete_builder.build()

        assert agent.callable_agents is not None
        assert len(agent.callable_agents) == 1

    def test_build_without_callable_agents_sets_none(self, complete_builder):
        """build without callable_agents should set to None."""
        agent = complete_builder.build()
        assert agent.callable_agents is None

    def test_build_can_be_called_multiple_times(self, complete_builder):
        """build can be called multiple times to create different agents."""
        agent1 = complete_builder.build()
        agent2 = complete_builder.build()

        assert agent1 is not agent2
        assert agent1.name == agent2.name


class TestMethodChaining:
    """Tests for fluent API method chaining."""

    def test_full_chain(self, mock_model, mock_tool, mock_agent):
        """All methods should chain together fluently."""
        agent = (
            AgentBuilder()
            .with_name("ChainedAgent")
            .with_model(mock_model)
            .with_system_prompt("Test prompt")
            .with_description("Test description")
            .add_tool(mock_tool)
            .add_agent(mock_agent)
            .build()
        )

        assert isinstance(agent, Agent)
        assert agent.name == "ChainedAgent"

    def test_chain_order_independent(self, mock_model):
        """Methods can be called in any order."""
        agent = (
            AgentBuilder()
            .with_description("Description first")
            .with_system_prompt("Prompt second")
            .with_name("Name third")
            .with_model(mock_model)
            .build()
        )

        assert agent.name == "Name third"
        assert agent.description == "Description first"

    def test_mixed_tool_methods(self, mock_model):
        """add_tool and add_tools can be mixed."""
        tool1 = MagicMock(spec=BaseTool)
        tool1.name = "tool1"
        tool2 = MagicMock(spec=BaseTool)
        tool2.name = "tool2"
        tool3 = MagicMock(spec=BaseTool)
        tool3.name = "tool3"

        agent = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
            .add_tool(tool1)
            .add_tools([tool2, tool3])
            .build()
        )

        assert agent.tools is not None
        assert len(agent.tools) == 3


class TestBuilderIsolation:
    """Tests for builder instance isolation."""

    def test_builders_are_independent(self, mock_model):
        """Different builder instances should not share state."""
        builder1 = AgentBuilder().with_name("Agent1")
        builder2 = AgentBuilder().with_name("Agent2")

        assert builder1._name == "Agent1"
        assert builder2._name == "Agent2"

    def test_tool_lists_are_independent(self, mock_tool):
        """Different builders should have independent tool lists."""
        builder1 = AgentBuilder().add_tool(mock_tool)
        builder2 = AgentBuilder()

        assert len(builder1._tools) == 1
        assert len(builder2._tools) == 0

    def test_agent_lists_are_independent(self, mock_agent):
        """Different builders should have independent agent lists."""
        builder1 = AgentBuilder().add_agent(mock_agent)
        builder2 = AgentBuilder()

        assert len(builder1._callable_agents) == 1
        assert len(builder2._callable_agents) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_build_empty_builder_raises(self):
        """Building an empty builder should raise AgentConfigurationError."""
        builder = AgentBuilder()
        with pytest.raises(AgentConfigurationError):
            builder.build()

    def test_special_characters_in_name(self, mock_model):
        """Name can contain special characters."""
        agent = (
            AgentBuilder()
            .with_name("Agent @#$%^&*() 日本語 🎉")
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
            .build()
        )

        assert agent.name == "Agent @#$%^&*() 日本語 🎉"

    def test_very_long_system_prompt(self, mock_model):
        """System prompt can be very long."""
        long_prompt = "A" * 10000
        agent = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_system_prompt(long_prompt)
            .with_description("Test")
            .build()
        )

        assert long_prompt in agent.system_prompt

    def test_many_tools(self, mock_model):
        """Builder should handle many tools."""
        tools = []
        for i in range(100):
            tool = MagicMock(spec=BaseTool)
            tool.name = f"tool_{i}"
            tools.append(tool)

        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
            .add_tools(tools)
        )

        agent = builder.build()
        assert agent.tools is not None
        assert len(agent.tools) == 100

    def test_many_callable_agents(self, mock_model):
        """Builder should handle many callable agents."""
        agents = [MagicMock(spec=Agent) for _ in range(50)]
        for i, agent in enumerate(agents):
            agent.name = f"Agent{i}"
            agent.description = f"Description {i}"

        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_model(mock_model)
            .with_system_prompt("Test")
            .with_description("Test")
            .add_agents(agents)
        )

        agent = builder.build()
        assert agent.callable_agents is not None
        assert len(agent.callable_agents) == 50


class TestBuilderReuse:
    """Tests for builder reuse patterns."""

    def test_builder_can_be_modified_after_build(self, complete_builder, mock_tool):
        """Builder can be modified after calling build."""
        agent1 = complete_builder.build()

        complete_builder.add_tool(mock_tool)
        agent2 = complete_builder.build()

        assert agent1.tools is None
        assert agent2.tools is not None

    def test_builder_state_persists(self, mock_model, mock_tool):
        """Builder state persists between method calls."""
        builder = AgentBuilder()
        builder.with_name("Test")
        builder.with_model(mock_model)
        builder.add_tool(mock_tool)
        builder.with_system_prompt("Prompt")
        builder.with_description("Desc")

        agent = builder.build()

        assert agent.name == "Test"
        assert agent.tools is not None
        assert len(agent.tools) == 1
