"""Comprehensive test suite for the Graph module.

Tests cover node types, edge connections, graph execution,
streaming, and hooks.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from lib.Context import Context
from lib.Agent import Agent
from lib.graph import (
    Graph,
    GraphBuilder,
    ActionNode,
    AgentNode,
    ClassifierNode,
    Edge,
    ConditionalEdge,
    GraphHookRegistry,
    GraphHookType,
    NodeEnterEvent,
    NodeExitEvent,
    EdgeTraverseEvent,
    ClassificationEvent,
    on_node_enter,
    on_node_exit,
)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock BaseChatModel."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content="Test response")
    model.ainvoke = AsyncMock(return_value=AIMessage(content="Async test response"))
    return model


@pytest.fixture
def mock_streaming_model():
    """Create a mock model that streams responses."""
    model = MagicMock(spec=BaseChatModel)

    def stream_generator(messages):
        chunks = [
            MagicMock(content="Hello"),
            MagicMock(content=" "),
            MagicMock(content="World"),
        ]
        for chunk in chunks:
            yield chunk

    model.stream.side_effect = stream_generator
    model.invoke.return_value = AIMessage(content="Hello World")
    return model


@pytest.fixture
def mock_classifier_model():
    """Create a mock model for classification."""
    model = MagicMock(spec=BaseChatModel)
    # Return text that contains the classification
    model.invoke.return_value = AIMessage(content="technical")
    model.ainvoke = AsyncMock(return_value=AIMessage(content="technical"))
    return model


@pytest.fixture
def basic_context():
    """Create a basic context with a human message."""
    context = Context()
    context.add_message(HumanMessage(content="Hello, world!"))
    return context


@pytest.fixture
def mock_agent(mock_model):
    """Create a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.invoke.return_value = Context().add_message(
        AIMessage(content="Agent response")
    )
    agent.ainvoke = AsyncMock(
        return_value=Context().add_message(AIMessage(content="Async agent response"))
    )
    return agent


# ============================================================================
# Edge Tests
# ============================================================================


class TestEdge:
    """Tests for the Edge class."""

    def test_edge_creation(self):
        """Edge should store source and target."""
        edge = Edge(source="node1", target="node2")
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.label is None

    def test_edge_with_label(self):
        """Edge should accept optional label."""
        edge = Edge(source="node1", target="node2", label="next")
        assert edge.label == "next"

    def test_edge_repr(self):
        """Edge should have a readable string representation."""
        edge = Edge(source="a", target="b")
        assert "a" in repr(edge) and "b" in repr(edge)


class TestConditionalEdge:
    """Tests for the ConditionalEdge class."""

    def test_conditional_edge_creation(self):
        """ConditionalEdge should store source, condition, and target."""
        edge = ConditionalEdge(source="router", condition="yes", target="handler")
        assert edge.source == "router"
        assert edge.condition == "yes"
        assert edge.target == "handler"

    def test_conditional_edge_repr(self):
        """ConditionalEdge should have a readable string representation."""
        edge = ConditionalEdge(source="router", condition="yes", target="handler")
        assert "router" in repr(edge)
        assert "yes" in repr(edge)
        assert "handler" in repr(edge)


# ============================================================================
# ActionNode Tests
# ============================================================================


class TestActionNode:
    """Tests for the ActionNode class."""

    def test_action_node_creation(self, mock_model):
        """ActionNode should store name, model, and system_prompt."""
        node = ActionNode(
            name="formatter", model=mock_model, system_prompt="Format text"
        )
        assert node.name == "formatter"
        assert node.system_prompt == "Format text"

    def test_action_node_invoke(self, mock_model, basic_context):
        """ActionNode.invoke should call model and update context."""
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        result = node.invoke(basic_context)

        mock_model.invoke.assert_called_once()
        messages = result.get_messages()
        assert len(messages) == 2  # Original + AI response
        assert messages[-1].content == "Test response"

    @pytest.mark.asyncio
    async def test_action_node_ainvoke(self, mock_model, basic_context):
        """ActionNode.ainvoke should call model asynchronously."""
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        result = await node.ainvoke(basic_context)

        mock_model.ainvoke.assert_called_once()
        messages = result.get_messages()
        assert len(messages) == 2
        assert messages[-1].content == "Async test response"

    def test_action_node_stream(self, mock_streaming_model, basic_context):
        """ActionNode.stream should yield chunks."""
        node = ActionNode(name="test", model=mock_streaming_model, system_prompt="Test")

        chunks = list(node.stream(basic_context))
        assert chunks == ["Hello", " ", "World"]

    def test_action_node_repr(self, mock_model):
        """ActionNode should have a readable repr."""
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        assert "ActionNode" in repr(node)
        assert "test" in repr(node)


# ============================================================================
# AgentNode Tests
# ============================================================================


class TestAgentNode:
    """Tests for the AgentNode class."""

    def test_agent_node_creation(self, mock_agent):
        """AgentNode should store name and agent."""
        node = AgentNode(name="researcher", agent=mock_agent)
        assert node.name == "researcher"
        assert node.agent == mock_agent

    def test_agent_node_invoke(self, mock_agent, basic_context):
        """AgentNode.invoke should delegate to agent."""
        node = AgentNode(name="test", agent=mock_agent)
        node.invoke(basic_context)

        mock_agent.invoke.assert_called_once_with(basic_context)

    @pytest.mark.asyncio
    async def test_agent_node_ainvoke(self, mock_agent, basic_context):
        """AgentNode.ainvoke should delegate to agent asynchronously."""
        node = AgentNode(name="test", agent=mock_agent)
        await node.ainvoke(basic_context)

        mock_agent.ainvoke.assert_called_once_with(basic_context)

    def test_agent_node_repr(self, mock_agent):
        """AgentNode should have a readable repr."""
        node = AgentNode(name="test", agent=mock_agent)
        assert "AgentNode" in repr(node)
        assert "test" in repr(node)


# ============================================================================
# ClassifierNode Tests
# ============================================================================


class TestClassifierNode:
    """Tests for the ClassifierNode class."""

    def test_classifier_node_creation(self, mock_classifier_model):
        """ClassifierNode should store name, model, and classes."""
        node = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )
        assert node.name == "router"
        assert node.classes == ["technical", "creative"]

    def test_classifier_node_invoke(self, mock_classifier_model, basic_context):
        """ClassifierNode.invoke should classify and store result."""
        node = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )

        result = node.invoke(basic_context)

        assert node.last_classification == "technical"
        assert node.last_confidence == 1.0
        # Classification message should be added
        messages = result.get_messages()
        assert "[Classification: technical]" in messages[-1].content

    @pytest.mark.asyncio
    async def test_classifier_node_ainvoke(self, mock_classifier_model, basic_context):
        """ClassifierNode.ainvoke should classify asynchronously."""
        node = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )

        result = await node.ainvoke(basic_context)

        assert node.last_classification == "technical"

    def test_classifier_node_get_classification(
        self, mock_classifier_model, basic_context
    ):
        """ClassifierNode.get_classification should return last result."""
        node = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )

        assert node.get_classification() is None
        node.invoke(basic_context)
        assert node.get_classification() == "technical"

    def test_classifier_node_invalid_classification_fallback(
        self, mock_classifier_model, basic_context
    ):
        """ClassifierNode should fallback to first class on invalid classification."""
        # Return text that doesn't match any class
        mock_classifier_model.invoke.return_value = AIMessage(content="invalid_class")

        node = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )

        node.invoke(basic_context)
        assert node.last_classification == "technical"  # First class as fallback


# ============================================================================
# Graph Tests
# ============================================================================


class TestGraph:
    """Tests for the Graph class."""

    def test_graph_creation(self):
        """Graph should initialize empty."""
        graph = Graph()
        assert graph.nodes == {}
        assert graph.edges == []
        assert graph.entry_point is None

    def test_graph_add_node(self, mock_model):
        """Graph.add_node should add nodes by name."""
        graph = Graph()
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")

        result = graph.add_node(node)

        assert "test" in graph.nodes
        assert result is graph  # Returns self for chaining

    def test_graph_add_duplicate_node_raises(self, mock_model):
        """Graph should raise on duplicate node names."""
        graph = Graph()
        node1 = ActionNode(name="test", model=mock_model, system_prompt="Test 1")
        node2 = ActionNode(name="test", model=mock_model, system_prompt="Test 2")

        graph.add_node(node1)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node2)

    def test_graph_add_edge(self, mock_model):
        """Graph.add_edge should connect nodes."""
        graph = Graph()
        node1 = ActionNode(name="a", model=mock_model, system_prompt="A")
        node2 = ActionNode(name="b", model=mock_model, system_prompt="B")

        graph.add_node(node1).add_node(node2)
        result = graph.add_edge("a", "b")

        assert len(graph.edges) == 1
        assert graph.edges[0].source == "a"
        assert graph.edges[0].target == "b"
        assert result is graph

    def test_graph_set_entry_point(self, mock_model):
        """Graph.set_entry_point should set the start node."""
        graph = Graph()
        node = ActionNode(name="start", model=mock_model, system_prompt="Start")

        graph.add_node(node)
        result = graph.set_entry_point("start")

        assert graph.entry_point == "start"
        assert result is graph

    def test_graph_validate_missing_entry(self, mock_model):
        """Graph should raise if no entry point set."""
        graph = Graph()
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        graph.add_node(node)

        with pytest.raises(ValueError, match="entry point"):
            graph.invoke(Context())

    def test_graph_validate_invalid_entry(self, mock_model):
        """Graph should raise if entry point doesn't exist."""
        graph = Graph()
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        graph.add_node(node)
        graph.set_entry_point("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            graph.invoke(Context())


class TestGraphExecution:
    """Tests for graph execution."""

    def test_graph_invoke_single_node(self, mock_model, basic_context):
        """Graph should execute a single node."""
        graph = Graph()
        node = ActionNode(name="only", model=mock_model, system_prompt="Only")

        graph.add_node(node)
        graph.set_entry_point("only")
        graph.set_finish_point("only")

        result = graph.invoke(basic_context)

        mock_model.invoke.assert_called_once()
        assert len(result.get_messages()) == 2

    def test_graph_invoke_linear_chain(self, mock_model, basic_context):
        """Graph should execute nodes in sequence."""
        graph = Graph()
        node1 = ActionNode(name="first", model=mock_model, system_prompt="First")
        node2 = ActionNode(name="second", model=mock_model, system_prompt="Second")

        graph.add_node(node1).add_node(node2)
        graph.add_edge("first", "second")
        graph.set_entry_point("first")
        graph.set_finish_point("second")

        result = graph.invoke(basic_context)

        assert mock_model.invoke.call_count == 2
        assert len(result.get_messages()) == 3  # Original + 2 responses

    def test_graph_invoke_conditional_branching(
        self, mock_model, mock_classifier_model, basic_context
    ):
        """Graph should follow conditional edges based on classification."""
        graph = Graph()

        classifier = ClassifierNode(
            name="router",
            model=mock_classifier_model,
            classes=["technical", "creative"],
        )
        tech_node = ActionNode(name="tech", model=mock_model, system_prompt="Tech")
        creative_node = ActionNode(
            name="creative", model=mock_model, system_prompt="Creative"
        )

        graph.add_node(classifier)
        graph.add_node(tech_node)
        graph.add_node(creative_node)
        graph.add_conditional_edges(
            "router", {"technical": "tech", "creative": "creative"}
        )
        graph.set_entry_point("router")
        graph.set_finish_point("tech")

        graph.invoke(basic_context)

        # Should have called tech node, not creative
        assert mock_model.invoke.call_count == 1  # Only tech node

    @pytest.mark.asyncio
    async def test_graph_ainvoke(self, mock_model, basic_context):
        """Graph.ainvoke should execute asynchronously."""
        graph = Graph()
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")

        graph.add_node(node)
        graph.set_entry_point("test")
        graph.set_finish_point("test")

        result = await graph.ainvoke(basic_context)

        mock_model.ainvoke.assert_called_once()

    def test_graph_stream(self, mock_streaming_model, basic_context):
        """Graph.stream should yield chunks from nodes."""
        graph = Graph()
        node = ActionNode(name="test", model=mock_streaming_model, system_prompt="Test")

        graph.add_node(node)
        graph.set_entry_point("test")
        graph.set_finish_point("test")

        chunks = list(graph.stream(basic_context))
        assert chunks == ["Hello", " ", "World"]


# ============================================================================
# GraphBuilder Tests
# ============================================================================


class TestGraphBuilder:
    """Tests for the GraphBuilder class."""

    def test_builder_creation(self):
        """GraphBuilder should initialize empty."""
        builder = GraphBuilder()
        assert builder._nodes == {}
        assert builder._entry is None

    def test_builder_add_action_node(self, mock_model):
        """GraphBuilder.add_action_node should create and add ActionNode."""
        builder = GraphBuilder()
        result = builder.add_action_node("test", mock_model, "Test prompt")

        assert "test" in builder._nodes
        assert isinstance(builder._nodes["test"], ActionNode)
        assert result is builder

    def test_builder_add_agent_node(self, mock_agent):
        """GraphBuilder.add_agent_node should create and add AgentNode."""
        builder = GraphBuilder()
        result = builder.add_agent_node("test", mock_agent)

        assert "test" in builder._nodes
        assert isinstance(builder._nodes["test"], AgentNode)
        assert result is builder

    def test_builder_add_classifier_node(self, mock_classifier_model):
        """GraphBuilder.add_classifier_node should create and add ClassifierNode."""
        builder = GraphBuilder()
        result = builder.add_classifier_node(
            "router", mock_classifier_model, ["a", "b"]
        )

        assert "router" in builder._nodes
        assert isinstance(builder._nodes["router"], ClassifierNode)
        assert result is builder

    def test_builder_connect(self, mock_model):
        """GraphBuilder.connect should add edges."""
        builder = GraphBuilder()
        builder.add_action_node("a", mock_model, "A")
        builder.add_action_node("b", mock_model, "B")

        result = builder.connect("a", "b")

        assert ("a", "b", None) in builder._edges
        assert result is builder

    def test_builder_branch(self, mock_classifier_model, mock_model):
        """GraphBuilder.branch should add conditional edges."""
        builder = GraphBuilder()
        builder.add_classifier_node("router", mock_classifier_model, ["yes", "no"])
        builder.add_action_node("yes_handler", mock_model, "Yes")
        builder.add_action_node("no_handler", mock_model, "No")

        result = builder.branch("router", {"yes": "yes_handler", "no": "no_handler"})

        assert "router" in builder._conditional_edges
        assert builder._conditional_edges["router"]["yes"] == "yes_handler"
        assert result is builder

    def test_builder_set_entry_and_finish(self, mock_model):
        """GraphBuilder should set entry and finish points."""
        builder = GraphBuilder()
        builder.add_action_node("start", mock_model, "Start")
        builder.add_action_node("end", mock_model, "End")

        builder.set_entry("start").set_finish("end")

        assert builder._entry == "start"
        assert builder._finish == "end"

    def test_builder_build(self, mock_model):
        """GraphBuilder.build should create a configured Graph."""
        builder = (
            GraphBuilder()
            .add_action_node("start", mock_model, "Start")
            .add_action_node("end", mock_model, "End")
            .connect("start", "end")
            .set_entry("start")
            .set_finish("end")
        )

        graph = builder.build()

        assert isinstance(graph, Graph)
        assert "start" in graph.nodes
        assert "end" in graph.nodes
        assert graph.entry_point == "start"
        assert graph.finish_point == "end"

    def test_builder_build_missing_entry_raises(self, mock_model):
        """GraphBuilder.build should raise if entry not set."""
        builder = GraphBuilder()
        builder.add_action_node("test", mock_model, "Test")

        with pytest.raises(ValueError, match="entry point"):
            builder.build()


# ============================================================================
# Graph Hook Tests
# ============================================================================


class TestGraphHooks:
    """Tests for graph hooks."""

    def test_hook_registry_creation(self):
        """GraphHookRegistry should initialize empty."""
        registry = GraphHookRegistry()
        for hook_type in GraphHookType:
            assert not registry.has_hooks(hook_type)

    def test_hook_registration(self):
        """GraphHookRegistry should register callbacks."""
        registry = GraphHookRegistry()
        callback = MagicMock()

        registry.register(GraphHookType.ON_NODE_ENTER, callback)

        assert registry.has_hooks(GraphHookType.ON_NODE_ENTER)

    def test_hook_trigger(self):
        """GraphHookRegistry should trigger callbacks with events."""
        registry = GraphHookRegistry()
        callback = MagicMock()
        registry.register(GraphHookType.ON_NODE_ENTER, callback)

        event = NodeEnterEvent(
            node_name="test", node_type="ActionNode", context=Context()
        )
        registry.trigger(GraphHookType.ON_NODE_ENTER, event)

        callback.assert_called_once_with(event)

    def test_hooks_triggered_during_execution(self, mock_model, basic_context):
        """Graph execution should trigger hooks."""
        enter_callback = MagicMock()
        exit_callback = MagicMock()

        hooks = GraphHookRegistry()
        hooks.register(GraphHookType.ON_NODE_ENTER, enter_callback)
        hooks.register(GraphHookType.ON_NODE_EXIT, exit_callback)

        graph = Graph(hooks=hooks)
        node = ActionNode(name="test", model=mock_model, system_prompt="Test")
        graph.add_node(node)
        graph.set_entry_point("test")
        graph.set_finish_point("test")

        graph.invoke(basic_context)

        enter_callback.assert_called_once()
        exit_callback.assert_called_once()

        # Verify event data
        enter_event = enter_callback.call_args[0][0]
        assert enter_event.node_name == "test"
        assert enter_event.node_type == "ActionNode"

    def test_hook_decorator(self):
        """Hook decorators should mark functions."""

        @on_node_enter
        def my_hook(event):
            pass

        assert hasattr(my_hook, "_is_graph_hook")
        assert my_hook._is_graph_hook is True
        assert my_hook._graph_hook_type == GraphHookType.ON_NODE_ENTER

    def test_register_hooks_from_object(self):
        """GraphHookRegistry should register hooks from decorated methods."""

        class MyHooks:
            @on_node_enter
            def handle_enter(self, event):
                pass

            @on_node_exit
            def handle_exit(self, event):
                pass

        registry = GraphHookRegistry()
        registry.register_hooks_from_object(MyHooks())

        assert registry.has_hooks(GraphHookType.ON_NODE_ENTER)
        assert registry.has_hooks(GraphHookType.ON_NODE_EXIT)

    def test_graph_builder_with_hooks(self, mock_model):
        """GraphBuilder should pass hooks to graph."""
        hooks = GraphHookRegistry()
        callback = MagicMock()
        hooks.register(GraphHookType.ON_NODE_ENTER, callback)

        graph = (
            GraphBuilder()
            .add_action_node("test", mock_model, "Test")
            .set_entry("test")
            .set_finish("test")
            .with_hooks(hooks)
            .build()
        )

        graph.invoke(Context().add_message(HumanMessage(content="Test")))

        callback.assert_called_once()
