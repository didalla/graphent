"""GraphBuilder for fluent graph construction.

This module provides a builder pattern implementation for creating
Graph instances with a clean, fluent API.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel

from lib.Agent import Agent
from lib.graph.base import BaseNode
from lib.graph.graph import Graph
from lib.graph.nodes import ActionNode, AgentNode, ClassifierNode
from lib.graph.hooks import GraphHookRegistry


class GraphBuilder:
    """A fluent builder for constructing Graph instances.

    The GraphBuilder provides a clean, chainable API for configuring
    and creating Graph objects with nodes and edges.

    Example:
        >>> graph = (GraphBuilder()
        ...     .add_action_node("step1", model, "Process input")
        ...     .add_agent_node("step2", my_agent)
        ...     .connect("step1", "step2")
        ...     .set_entry("step1")
        ...     .set_finish("step2")
        ...     .build())

    Attributes:
        _nodes: Dictionary of nodes by name.
        _edges: List of (source, target, label) tuples.
        _conditional_edges: Dictionary mapping source to condition->target mappings.
        _entry: Name of the entry node.
        _finish: Name of the finish node.
        _hooks: Hook registry for the graph.
    """

    def __init__(self):
        """Initialize a new GraphBuilder."""
        self._nodes: dict[str, BaseNode] = {}
        self._edges: list[tuple[str, str, Optional[str]]] = []
        self._conditional_edges: dict[str, dict[str, str]] = {}
        self._entry: Optional[str] = None
        self._finish: Optional[str] = None
        self._hooks: GraphHookRegistry = GraphHookRegistry()

    def add_node(self, node: BaseNode) -> "GraphBuilder":
        """Add a pre-built node to the graph.

        Args:
            node: The node to add.

        Returns:
            Self for method chaining.
        """
        self._nodes[node.name] = node
        return self

    def add_action_node(
        self, name: str, model: BaseChatModel, system_prompt: str
    ) -> "GraphBuilder":
        """Add a simple action node.

        Creates an ActionNode that performs a single model call.

        Args:
            name: Unique identifier for the node.
            model: The language model to use.
            system_prompt: The system prompt for this action.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_action_node("format", model, "Format as markdown")
        """
        node = ActionNode(name=name, model=model, system_prompt=system_prompt)
        self._nodes[name] = node
        return self

    def add_agent_node(self, name: str, agent: Agent) -> "GraphBuilder":
        """Add a reactive agent node.

        Creates an AgentNode that wraps a full Agent with tool support.

        Args:
            name: Unique identifier for the node.
            agent: The Agent instance to wrap.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_agent_node("researcher", my_research_agent)
        """
        node = AgentNode(name=name, agent=agent)
        self._nodes[name] = node
        return self

    def add_classifier_node(
        self,
        name: str,
        model: BaseChatModel,
        classes: list[str],
        system_prompt: Optional[str] = None,
    ) -> "GraphBuilder":
        """Add a classifier node for conditional branching.

        Creates a ClassifierNode that uses LLM to classify input
        and route to different paths.

        Args:
            name: Unique identifier for the node.
            model: The language model to use for classification.
            classes: List of valid classification categories.
            system_prompt: Optional custom system prompt.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_classifier_node(
            ...     "router",
            ...     model,
            ...     classes=["technical", "creative"]
            ... )
        """
        node = ClassifierNode(
            name=name, model=model, classes=classes, system_prompt=system_prompt
        )
        self._nodes[name] = node
        return self

    def connect(
        self, source: str, target: str, label: Optional[str] = None
    ) -> "GraphBuilder":
        """Connect two nodes with an edge.

        Args:
            source: Name of the source node.
            target: Name of the target node.
            label: Optional label for the edge.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.connect("step1", "step2")
        """
        self._edges.append((source, target, label))
        return self

    def branch(self, source: str, mapping: dict[str, str]) -> "GraphBuilder":
        """Add conditional branches from a classifier node.

        Maps classification values to target nodes.

        Args:
            source: Name of the classifier node.
            mapping: Dictionary mapping classification values to target node names.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.branch("router", {
            ...     "technical": "tech_handler",
            ...     "creative": "creative_handler"
            ... })
        """
        self._conditional_edges[source] = mapping
        return self

    def set_entry(self, node_name: str) -> "GraphBuilder":
        """Set the entry point node.

        Args:
            node_name: Name of the node to start execution from.

        Returns:
            Self for method chaining.
        """
        self._entry = node_name
        return self

    def set_finish(self, node_name: str) -> "GraphBuilder":
        """Set the finish point node.

        Args:
            node_name: Name of the node that ends execution.

        Returns:
            Self for method chaining.
        """
        self._finish = node_name
        return self

    def with_hooks(self, hooks: GraphHookRegistry) -> "GraphBuilder":
        """Set a custom hook registry for the graph.

        Args:
            hooks: A pre-configured GraphHookRegistry instance.

        Returns:
            Self for method chaining.
        """
        self._hooks = hooks
        return self

    def add_hook(self, hook_type, callback) -> "GraphBuilder":
        """Add a hook callback for a specific event type.

        Args:
            hook_type: The GraphHookType to register for.
            callback: The callback function to invoke.

        Returns:
            Self for method chaining.
        """
        self._hooks.register(hook_type, callback)
        return self

    def add_hooks_from_object(self, obj: object) -> "GraphBuilder":
        """Add all hook handlers found on an object.

        Scans the object for methods decorated with graph hook decorators
        and registers them automatically.

        Args:
            obj: An object with decorated hook handler methods.

        Returns:
            Self for method chaining.
        """
        self._hooks.register_hooks_from_object(obj)
        return self

    def build(self) -> Graph:
        """Build and return the configured Graph.

        Returns:
            A new Graph instance with the configured nodes and edges.

        Raises:
            ValueError: If entry point is not set or if referenced nodes don't exist.
        """
        if self._entry is None:
            raise ValueError("Graph must have an entry point. Call set_entry() first.")

        graph = Graph(hooks=self._hooks)

        # Add all nodes
        for node in self._nodes.values():
            graph.add_node(node)

        # Add all edges
        for source, target, label in self._edges:
            graph.add_edge(source, target, label)

        # Add conditional edges
        for source, mapping in self._conditional_edges.items():
            graph.add_conditional_edges(source, mapping)

        # Set entry and finish points
        graph.set_entry_point(self._entry)
        if self._finish:
            graph.set_finish_point(self._finish)

        return graph
