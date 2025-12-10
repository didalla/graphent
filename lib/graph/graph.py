"""Graph execution engine for workflow orchestration.

This module provides the Graph class that manages node execution,
edge traversal, and context flow through the graph.
"""

from typing import Optional, Generator, AsyncGenerator

from lib.Context import Context
from lib.graph.base import BaseNode
from lib.graph.edges import Edge, ConditionalEdge
from lib.graph.nodes import ClassifierNode
from lib.graph.hooks import (
    GraphHookRegistry,
    GraphHookType,
    NodeEnterEvent,
    NodeExitEvent,
    EdgeTraverseEvent,
    ClassificationEvent,
)


class Graph:
    """Executable graph of nodes and edges.

    The Graph class manages a directed graph of nodes connected by edges.
    It handles execution flow, including conditional branching based on
    classifier nodes.

    Attributes:
        nodes: Dictionary of nodes by name.
        edges: List of simple edges.
        conditional_edges: List of conditional edges.
        entry_point: Name of the entry node.
        finish_point: Name of the finish node (optional).
        _hooks: Hook registry for graph events.

    Example:
        >>> graph = Graph()
        >>> graph.add_node(my_action_node)
        >>> graph.add_node(my_agent_node)
        >>> graph.add_edge("action", "agent")
        >>> graph.set_entry_point("action")
        >>> result = graph.invoke(context)
    """

    def __init__(self, hooks: Optional[GraphHookRegistry] = None):
        """Initialize an empty graph.

        Args:
            hooks: Optional hook registry for graph events.
        """
        self.nodes: dict[str, BaseNode] = {}
        self.edges: list[Edge] = []
        self.conditional_edges: list[ConditionalEdge] = []
        self.entry_point: Optional[str] = None
        self.finish_point: Optional[str] = None
        self._hooks = hooks or GraphHookRegistry()

    def add_node(self, node: BaseNode) -> "Graph":
        """Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists in the graph")
        node.set_hooks(self._hooks)
        self.nodes[node.name] = node
        return self

    def add_edge(
        self, source: str, target: str, label: Optional[str] = None
    ) -> "Graph":
        """Add a simple edge between two nodes.

        Args:
            source: Name of the source node.
            target: Name of the target node.
            label: Optional label for the edge.

        Returns:
            Self for method chaining.
        """
        self.edges.append(Edge(source=source, target=target, label=label))
        return self

    def add_conditional_edges(self, source: str, mapping: dict[str, str]) -> "Graph":
        """Add conditional edges from a classifier node.

        Args:
            source: Name of the classifier node.
            mapping: Dictionary mapping classification values to target node names.

        Returns:
            Self for method chaining.
        """
        for condition, target in mapping.items():
            self.conditional_edges.append(
                ConditionalEdge(source=source, condition=condition, target=target)
            )
        return self

    def set_entry_point(self, node_name: str) -> "Graph":
        """Set the entry point node.

        Args:
            node_name: Name of the node to start execution from.

        Returns:
            Self for method chaining.
        """
        self.entry_point = node_name
        return self

    def set_finish_point(self, node_name: str) -> "Graph":
        """Set the finish point node.

        Args:
            node_name: Name of the node that ends execution.

        Returns:
            Self for method chaining.
        """
        self.finish_point = node_name
        return self

    def _validate(self) -> None:
        """Validate the graph configuration.

        Raises:
            ValueError: If the graph is invalid.
        """
        if self.entry_point is None:
            raise ValueError("Graph must have an entry point")

        if self.entry_point not in self.nodes:
            raise ValueError(f"Entry point '{self.entry_point}' not found in nodes")

        if self.finish_point and self.finish_point not in self.nodes:
            raise ValueError(f"Finish point '{self.finish_point}' not found in nodes")

        # Validate edges reference existing nodes
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in self.nodes:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")

        for edge in self.conditional_edges:
            if edge.source not in self.nodes:
                raise ValueError(
                    f"Conditional edge source '{edge.source}' not found in nodes"
                )
            if edge.target not in self.nodes:
                raise ValueError(
                    f"Conditional edge target '{edge.target}' not found in nodes"
                )

    def _get_next_node(self, current_node_name: str) -> Optional[str]:
        """Determine the next node to execute.

        Args:
            current_node_name: Name of the current node.

        Returns:
            Name of the next node, or None if no more nodes.
        """
        current_node = self.nodes[current_node_name]

        # Check if current node is a classifier with conditional edges
        if isinstance(current_node, ClassifierNode):
            classification = current_node.get_classification()
            if classification:
                # Find conditional edge matching the classification
                for edge in self.conditional_edges:
                    if (
                        edge.source == current_node_name
                        and edge.condition == classification
                    ):
                        # Trigger classification hook
                        self._hooks.trigger(
                            GraphHookType.ON_CLASSIFICATION,
                            ClassificationEvent(
                                node_name=current_node_name,
                                classification=classification,
                                target_node=edge.target,
                                confidence=current_node.last_confidence,
                            ),
                        )
                        return edge.target

        # Check regular edges
        for edge in self.edges:
            if edge.source == current_node_name:
                return edge.target

        return None

    async def _aget_next_node(self, current_node_name: str) -> Optional[str]:
        """Asynchronously determine the next node to execute.

        Args:
            current_node_name: Name of the current node.

        Returns:
            Name of the next node, or None if no more nodes.
        """
        current_node = self.nodes[current_node_name]

        # Check if current node is a classifier with conditional edges
        if isinstance(current_node, ClassifierNode):
            classification = current_node.get_classification()
            if classification:
                # Find conditional edge matching the classification
                for edge in self.conditional_edges:
                    if (
                        edge.source == current_node_name
                        and edge.condition == classification
                    ):
                        # Trigger classification hook
                        await self._hooks.atrigger(
                            GraphHookType.ON_CLASSIFICATION,
                            ClassificationEvent(
                                node_name=current_node_name,
                                classification=classification,
                                target_node=edge.target,
                                confidence=current_node.last_confidence,
                            ),
                        )
                        return edge.target

        # Check regular edges
        for edge in self.edges:
            if edge.source == current_node_name:
                return edge.target

        return None

    def invoke(self, context: Context) -> Context:
        """Execute the graph with the given context.

        Args:
            context: The initial conversation context.

        Returns:
            The final context after graph execution.
        """
        self._validate()

        current_node_name = self.entry_point

        while current_node_name is not None:
            node = self.nodes[current_node_name]

            # Trigger node enter hook
            self._hooks.trigger(
                GraphHookType.ON_NODE_ENTER,
                NodeEnterEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Execute the node
            context = node.invoke(context)

            # Trigger node exit hook
            self._hooks.trigger(
                GraphHookType.ON_NODE_EXIT,
                NodeExitEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Check if we've reached the finish point
            if current_node_name == self.finish_point:
                break

            # Get next node
            next_node = self._get_next_node(current_node_name)

            # Trigger edge traverse hook
            if next_node:
                self._hooks.trigger(
                    GraphHookType.ON_EDGE_TRAVERSE,
                    EdgeTraverseEvent(
                        source_node=current_node_name,
                        target_node=next_node,
                        edge_type="conditional"
                        if isinstance(node, ClassifierNode)
                        else "simple",
                    ),
                )

            current_node_name = next_node

        return context

    async def ainvoke(self, context: Context) -> Context:
        """Asynchronously execute the graph.

        Args:
            context: The initial conversation context.

        Returns:
            The final context after graph execution.
        """
        self._validate()

        current_node_name = self.entry_point

        while current_node_name is not None:
            node = self.nodes[current_node_name]

            # Trigger node enter hook
            await self._hooks.atrigger(
                GraphHookType.ON_NODE_ENTER,
                NodeEnterEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Execute the node
            context = await node.ainvoke(context)

            # Trigger node exit hook
            await self._hooks.atrigger(
                GraphHookType.ON_NODE_EXIT,
                NodeExitEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Check if we've reached the finish point
            if current_node_name == self.finish_point:
                break

            # Get next node
            next_node = await self._aget_next_node(current_node_name)

            # Trigger edge traverse hook
            if next_node:
                await self._hooks.atrigger(
                    GraphHookType.ON_EDGE_TRAVERSE,
                    EdgeTraverseEvent(
                        source_node=current_node_name,
                        target_node=next_node,
                        edge_type="conditional"
                        if isinstance(node, ClassifierNode)
                        else "simple",
                    ),
                )

            current_node_name = next_node

        return context

    def stream(self, context: Context) -> Generator[str, None, Context]:
        """Stream execution through the graph.

        Yields response chunks from each node as they execute.

        Args:
            context: The initial conversation context.

        Yields:
            String chunks from node responses.

        Returns:
            The final context after graph execution.
        """
        self._validate()

        current_node_name = self.entry_point

        while current_node_name is not None:
            node = self.nodes[current_node_name]

            # Trigger node enter hook
            self._hooks.trigger(
                GraphHookType.ON_NODE_ENTER,
                NodeEnterEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Stream the node execution
            for chunk in node.stream(context):
                yield chunk

            # Trigger node exit hook
            self._hooks.trigger(
                GraphHookType.ON_NODE_EXIT,
                NodeExitEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Check if we've reached the finish point
            if current_node_name == self.finish_point:
                break

            # Get next node
            next_node = self._get_next_node(current_node_name)

            # Trigger edge traverse hook
            if next_node:
                self._hooks.trigger(
                    GraphHookType.ON_EDGE_TRAVERSE,
                    EdgeTraverseEvent(
                        source_node=current_node_name,
                        target_node=next_node,
                        edge_type="conditional"
                        if isinstance(node, ClassifierNode)
                        else "simple",
                    ),
                )

            current_node_name = next_node

        return context

    async def astream(self, context: Context) -> AsyncGenerator[str, None]:
        """Asynchronously stream execution through the graph.

        Args:
            context: The initial conversation context.

        Yields:
            String chunks from node responses.
        """
        self._validate()

        current_node_name = self.entry_point

        while current_node_name is not None:
            node = self.nodes[current_node_name]

            # Trigger node enter hook
            await self._hooks.atrigger(
                GraphHookType.ON_NODE_ENTER,
                NodeEnterEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Stream the node execution
            async for chunk in node.astream(context):
                yield chunk

            # Trigger node exit hook
            await self._hooks.atrigger(
                GraphHookType.ON_NODE_EXIT,
                NodeExitEvent(
                    node_name=current_node_name,
                    node_type=node.__class__.__name__,
                    context=context,
                ),
            )

            # Check if we've reached the finish point
            if current_node_name == self.finish_point:
                break

            # Get next node
            next_node = await self._aget_next_node(current_node_name)

            # Trigger edge traverse hook
            if next_node:
                await self._hooks.atrigger(
                    GraphHookType.ON_EDGE_TRAVERSE,
                    EdgeTraverseEvent(
                        source_node=current_node_name,
                        target_node=next_node,
                        edge_type="conditional"
                        if isinstance(node, ClassifierNode)
                        else "simple",
                    ),
                )

            current_node_name = next_node

    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        return (
            f"Graph(nodes={list(self.nodes.keys())}, "
            f"entry={self.entry_point}, finish={self.finish_point})"
        )
