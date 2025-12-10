"""Graph-based workflow orchestration for Graphent.

This subpackage provides graph-based architecture for composing AI pipelines
with different node types: simple actions, reactive agents, and classifiers.

Example:
    >>> from graphent.graph import GraphBuilder, Graph
    >>> graph = (GraphBuilder()
    ...     .add_action_node("step1", model, "Process input")
    ...     .add_agent_node("step2", my_agent)
    ...     .connect("step1", "step2")
    ...     .set_entry("step1")
    ...     .set_finish("step2")
    ...     .build())
    >>> result = graph.invoke(context)
"""

from lib.graph.base import BaseNode
from lib.graph.nodes import ActionNode, AgentNode, ClassifierNode
from lib.graph.edges import Edge, ConditionalEdge
from lib.graph.graph import Graph
from lib.graph.builder import GraphBuilder
from lib.graph.hooks import (
    GraphHookRegistry,
    GraphHookType,
    NodeEnterEvent,
    NodeExitEvent,
    EdgeTraverseEvent,
    ClassificationEvent,
    on_node_enter,
    on_node_exit,
    on_edge_traverse,
    on_classification,
)

__all__ = [
    # Core classes
    "Graph",
    "GraphBuilder",
    # Nodes
    "BaseNode",
    "ActionNode",
    "AgentNode",
    "ClassifierNode",
    # Edges
    "Edge",
    "ConditionalEdge",
    # Hooks
    "GraphHookRegistry",
    "GraphHookType",
    "NodeEnterEvent",
    "NodeExitEvent",
    "EdgeTraverseEvent",
    "ClassificationEvent",
    "on_node_enter",
    "on_node_exit",
    "on_edge_traverse",
    "on_classification",
]
