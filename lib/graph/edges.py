"""Edge classes for connecting graph nodes.

This module provides edge types for defining connections between nodes
in the graph workflow.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Edge:
    """Simple edge connecting two nodes.

    Represents a direct, unconditional connection from one node to another.

    Attributes:
        source: Name of the source node.
        target: Name of the target node.
        label: Optional label for the edge.

    Example:
        >>> edge = Edge(source="step1", target="step2")
    """

    source: str
    target: str
    label: Optional[str] = None

    def __repr__(self) -> str:
        """Return a string representation of the edge."""
        if self.label:
            return f"Edge({self.source} --[{self.label}]--> {self.target})"
        return f"Edge({self.source} --> {self.target})"


@dataclass
class ConditionalEdge:
    """Edge activated by classifier output.

    Represents a conditional connection from a classifier node to a target,
    activated when the classification matches the condition.

    Attributes:
        source: Name of the classifier node.
        condition: Classification value that activates this edge.
        target: Name of the target node.

    Example:
        >>> edge = ConditionalEdge(
        ...     source="router",
        ...     condition="technical",
        ...     target="tech_handler"
        ... )
    """

    source: str
    condition: str
    target: str

    def __repr__(self) -> str:
        """Return a string representation of the conditional edge."""
        return f"ConditionalEdge({self.source} --[{self.condition}]--> {self.target})"
