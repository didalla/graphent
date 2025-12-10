# API Reference

This section provides auto-generated API documentation from the Graphent source code.

## Core Classes

### Agent

::: lib.Agent.Agent
    options:
      show_root_heading: true
      heading_level: 3

### AgentBuilder

::: lib.AgentBuilder.AgentBuilder
    options:
      show_root_heading: true
      heading_level: 3

### Context

::: lib.Context.Context
    options:
      show_root_heading: true
      heading_level: 3

---

## Configuration

### GraphentConfig

::: lib.config.GraphentConfig
    options:
      show_root_heading: true
      heading_level: 3

---

## Hooks System

### HookRegistry

::: lib.hooks.HookRegistry
    options:
      show_root_heading: true
      heading_level: 3

### Event Types

::: lib.hooks.ToolCallEvent
    options:
      show_root_heading: true
      heading_level: 4

::: lib.hooks.ToolResultEvent
    options:
      show_root_heading: true
      heading_level: 4

::: lib.hooks.ResponseEvent
    options:
      show_root_heading: true
      heading_level: 4

::: lib.hooks.ModelCallEvent
    options:
      show_root_heading: true
      heading_level: 4

::: lib.hooks.ModelResultEvent
    options:
      show_root_heading: true
      heading_level: 4

::: lib.hooks.DelegationEvent
    options:
      show_root_heading: true
      heading_level: 4

---

## Graph System

### Graph

::: lib.graph.Graph
    options:
      show_root_heading: true
      heading_level: 3

### GraphBuilder

::: lib.graph.GraphBuilder
    options:
      show_root_heading: true
      heading_level: 3

### Nodes

::: lib.graph.nodes.ActionNode
    options:
      show_root_heading: true
      heading_level: 4

::: lib.graph.nodes.AgentNode
    options:
      show_root_heading: true
      heading_level: 4

::: lib.graph.nodes.ClassifierNode
    options:
      show_root_heading: true
      heading_level: 4

### Edges

::: lib.graph.edges.Edge
    options:
      show_root_heading: true
      heading_level: 4

::: lib.graph.edges.ConditionalEdge
    options:
      show_root_heading: true
      heading_level: 4

---

## Exceptions

::: lib.exceptions.GraphentError
    options:
      show_root_heading: true
      heading_level: 3

::: lib.exceptions.AgentConfigurationError
    options:
      show_root_heading: true
      heading_level: 3

::: lib.exceptions.ToolExecutionError
    options:
      show_root_heading: true
      heading_level: 3

::: lib.exceptions.DelegationError
    options:
      show_root_heading: true
      heading_level: 3

::: lib.exceptions.MaxIterationsExceededError
    options:
      show_root_heading: true
      heading_level: 3

---

## YAML Loading

::: lib.yaml_loader.load_agent_from_yaml
    options:
      show_root_heading: true
      heading_level: 3

::: lib.yaml_loader.load_graph_from_yaml
    options:
      show_root_heading: true
      heading_level: 3
