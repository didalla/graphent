# Graph Workflows

Graphent includes a graph-based workflow system for building complex AI pipelines with nodes, edges, and branching logic.

## Overview

Graph workflows allow you to:

- Define nodes that perform specific actions
- Connect nodes with edges to create pipelines
- Use classifiers to branch execution paths
- Monitor execution with hooks

## Node Types

| Node Type | Description |
|-----------|-------------|
| `ActionNode` | Executes a simple action (function or LLM call) |
| `AgentNode` | Runs a full agent with tools and sub-agents |
| `ClassifierNode` | Classifies input and branches to different edges |

## Creating a Simple Graph

```python
from langchain_openai import ChatOpenAI
from lib import GraphBuilder, ActionNode, Context

model = ChatOpenAI(model="gpt-4")

# Define an action node
def summarize_action(context: Context, model) -> str:
    """Summarize the input."""
    messages = context.get_messages()
    response = model.invoke(messages)
    return response.content

# Build a graph
graph = (GraphBuilder()
    .add_node(ActionNode(
        name="summarize",
        action=summarize_action,
        model=model
    ))
    .add_edge("__start__", "summarize")
    .add_edge("summarize", "__end__")
    .build())

# Run the graph
from langchain_core.messages import HumanMessage

context = Context().add_message(HumanMessage(content="Summarize this text..."))
result = graph.run(context)
```

## Using Agent Nodes

Agent nodes run full agents within the graph:

```python
from lib import GraphBuilder, AgentNode, AgentBuilder

# Create an agent
research_agent = (AgentBuilder()
    .with_name("Researcher")
    .with_model(model)
    .with_system_prompt("You research topics thoroughly.")
    .with_description("Research agent")
    .add_tool(web_search)
    .build())

# Use it in a graph
graph = (GraphBuilder()
    .add_node(AgentNode(
        name="research",
        agent=research_agent
    ))
    .add_edge("__start__", "research")
    .add_edge("research", "__end__")
    .build())
```

## Branching with Classifiers

Classifier nodes route execution based on input classification:

```python
from lib import GraphBuilder, ClassifierNode, ActionNode, ConditionalEdge

# Create a classifier node
classifier = ClassifierNode(
    name="intent_classifier",
    model=model,
    categories=["question", "command", "greeting"],
    system_prompt="Classify the user's intent."
)

# Create action nodes for each branch
question_handler = ActionNode(name="handle_question", ...)
command_handler = ActionNode(name="handle_command", ...)
greeting_handler = ActionNode(name="handle_greeting", ...)

# Build graph with conditional edges
graph = (GraphBuilder()
    .add_node(classifier)
    .add_node(question_handler)
    .add_node(command_handler)
    .add_node(greeting_handler)
    .add_edge("__start__", "intent_classifier")
    .add_conditional_edge(
        "intent_classifier",
        {
            "question": "handle_question",
            "command": "handle_command",
            "greeting": "handle_greeting"
        }
    )
    .add_edge("handle_question", "__end__")
    .add_edge("handle_command", "__end__")
    .add_edge("handle_greeting", "__end__")
    .build())
```

## Graph Hooks

Monitor graph execution with specialized hooks:

```python
from lib import (
    on_node_enter,
    on_node_exit,
    on_edge_traverse,
    on_classification,
    NodeEnterEvent,
    NodeExitEvent,
    EdgeTraverseEvent,
    ClassificationEvent
)

class GraphMonitor:
    @on_node_enter
    def entering_node(self, event: NodeEnterEvent):
        print(f"Entering node: {event.node_name}")
    
    @on_node_exit
    def exiting_node(self, event: NodeExitEvent):
        print(f"Exiting node: {event.node_name}")
        print(f"Output: {event.output}")
    
    @on_edge_traverse
    def traversing_edge(self, event: EdgeTraverseEvent):
        print(f"Traversing: {event.from_node} -> {event.to_node}")
    
    @on_classification
    def classified(self, event: ClassificationEvent):
        print(f"Classified as: {event.category}")
```

## Async Execution

Graphs support async execution:

```python
import asyncio

async def main():
    context = Context().add_message(HumanMessage(content="Hello!"))
    result = await graph.arun(context)
    print(result)

asyncio.run(main())
```

## Complex Pipeline Example

Here's a more complex example combining multiple node types:

```python
from lib import GraphBuilder, ActionNode, AgentNode, ClassifierNode

# Define nodes
intent_classifier = ClassifierNode(
    name="classify_intent",
    model=model,
    categories=["simple_question", "complex_research", "task"],
)

quick_answer = ActionNode(
    name="quick_answer",
    action=lambda ctx, m: m.invoke(ctx.get_messages()).content,
    model=model
)

research_agent = AgentNode(
    name="deep_research",
    agent=(AgentBuilder()
        .with_name("Researcher")
        .with_model(model)
        .with_system_prompt("You conduct thorough research.")
        .with_description("Research agent")
        .add_tool(web_search)
        .add_tool(summarize)
        .build())
)

task_agent = AgentNode(
    name="task_executor",
    agent=(AgentBuilder()
        .with_name("Task Executor")
        .with_model(model)
        .with_system_prompt("You execute tasks.")
        .with_description("Task execution agent")
        .add_tool(file_write)
        .add_tool(send_email)
        .build())
)

# Build the graph
graph = (GraphBuilder()
    .add_node(intent_classifier)
    .add_node(quick_answer)
    .add_node(research_agent)
    .add_node(task_agent)
    .add_edge("__start__", "classify_intent")
    .add_conditional_edge("classify_intent", {
        "simple_question": "quick_answer",
        "complex_research": "deep_research",
        "task": "task_executor"
    })
    .add_edge("quick_answer", "__end__")
    .add_edge("deep_research", "__end__")
    .add_edge("task_executor", "__end__")
    .build())
```

For complete API documentation, see the [API Reference](../reference/index.md#graph-system).
