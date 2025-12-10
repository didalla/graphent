# Graphent

A **multi-agent framework** for building conversational AI systems with LangChain.

## Features

- 🤖 **Agent Builder Pattern** - Fluent API for constructing agents with tools, hooks, and sub-agents
- 🔗 **Multi-Agent Delegation** - Agents can seamlessly delegate tasks to specialized sub-agents
- 🛠️ **Tool Integration** - Easy integration with LangChain tools and custom functions
- 🎣 **Event Hooks** - Comprehensive hook system for monitoring all agent activity
- 🔄 **Graph Workflows** - Build complex AI workflows with nodes, edges, and branching logic
- ⚡ **Async & Streaming** - Full async/await support with real-time response streaming

## Quick Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from lib import AgentBuilder, Context

# Create a model
model = ChatOpenAI(model="gpt-4")

# Build an agent
agent = (AgentBuilder()
    .with_name("Assistant")
    .with_model(model)
    .with_system_prompt("You are a helpful assistant.")
    .with_description("A general-purpose assistant")
    .build())

# Invoke the agent
context = Context().add_message(HumanMessage(content="Hello!"))
result = agent.invoke(context)

print(result.get_messages()[-1].content)
```

## Installation

```bash
pip install graphent
```

## Next Steps

- [Getting Started](getting-started.md) - Installation and your first agent
- [Core Concepts](core-concepts/agent.md) - Understand how agents work
- [Guides](guides/multi-agent.md) - Build multi-agent systems and graph workflows
- [API Reference](reference/index.md) - Complete API documentation
