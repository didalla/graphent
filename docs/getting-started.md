# Getting Started

This guide will help you install Graphent and build your first agent.

## Installation

Install Graphent using pip:

```bash
pip install graphent
```

## Prerequisites

- Python 3.13+
- An LLM provider (e.g., OpenAI API key)

## Your First Agent

### 1. Set Up Your Environment

```bash
export OPENAI_API_KEY="your-api-key"
```

### 2. Create a Simple Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from lib import AgentBuilder, Context

# Initialize the model
model = ChatOpenAI(model="gpt-4")

# Build an agent using the fluent builder API
agent = (AgentBuilder()
    .with_name("Assistant")
    .with_model(model)
    .with_system_prompt("You are a helpful assistant.")
    .with_description("A general-purpose assistant")
    .build())

# Create a conversation context
context = Context().add_message(HumanMessage(content="Hello! What can you help me with?"))

# Invoke the agent
result = agent.invoke(context)

# Print the response
print(result.get_messages()[-1].content)
```

### 3. Add Tools to Your Agent

Agents become powerful when they can use tools:

```python
from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Build an agent with tools
agent = (AgentBuilder()
    .with_name("Tool Agent")
    .with_model(model)
    .with_system_prompt("You are a helpful assistant with access to tools.")
    .with_description("An agent that can use tools")
    .add_tool(get_current_time)
    .add_tool(calculate)
    .build())
```

## Async Usage

Graphent fully supports async/await:

```python
import asyncio

async def main():
    context = Context().add_message(HumanMessage(content="Hello!"))
    result = await agent.ainvoke(context)
    print(result.get_messages()[-1].content)

asyncio.run(main())
```

## Streaming Responses

Stream responses in real-time:

```python
context = Context().add_message(HumanMessage(content="Tell me a story"))

for chunk in agent.stream(context):
    print(chunk, end="", flush=True)
```

## Configuration

Configure Graphent via environment variables:

```bash
export GRAPHENT_LOG_LEVEL=25
export GRAPHENT_TRUNCATE_LIMIT=250
export GRAPHENT_MAX_ITERATIONS=10
export GRAPHENT_DEBUG=false
```

Or programmatically:

```python
from lib import GraphentConfig, set_config

config = GraphentConfig(
    log_level=20,
    truncate_limit=500,
    max_iterations=15,
    debug=True
)
set_config(config)
```

## Next Steps

- Learn about [Agents](core-concepts/agent.md) in depth
- Understand [Context](core-concepts/context.md) management
- Add [Hooks](core-concepts/hooks.md) to monitor agent activity
- Build [Multi-Agent Systems](guides/multi-agent.md)
