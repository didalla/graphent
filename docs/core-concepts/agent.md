# Agent

The `Agent` class is the core of Graphent. It wraps a language model and handles conversation processing, tool execution, and delegation to sub-agents.

## Overview

An Agent:

- Processes conversations using a language model
- Can execute tools (functions the LLM can call)
- Can delegate to sub-agents for specialized tasks
- Supports both synchronous and asynchronous invocation
- Can stream responses in real-time

## Creating Agents

Use `AgentBuilder` to create agents:

```python
from langchain_openai import ChatOpenAI
from lib import AgentBuilder

model = ChatOpenAI(model="gpt-4")

agent = (AgentBuilder()
    .with_name("Assistant")
    .with_model(model)
    .with_system_prompt("You are a helpful assistant.")
    .with_description("A general-purpose assistant")
    .build())
```

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `name` | Unique identifier for the agent |
| `model` | The LangChain chat model to use |
| `system_prompt` | Instructions for the agent's behavior |
| `description` | Short description (used when delegating) |

### Optional Parameters

| Parameter | Description |
|-----------|-------------|
| `tools` | List of tools the agent can use |
| `sub_agents` | List of agents this agent can delegate to |
| `hooks` | HookRegistry for monitoring events |

## Invoking Agents

### Synchronous Invocation

```python
from langchain_core.messages import HumanMessage
from lib import Context

context = Context().add_message(HumanMessage(content="Hello!"))
result = agent.invoke(context)

# Get the last message (the agent's response)
response = result.get_messages()[-1].content
```

### Asynchronous Invocation

```python
import asyncio

async def main():
    context = Context().add_message(HumanMessage(content="Hello!"))
    result = await agent.ainvoke(context)
    print(result.get_messages()[-1].content)

asyncio.run(main())
```

## Streaming

Stream responses as they're generated:

### Synchronous Streaming

```python
context = Context().add_message(HumanMessage(content="Tell me a story"))

for chunk in agent.stream(context):
    print(chunk, end="", flush=True)
```

### Asynchronous Streaming

```python
async def stream_response():
    context = Context().add_message(HumanMessage(content="Tell me a story"))
    
    async for chunk in agent.astream(context):
        print(chunk, end="", flush=True)
```

## Max Iterations

Agents have a maximum iteration limit to prevent infinite loops when using tools:

```python
# Default is 10 iterations
result = agent.invoke(context, max_iterations=20)
```

If the limit is exceeded, a `MaxIterationsExceededError` is raised.

For complete API documentation, see the [API Reference](../reference/index.md#agent).
