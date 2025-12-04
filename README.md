# Graphent

[![PyPI version](https://badge.fury.io/py/graphent.svg)](https://badge.fury.io/py/graphent)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-agent framework for building conversational AI systems with LangChain.

## Features

- 🤖 **Agent Builder Pattern** - Fluent API for constructing agents
- 🔗 **Multi-Agent Delegation** - Agents can delegate tasks to sub-agents
- 🛠️ **Tool Integration** - Easy integration with LangChain tools
- 🎣 **Event Hooks** - Comprehensive hook system for monitoring agent activity
- 📝 **Conversation Context** - Manage multi-turn conversations easily
- ⚡ **Async Support** - Full async/await support for all operations
- 🔄 **Streaming** - Stream responses in real-time

## Installation

```bash
pip install graphent
```

## Quick Start

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

# Create a conversation context and invoke
context = Context().add_message(HumanMessage(content="Hello!"))
result = agent.invoke(context)

print(result.get_messages()[-1].content)
```

## Multi-Agent Example

```python
from lib import AgentBuilder, Context
from lib.tools import get_coords, get_weather

# Create a specialized weather agent
weather_agent = (AgentBuilder()
    .with_name("Weather Agent")
    .with_model(model)
    .with_system_prompt("You answer weather questions.")
    .with_description("Gets weather information for locations")
    .add_tool(get_coords)
    .add_tool(get_weather)
    .build())

# Create a main agent that can delegate to the weather agent
main_agent = (AgentBuilder()
    .with_name("Main Agent")
    .with_model(model)
    .with_system_prompt("You are a helpful assistant.")
    .with_description("Main orchestrator agent")
    .add_agent(weather_agent)
    .build())

# The main agent will automatically delegate weather queries
context = Context().add_message(HumanMessage(content="What's the weather in Berlin?"))
result = main_agent.invoke(context)
```

## Event Hooks

Monitor agent activity with hooks:

```python
from lib import AgentBuilder, on_tool_call, after_tool_call, ToolCallEvent, ToolResultEvent

class MyHooks:
    @on_tool_call
    def log_tool_call(self, event: ToolCallEvent):
        print(f"Calling tool: {event.tool_name}")
    
    @after_tool_call
    def log_tool_result(self, event: ToolResultEvent):
        print(f"Tool {event.tool_name} returned: {event.result}")

agent = (AgentBuilder()
    .with_name("Agent")
    .with_model(model)
    .with_system_prompt("You are helpful.")
    .with_description("An agent with hooks")
    .add_hooks_from_object(MyHooks())
    .build())
```

## Async Support

```python
import asyncio
from lib import AgentBuilder, Context

async def main():
    context = Context().add_message(HumanMessage(content="Hello!"))
    result = await agent.ainvoke(context)
    print(result.get_messages()[-1].content)

asyncio.run(main())
```

## Streaming

```python
from lib import AgentBuilder, Context

context = Context().add_message(HumanMessage(content="Tell me a story"))

for chunk in agent.stream(context):
    print(chunk, end="", flush=True)
```

## Configuration

Configure via environment variables:

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

## API Reference

### Core Classes

- **`Agent`** - The main agent class that processes conversations
- **`AgentBuilder`** - Fluent builder for constructing agents
- **`Context`** - Container for conversation messages

### Hooks

- **`@on_tool_call`** - Before a tool is called
- **`@after_tool_call`** - After a tool returns
- **`@on_response`** - When the model generates a response
- **`@before_model_call`** - Before invoking the model
- **`@after_model_call`** - After the model returns
- **`@on_delegation`** - When delegating to a sub-agent

### Exceptions

- **`GraphentError`** - Base exception for all Graphent errors
- **`AgentConfigurationError`** - Invalid agent configuration
- **`ToolExecutionError`** - Tool execution failed
- **`DelegationError`** - Agent delegation failed
- **`MaxIterationsExceededError`** - Too many iterations

## Development

```bash
# Clone the repository
git clone https://github.com/didalla/graphent.git
cd graphent

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

