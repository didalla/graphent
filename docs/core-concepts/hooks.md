# Hooks

The hooks system allows you to monitor and react to events during agent execution. Hooks are callbacks that fire at specific points in the agent lifecycle.

## Overview

Graphent provides hooks for:

- **Tool calls** - Before and after tool execution
- **Model calls** - Before and after LLM invocation
- **Responses** - When the agent generates a final response
- **Delegation** - When delegating to a sub-agent

## Available Hooks

| Decorator | Event Type | Description |
|-----------|------------|-------------|
| `@on_tool_call` | `ToolCallEvent` | Before a tool is called |
| `@after_tool_call` | `ToolResultEvent` | After a tool returns |
| `@before_model_call` | `ModelCallEvent` | Before invoking the LLM |
| `@after_model_call` | `ModelResultEvent` | After the LLM returns |
| `@on_response` | `ResponseEvent` | When generating final response |
| `@on_delegation` | `DelegationEvent` | When delegating to sub-agent |

## Using Hooks

### Decorator-Based Hooks

Create a class with decorated methods:

```python
from lib import (
    AgentBuilder, 
    on_tool_call, 
    after_tool_call,
    on_response,
    ToolCallEvent, 
    ToolResultEvent,
    ResponseEvent
)

class MyHooks:
    @on_tool_call
    def log_tool_call(self, event: ToolCallEvent):
        print(f"Calling tool: {event.tool_name}")
        print(f"Arguments: {event.tool_args}")
    
    @after_tool_call
    def log_tool_result(self, event: ToolResultEvent):
        print(f"Tool {event.tool_name} returned: {event.result}")
    
    @on_response
    def log_response(self, event: ResponseEvent):
        print(f"Agent {event.agent_name} responded")

# Register hooks with an agent
agent = (AgentBuilder()
    .with_name("Agent")
    .with_model(model)
    .with_system_prompt("You are helpful.")
    .with_description("An agent with hooks")
    .add_hooks_from_object(MyHooks())
    .build())
```

### Manual Hook Registration

```python
from lib import HookRegistry, HookType, ToolCallEvent

def my_hook(event: ToolCallEvent):
    print(f"Tool called: {event.tool_name}")

registry = HookRegistry()
registry.register(HookType.ON_TOOL_CALL, my_hook)

agent = (AgentBuilder()
    .with_name("Agent")
    .with_model(model)
    .with_system_prompt("You are helpful.")
    .with_description("An agent")
    .with_hooks(registry)
    .build())
```

## Event Data Classes

### ToolCallEvent

Fired before a tool is executed.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `tool_name` | `str` | Name of the tool being called |
| `tool_args` | `dict` | Arguments passed to the tool |

### ToolResultEvent

Fired after a tool returns.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `tool_name` | `str` | Name of the tool |
| `tool_args` | `dict` | Arguments that were passed |
| `result` | `Any` | The tool's return value |

### ModelCallEvent

Fired before the LLM is invoked.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `messages` | `list` | Messages being sent to the model |

### ModelResultEvent

Fired after the LLM returns.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `response` | `AIMessage` | The model's response |

### ResponseEvent

Fired when the agent produces a final response.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent |
| `response` | `str` | The response content |

### DelegationEvent

Fired when delegating to a sub-agent.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the delegating agent |
| `target_agent` | `str` | Name of the target sub-agent |
| `task` | `str` | The task being delegated |

## Async Hooks

Hooks can be async functions:

```python
class AsyncHooks:
    @on_tool_call
    async def async_log(self, event: ToolCallEvent):
        await some_async_operation()
        print(f"Tool: {event.tool_name}")
```

## Error Handling

Hook errors are logged but don't break the main execution flow. This ensures that monitoring code can't crash your agent.

For complete API documentation, see the [API Reference](../reference/index.md#hookregistry).
