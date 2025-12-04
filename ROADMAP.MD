# Roadmap

## Project summary

Graphent: an extensible, production-grade agent framework for building, orchestrating, and deploying autonomous agents and agent-assisted applications. Graphent provides a lightweight runtime, planner, memory and tool integration primitives, SDKs (Python-first), observability, and safety controls so teams can iterate from prototypes to scalable production deployments.

## Implemented Features

1. ✅ Streaming Support
Added `stream()` and `astream()` methods to the Agent class for chunk-wise response streaming. Features include:
- Real-time streaming of response chunks as they are generated
- Full support for tool calls during streaming
- Recursive streaming continuation after tool execution
- Proper context management with complete response preservation
- Both synchronous (`stream()`) and asynchronous (`astream()`) variants

2. ✅ Event Hooks/Callbacks
Added hooks for monitoring agent execution with decorator-based registration:
- `@on_tool_call` - Before a tool is called (receives ToolCallEvent with tool_name, tool_args, tool_call_id, agent_name)
- `@after_tool_call` - After a tool returns (receives ToolResultEvent with tool info and result)
- `@on_response` - When the model generates a final response (receives ResponseEvent)
- `@before_model_call` - Before invoking the model (receives ModelCallEvent)
- `@after_model_call` - After the model returns (receives ModelResultEvent)
- `@on_delegation` - When delegating to a sub-agent (receives DelegationEvent with from_agent, to_agent, task)

Features:
- HookRegistry for managing callbacks
- Support for both sync and async hooks
- AgentBuilder integration with `add_hook()`, `add_hooks_from_object()`, and `with_hooks()`
- Exception-safe: hook errors are logged but don't break the main flow

## Future Features

1. Serialization/Deserialization
Add ability to save/load agent configurations and conversation contexts.

2. Token Counting/Cost Tracking
Track API usage for cost monitoring.