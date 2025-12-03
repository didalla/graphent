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

## Future Features

1. Serialization/Deserialization
Add ability to save/load agent configurations and conversation contexts.

2. Event Hooks/Callbacks
Add hooks for monitoring:

@on_tool_cool

@on_response

@before_model_call

@after_model_call

@on_delegation

3. Token Counting/Cost Tracking
Track API usage for cost monitoring.