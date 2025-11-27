# Roadmap

## Project summary

Graphent: an extensible, production-grade agent framework for building, orchestrating, and deploying autonomous agents and agent-assisted applications. Graphent provides a lightweight runtime, planner, memory and tool integration primitives, SDKs (Python-first), observability, and safety controls so teams can iterate from prototypes to scalable production deployments.

## Future Features

1. Streaming Support
For better UX, consider adding streaming responses.

2. Serialization/Deserialization
Add ability to save/load agent configurations and conversation contexts.

3. Event Hooks/Callbacks
Add hooks for monitoring:

@on_tool_cool

@on_response

@before_model_call

@after_model_call

@on_delegation

4. Token Counting/Cost Tracking
Track API usage for cost monitoring.