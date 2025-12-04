# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Event Hooks/Callbacks**: New hook system for monitoring and reacting to agent execution events.
  - New `lib/hooks.py` module with decorators and event data classes
  - `@on_tool_call` - Hook triggered before a tool is called (receives `ToolCallEvent`)
  - `@after_tool_call` - Hook triggered after a tool returns (receives `ToolResultEvent`)
  - `@on_response` - Hook triggered when the model generates a final response (receives `ResponseEvent`)
  - `@before_model_call` - Hook triggered before invoking the model (receives `ModelCallEvent`)
  - `@after_model_call` - Hook triggered after the model returns (receives `ModelResultEvent`)
  - `@on_delegation` - Hook triggered when delegating to a sub-agent (receives `DelegationEvent`)
  - `HookRegistry` class for managing callback registration and triggering
  - Support for both synchronous and asynchronous hook callbacks
  - Exception-safe: hook errors are logged but don't break the main execution flow
  - Event data classes include relevant context (agent name, tool info, parameters, results)

- **AgentBuilder Hook Methods**: New methods for registering hooks via the builder pattern:
  - `add_hook(hook_type, callback)` - Register individual hook callbacks
  - `add_hooks_from_object(obj)` - Auto-register all decorated methods from an object
  - `with_hooks(registry)` - Use a pre-configured `HookRegistry`

- **Agent Hooks Parameter**: `Agent.__init__()` now accepts an optional `hooks` parameter for passing a `HookRegistry`

- **Comprehensive Hook Tests**: Added `tests/test_hooks.py` with tests for:
  - Event data classes (`ToolCallEvent`, `ToolResultEvent`, `ResponseEvent`, `ModelCallEvent`, `ModelResultEvent`, `DelegationEvent`)
  - Hook decorators (all 6 decorators properly mark functions)
  - `HookRegistry` functionality (registration, triggering, exception handling, clearing)
  - Async hook support (sync callbacks, async callbacks, mixed callbacks)

- **Streaming Support**: New `stream()` and `astream()` methods on the `Agent` class for real-time chunk-wise response streaming.
  - `stream(context, max_iterations=10)` - Synchronous generator that yields response chunks as they are generated
  - `astream(context, max_iterations=10)` - Asynchronous generator for async contexts
  - Both methods support tool calls with automatic recursive continuation after tool execution
  - Proper accumulation and parsing of streamed tool call chunks
  - Context is updated with the complete response message after streaming completes
  - Respects `max_iterations` to prevent infinite loops
  - Sets `_last_response` attribute for wrapper/logging compatibility

- **Comprehensive Test Suite**: Added extensive tests for streaming functionality in `tests/test_agent.py`:
  - `TestStreamingBasic` - Basic streaming behavior (chunk yielding, context updates, empty content handling, max iterations)
  - `TestStreamingWithTools` - Streaming with tool calls (tool execution, missing tool handling)
  - `TestAsyncStreaming` - Async streaming tests (async chunk yielding, context updates, async tool calls)
  - `TestStreamingEdgeCases` - Edge cases (chunks without content attribute, malformed JSON in tool args, None tools guard)

### Changed

- Updated `ROADMAP.md` to reflect implemented streaming feature

## [0.1.0] - 2024-XX-XX

### Added

- Initial release of Graphent agent framework
- `Agent` class with `invoke()` and `ainvoke()` methods for conversation processing
- Tool binding and execution support
- Sub-agent delegation via `hand_off_to_subagent` tool
- `AgentBuilder` for fluent agent construction
- `Context` class for managing conversation state
- Logging utilities with `@log_agent_activity` decorator
- Basic CLI interface
