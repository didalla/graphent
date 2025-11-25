# Graphent Library Audit

> **Date**: November 25, 2025  
> **Last Updated**: November 25, 2025  
> **Scope**: `lib/` directory - Agent.py, AgentBuilder.py, Context.py, logging_utils.py

---

## Executive Summary

The library provides a solid foundation for a multi-agent framework, but has several areas that need attention ranging from critical bugs to code quality improvements. This document categorizes issues by severity and provides actionable recommendations.

---

## ✅ Resolved Critical Issues

### 1. ~~**Typo in Class/Function Name: "HandOf" → "HandOff"**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: Renamed `AgentHandOf` → `AgentHandOff`, `hand_of_to_subagent` → `hand_off_to_subagent` throughout.

---

### 2. ~~**ToolMessage Missing Required `tool_call_id`**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: Added `tool_call_id` parameter to all `ToolMessage` instantiations in `invoke()`. The `_hand_off_to_subagent` method now returns strings directly (proper tool return type) instead of `ToolMessage` objects.

---

### 3. ~~**Context Loss in `invoke()` Method**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: The `invoke()` method now preserves and returns the original context with the AI response added, instead of creating a new Context.

---

### 4. ~~**AI Response Not Added to Context Before Tool Execution**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: The AI response is now added to context immediately after receiving it, before processing tool calls.

---

## ✅ Resolved Important Issues

### 6. ~~**Unused Import**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: Removed unused `from langchain.tools import tool` import.

---

### 8. ~~**Mutable Default Arguments Typing**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: Added `from typing import Optional` and updated type hints to use `Optional[list[BaseTool]]` and `Optional[list['Agent']]`.

---

## 🟠 Remaining Important Issues

### ~~5. **Unbounded Recursion in `invoke()`**~~ ✅ FIXED
**File**: `Agent.py`

**Resolution**: Added `max_iterations` parameter (default: 10) and `_current_iteration` counter to prevent infinite loops. When the limit is reached, a friendly message is returned to the user.

---

### 7. **Sub-agent Context Isolation**
**File**: `Agent.py`

When delegating to a sub-agent, a fresh context is created with just the task. The sub-agent has no access to the parent conversation context.

```python
return usable_agents[0].invoke(Context().add_message(AIMessage(content=task))).get_messages(last_n=1)[0]
```

**Impact**: Sub-agents lack conversation context that might be relevant.

**Consideration**: This might be intentional design, but should be documented. Consider adding an option to pass parent context.

---

## 🟡 Code Quality Issues

### ~~9. **Missing Docstrings**~~ ✅ FIXED
**Files**: All files in `lib/`

**Resolution**: Added comprehensive docstrings to all classes, methods, and modules in `Agent.py`, `AgentBuilder.py`, `Context.py`, and `logging_utils.py`.

---

### 10. **Inconsistent Error Handling**
**File**: `Agent.py`

Errors are returned as `ToolMessage` with error content rather than raising exceptions. This makes debugging harder.

**Recommendation**: Consider a custom exception hierarchy:
```python
class AgentError(Exception):
    pass

class AgentNotFoundError(AgentError):
    pass

class ToolNotFoundError(AgentError):
    pass
```

---

### 11. **Magic Strings in System Prompt**
**File**: `Agent.py` (lines 48-57)

The system prompt modification contains hardcoded strings with formatting issues (inconsistent indentation).

**Recommendation**: Move to a template or constants:
```python
AGENT_DELEGATION_PROMPT = """
Planning: If you don't have information for a tool call, check if you can use a tool or a subagent.

## Available Agents
{agent_list}
"""
```

---

### 12. **Context Class Could Be More Feature-Rich**
**File**: `Context.py`

The `Context` class is minimal. Consider adding:

```python
class Context:
    def clear(self) -> "Context":
        """Clear all messages."""
        self._messages.clear()
        return self
    
    def copy(self) -> "Context":
        """Create a copy of this context."""
        new_ctx = Context()
        new_ctx._messages = self._messages.copy()
        return new_ctx
    
    def get_last_message(self) -> Optional[BaseMessage]:
        """Get the most recent message."""
        return self._messages[-1] if self._messages else None
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __iter__(self):
        return iter(self._messages)
```

---

### 13. **Logging Typo**
**File**: `logging_utils.py` (line 66)

```python
logging.log(LOG_LEVEL, f"START {agent_name} METHODE {method_name}...")
```

"METHODE" should be "METHOD" (unless intentionally German 😄).

---

### 14. **AgentBuilder Lacks Reset Method**
**File**: `AgentBuilder.py`

Once a builder is used, it retains state. Consider adding:

```python
def reset(self) -> 'AgentBuilder':
    """Reset the builder to its initial state."""
    self._name = None
    self._model = None
    self._system_prompt = None
    self._description = None
    self._tools = []
    self._callable_agents = []
    return self
```

---

### 15. **No Type Hints for Return Values in Some Methods**
**File**: `Agent.py`

```python
def _hand_of_to_subagent(self, agent_name: str, task: str) -> BaseMessage:
```

This returns `BaseMessage` but actually should return the content string for the tool, or the return type should be clarified.

---

## 🔵 Suggestions for Future Enhancements

### 16. **Async Support**
The framework is synchronous. Consider adding async variants:
```python
async def ainvoke(self, context: Context) -> Context:
    ...
```

### 17. **Streaming Support**
For better UX, consider adding streaming responses.

### 18. **Serialization/Deserialization**
Add ability to save/load agent configurations and conversation contexts.

### 19. **Event Hooks/Callbacks**
Add hooks for monitoring:
```python
agent.on_tool_call(callback)
agent.on_response(callback)
agent.on_delegation(callback)
```

### 20. **Token Counting/Cost Tracking**
Track API usage for cost monitoring.

---

## Summary Table

| # | Issue | Severity | File | Status |
|---|-------|----------|------|--------|
| 1 | Typo: HandOf → HandOff | 🔴 Critical | Agent.py | ✅ Fixed |
| 2 | Missing tool_call_id | 🔴 Critical | Agent.py | ✅ Fixed |
| 3 | Context loss in invoke() | 🔴 Critical | Agent.py | ✅ Fixed |
| 4 | AI response not in context | 🔴 Critical | Agent.py | ✅ Fixed |
| 5 | Unbounded recursion | 🟠 Important | Agent.py | ✅ Fixed |
| 6 | Unused import | 🟠 Important | Agent.py | ✅ Fixed |
| 7 | Sub-agent context isolation | 🟠 Important | Agent.py | ⏳ Open |
| 8 | Mutable default args typing | 🟠 Important | Agent.py | ✅ Fixed |
| 9 | Missing docstrings | 🟡 Quality | All | ✅ Fixed |
| 10 | Inconsistent error handling | 🟡 Quality | Agent.py | ⏳ Open |
| 11 | Magic strings | 🟡 Quality | Agent.py | ⏳ Open |
| 12 | Context class features | 🟡 Quality | Context.py | ⏳ Open |
| 13 | Logging typo | 🟡 Quality | logging_utils.py | ⏳ Open |
| 14 | Builder reset method | 🟡 Quality | AgentBuilder.py | ⏳ Open |
| 15 | Return type clarity | 🟡 Quality | Agent.py | ✅ Fixed |

---

## Recommended Priority

1. ~~**Immediate**: Fix critical issues #1-4 (context handling, tool_call_id)~~ ✅ DONE
2. **Soon**: Address remaining important issues #5, #7 (recursion limit, context isolation)
3. **Ongoing**: Improve code quality issues #9-14
4. **Future**: Consider enhancements #16-20
