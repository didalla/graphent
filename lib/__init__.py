"""Graphent - A multi-agent framework for building conversational AI systems.

This module provides the public API for the Graphent framework.

Example:
    >>> from graphent import Agent, AgentBuilder, Context
    >>> agent = (AgentBuilder()
    ...     .with_name("Assistant")
    ...     .with_model(my_llm)
    ...     .with_system_prompt("You are helpful.")
    ...     .with_description("A helpful assistant")
    ...     .build())
    >>> context = Context().add_message(HumanMessage(content="Hello!"))
    >>> result = agent.invoke(context)
"""

__version__ = "0.1.0"

# Core classes
from lib.Agent import Agent
from lib.AgentBuilder import AgentBuilder
from lib.Context import Context

# Hooks system
from lib.hooks import (
    HookRegistry,
    HookType,
    HookCallback,
    # Event data classes
    ToolCallEvent,
    ToolResultEvent,
    ResponseEvent,
    ModelCallEvent,
    ModelResultEvent,
    DelegationEvent,
    # Hook decorators
    on_tool_call,
    after_tool_call,
    on_response,
    before_model_call,
    after_model_call,
    on_delegation,
)

# Configuration
from lib.config import (
    GraphentConfig,
    get_config,
    set_config,
    reset_config,
)

# Exceptions
from lib.exceptions import (
    GraphentError,
    AgentConfigurationError,
    ToolExecutionError,
    DelegationError,
    MaxIterationsExceededError,
    HookExecutionError,
)

# Logging utilities
from lib.logging_utils import (
    AgentLoggerConfig,
    log_agent_activity,
)

# Submodules (for explicit imports like `from lib import tools`)
from lib import tools
from lib import agents

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Agent",
    "AgentBuilder",
    "Context",
    # Hooks
    "HookRegistry",
    "HookType",
    "HookCallback",
    "ToolCallEvent",
    "ToolResultEvent",
    "ResponseEvent",
    "ModelCallEvent",
    "ModelResultEvent",
    "DelegationEvent",
    "on_tool_call",
    "after_tool_call",
    "on_response",
    "before_model_call",
    "after_model_call",
    "on_delegation",
    # Configuration
    "GraphentConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Exceptions
    "GraphentError",
    "AgentConfigurationError",
    "ToolExecutionError",
    "DelegationError",
    "MaxIterationsExceededError",
    "HookExecutionError",
    # Logging
    "AgentLoggerConfig",
    "log_agent_activity",
    # Submodules
    "tools",
    "agents",
]
