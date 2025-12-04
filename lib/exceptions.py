"""Custom exceptions for the Graphent agent framework.

This module defines a hierarchy of exceptions for error handling
throughout the Graphent library.
"""


class GraphentError(Exception):
    """Base exception for all Graphent errors.

    All custom exceptions in the Graphent framework inherit from this class,
    allowing users to catch all Graphent-related errors with a single handler.

    Example:
        >>> try:
        ...     agent.invoke(context)
        ... except GraphentError as e:
        ...     print(f"Graphent error: {e}")
    """
    pass


class AgentConfigurationError(GraphentError):
    """Raised when an agent is misconfigured.

    This exception is raised when required agent parameters are missing
    or invalid during agent construction.

    Example:
        >>> agent = AgentBuilder().build()  # Missing required fields
        AgentConfigurationError: Agent requires a name
    """
    pass


class ToolExecutionError(GraphentError):
    """Raised when a tool fails during execution.

    This exception wraps errors that occur during tool invocation,
    providing context about which tool failed and why.

    Attributes:
        tool_name: The name of the tool that failed.
        original_error: The original exception that was raised.
    """

    def __init__(self, tool_name: str, original_error: Exception | None = None, message: str | None = None):
        """Initialize a ToolExecutionError.

        Args:
            tool_name: The name of the tool that failed.
            original_error: The original exception that caused the failure.
            message: Optional custom error message.
        """
        self.tool_name = tool_name
        self.original_error = original_error
        msg = message or f"Tool '{tool_name}' failed during execution"
        if original_error:
            msg = f"{msg}: {original_error}"
        super().__init__(msg)


class DelegationError(GraphentError):
    """Raised when agent delegation fails.

    This exception is raised when an agent cannot delegate a task
    to a sub-agent, such as when the target agent is not available.

    Attributes:
        from_agent: The name of the agent attempting to delegate.
        to_agent: The name of the target agent.
    """

    def __init__(self, from_agent: str, to_agent: str, message: str | None = None):
        """Initialize a DelegationError.

        Args:
            from_agent: The name of the agent attempting to delegate.
            to_agent: The name of the target agent.
            message: Optional custom error message.
        """
        self.from_agent = from_agent
        self.to_agent = to_agent
        msg = message or f"Agent '{from_agent}' failed to delegate to '{to_agent}'"
        super().__init__(msg)


class MaxIterationsExceededError(GraphentError):
    """Raised when an agent exceeds its maximum iteration limit.

    This exception is raised when an agent's invoke loop exceeds
    the configured maximum number of iterations, preventing infinite loops.

    Attributes:
        agent_name: The name of the agent that exceeded the limit.
        max_iterations: The maximum number of iterations allowed.
    """

    def __init__(self, agent_name: str, max_iterations: int):
        """Initialize a MaxIterationsExceededError.

        Args:
            agent_name: The name of the agent that exceeded the limit.
            max_iterations: The maximum number of iterations allowed.
        """
        self.agent_name = agent_name
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent '{agent_name}' exceeded maximum iterations ({max_iterations}). "
            "Consider breaking down the request into smaller parts."
        )


class HookExecutionError(GraphentError):
    """Raised when a hook callback fails during execution.

    This exception wraps errors that occur during hook invocation,
    providing context about which hook failed.

    Attributes:
        hook_type: The type of hook that failed.
        original_error: The original exception that was raised.
    """

    def __init__(self, hook_type: str, original_error: Exception | None = None):
        """Initialize a HookExecutionError.

        Args:
            hook_type: The type of hook that failed.
            original_error: The original exception that caused the failure.
        """
        self.hook_type = hook_type
        self.original_error = original_error
        msg = f"Hook '{hook_type}' failed during execution"
        if original_error:
            msg = f"{msg}: {original_error}"
        super().__init__(msg)

