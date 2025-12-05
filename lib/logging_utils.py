"""Logging utilities for the Graphent agent framework.

This module provides logging configuration and decorators for
tracking agent activity during conversations.
"""

import asyncio
import inspect
import logging
import os
import threading
from functools import wraps
from typing import Any, Callable, Dict

from lib.Context import Context

# Module-level logger
_logger = logging.getLogger(__name__)

#: Custom log level for agent activity (between INFO and WARNING)
LOG_LEVEL = 25

#: Maximum length for truncated log output
TRUNCATE_LIMIT = int(os.environ.get("GRAPHENT_TRUNCATE_LIMIT", os.environ.get("TRUNCATE_LIMIT", 250)))


class AgentLoggerConfig:
    """Configuration class for agent logging.
    
    This class manages the singleton setup of logging for agents,
    ensuring logging is configured only once per application run.
    
    Attributes:
        _setup_done: Flag to prevent duplicate setup.
        _last_returned_response_id: Cache to avoid duplicate log entries.
        _lock: Thread lock for thread-safe operations.
    """
    _setup_done = False
    _last_returned_response_id = None
    _lock = threading.Lock()

    @staticmethod
    def setup(level: int = LOG_LEVEL, log_file: str | None = None) -> None:
        """Set up logging for agent activity.
        
        Configures the root logger with a timestamped format. Can log
        to either console or a file. Thread-safe.

        Args:
            level: The logging level to use (default: LOG_LEVEL).
            log_file: Optional path to a log file. If None, logs to console.
        """
        with AgentLoggerConfig._lock:
            if AgentLoggerConfig._setup_done:
                return

            formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%H:%M:%S'
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(level)

            if log_file:
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)

            else:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            AgentLoggerConfig._setup_done = True


def log_agent_activity(func: Callable) -> Callable:
    """Decorator that logs agent method invocations.
    
    Wraps agent methods to log the start of execution with input details
    and the returned result. Automatically extracts context information
    and truncates long outputs. Supports both sync and async methods.
    
    Args:
        func: The agent method to wrap (sync or async).
        
    Returns:
        The wrapped function with logging.
    
    Example:
        >>> class MyAgent:
        ...     @log_agent_activity
        ...     def invoke(self, context: Context) -> Context:
        ...         ...
        ...     @log_agent_activity
        ...     async def ainvoke(self, context: Context) -> Context:
        ...         ...
    """
    def _log_start(self, args):
        """Helper to log the start of method execution."""
        agent_name = getattr(self, "name", "Unknown Agent")
        method_name = func.__name__
        inputs = format_args(args)

        last_input = "<no input>"
        last_input_type = "<none>"
        try:
            ctx = inputs["context"]
            if isinstance(ctx, Context):
                msgs = ctx.get_messages()
            elif isinstance(ctx, list):
                msgs = ctx
            else:
                msgs = []

            if msgs:
                last = msgs[-1]
                last_input = getattr(last, "content", str(last))
                last_input_type = type(last).__name__
        except Exception:
            last_input = "<error reading context>"
            last_input_type = "<error>"

        _logger.log(LOG_LEVEL, f"START {agent_name} METHOD {method_name} WITH {last_input_type}:\"{truncate(last_input)}\"")
        return agent_name, method_name

    def _log_end(self, agent_name, method_name):
        """Helper to log the end of method execution."""
        response = getattr(self, "_last_response", None)
        if response:
            resp_metadata = _get(response, "response_metadata", None)
            resp_content = _get(response, "content", "<error>")
            resp_content_type = type(resp_content).__name__

            resp_id = _get(resp_metadata, "id", "<error reading id>") if resp_metadata else "<error reading id>"

            token_usage = _get(resp_metadata, "token_usage", None) if resp_metadata else None
            input_tokens = _get(token_usage, "prompt_tokens", "<error>") if token_usage else "<error>"
            output_tokens = _get(token_usage, "completion_tokens", "<error>") if token_usage else "<error>"
            total_tokens = _get(token_usage, "total_tokens", "<error>") if token_usage else "<error>"

            with AgentLoggerConfig._lock:
                if resp_id != AgentLoggerConfig._last_returned_response_id:
                    _logger.log(LOG_LEVEL, f"END {agent_name} METHOD {method_name} WITH {resp_content_type}: {truncate(resp_content)}")
                    if token_usage:
                        _logger.log(LOG_LEVEL, f"   Cost: {input_tokens} input tokens, {output_tokens} output tokens, {total_tokens} total tokens")
                    else:
                        _logger.log(LOG_LEVEL, f"   Cost: Token usage not available.")
                    AgentLoggerConfig._last_returned_response_id = resp_id
        else:
            _logger.log(LOG_LEVEL, f"ERROR: Something went wrong! No response object found for {agent_name} method {method_name}.")

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            agent_name, method_name = _log_start(self, args)
            result = await func(self, *args, **kwargs)
            _log_end(self, agent_name, method_name)
            return result
        return async_wrapper

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        agent_name, method_name = _log_start(self, args)
        result = func(self, *args, **kwargs)
        _log_end(self, agent_name, method_name)
        return result

    return wrapper


def format_args(args: tuple) -> Dict[str, Any]:
    """Extract and categorize function arguments for logging.
    
    Separates Context objects from other arguments for structured logging.
    
    Args:
        args: The positional arguments passed to a function.
        
    Returns:
        A dictionary with 'context' (Context or None) and 'other' (list of strings).
    """
    arg_strings: Dict[str, Any] = {
        "context": None,
        "other": [],
    }
    for arg in args:
        if isinstance(arg, Context):
            arg_strings["context"] = arg
        else:
            s = str(arg)
            arg_strings["other"].append(s)

    return arg_strings


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def truncate(value: Any) -> str:
    """Truncate a value's string representation for logging.
    
    Args:
        value: Any value to convert and potentially truncate.
        
    Returns:
        A string representation, truncated to TRUNCATE_LIMIT characters
        if necessary, with '[truncated]' suffix.
    """
    if value is None:
        return "<none>"

    string = str(value).replace('\n', '\\n')
    if len(string) > TRUNCATE_LIMIT:
        return string[:TRUNCATE_LIMIT] + '...[truncated]'
    else:
        return string