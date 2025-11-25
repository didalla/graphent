"""Logging utilities for the Graphent agent framework.

This module provides logging configuration and decorators for
tracking agent activity during conversations.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict

from lib.Context import Context

#: Custom log level for agent activity (between INFO and WARNING)
LOG_LEVEL = 25

#: Maximum length for truncated log output
TRUNCATE_LIMIT = int(os.environ.get("TRUNCATE_LIMIT", 250))


class AgentLoggerConfig:
    """Configuration class for agent logging.
    
    This class manages the singleton setup of logging for agents,
    ensuring logging is configured only once per application run.
    
    Attributes:
        _setup_done: Flag to prevent duplicate setup.
        _last_returned_result: Cache to avoid duplicate log entries.
    """
    _setup_done = False
    _last_returned_result = None

    @staticmethod
    def setup(level: int = LOG_LEVEL, log_file: str | None = None) -> None:
        """Set up logging for agent activity.
        
        Configures the root logger with a timestamped format. Can log
        to either console or a file.
        
        Args:
            level: The logging level to use (default: LOG_LEVEL).
            log_file: Optional path to a log file. If None, logs to console.
        """
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
    and truncates long outputs.
    
    Args:
        func: The agent method to wrap.
        
    Returns:
        The wrapped function with logging.
    
    Example:
        >>> class MyAgent:
        ...     @log_agent_activity
        ...     def invoke(self, context: Context) -> Context:
        ...         ...
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        agent_name = getattr(self, "name", "Unknown Agent")
        method_name = func.__name__

        inputs = format_args(args)

        # --- simplified: safely get last message and its type ---
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

        logging.log(LOG_LEVEL, f"START {agent_name} METHODE {method_name} WITH {last_input_type}:\"{truncate(last_input)}\"")

        result = func(self, *args, **kwargs)

        if result != AgentLoggerConfig._last_returned_result:
            logging.log(LOG_LEVEL, f"Function {func.__name__} returned: {truncate(result)}")
            AgentLoggerConfig._last_returned_result = result

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

    string = str(value)
    if len(string) > TRUNCATE_LIMIT:
        return string[:TRUNCATE_LIMIT] + '...[truncated]'
    else:
        return string