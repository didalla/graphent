import logging
import os
from functools import wraps
from typing import Any, Dict

from lib.Context import Context

LOG_LEVEL = 25
TRUNCATE_LIMIT = int(os.environ.get("TRUNCATE_LIMIT", 250))

class AgentLoggerConfig:
    _setup_done = False
    _last_returned_result = None

    @staticmethod
    def setup(level=LOG_LEVEL, log_file=None):
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


def log_agent_activity(func):
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
    if value is None:
        return "<none>"

    string = str(value)
    if len(string) > TRUNCATE_LIMIT:
        return string[:TRUNCATE_LIMIT] + '...[truncated]'
    else:
        return string