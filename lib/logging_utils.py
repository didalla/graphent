import logging
from functools import wraps

LOG_LEVEL = 25

class AgentLoggerConfig:
    _setup_done = False

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

def log_agent_activity(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        agent_name = getattr(self, "name", "Unknown Agent")
        method_name = func.__name__

        inputs = format_args(args)

        logging.log(LOG_LEVEL,f"START {agent_name}.{method_name} WITH {inputs[-1]}")
        # logging.info(f"Calling function: {method_name} with agent: {agent_name}.")
        # logging.info(f"Input args: {args}, kwargs: {kwargs}")

        result = func(self, *args, **kwargs)

        logging.log(LOG_LEVEL,f"Function {func.__name__} returned: {result}")

        return result

    return wrapper

def format_args(args):
    arg_strings = []
    for arg in args[0]:
        s = str(arg)
        arg_strings.append(s)

    return arg_strings