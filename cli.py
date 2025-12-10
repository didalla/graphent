#!/usr/bin/env python3
"""
Graphent CLI - A beautiful command-line interface for multi-turn agent conversations.
"""

import os
import sys
import warnings


import readline  # Enables arrow key navigation and history in input
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from lib import (
<<<<<<< HEAD
    AgentBuilder, Context, AgentLoggerConfig, HookRegistry,
    ToolCallEvent, ToolResultEvent,
    ModelCallEvent, DelegationEvent, TodoChangeEvent,
    on_tool_call, after_tool_call,
    before_model_call, on_delegation, on_todo_change
=======
    AgentBuilder,
    Context,
    AgentLoggerConfig,
    HookRegistry,
    ToolCallEvent,
    ToolResultEvent,
    ModelCallEvent,
    DelegationEvent,
    TodoChangeEvent,
    on_tool_call,
    after_tool_call,
    before_model_call,
    on_delegation,
    on_todo_change,
>>>>>>> a407f87 (refactor: remove unused hook imports and reformat CLI code for improved readability)
)
from lib.tools import get_coords, get_weather, create_todo_tools


# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings(
    "ignore", message="Core Pydantic V1 functionality isn't compatible"
)


# ANSI color codes for terminal styling
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    MAGENTA = "\033[35m"


class CLIHooksHandler:
    """Hook handler for displaying agent state in the CLI."""

    def __init__(self, colors: type):
        self.colors = colors
        self._indent = "  "

    @on_tool_call
    def handle_tool_call(self, event: ToolCallEvent):
        """Display when a tool is being called."""
        print(
            f"\n{self._indent}{self.colors.YELLOW}🔧 Tool Call:{self.colors.RESET} "
            f"{self.colors.BOLD}{event.tool_name}{self.colors.RESET}"
        )
        if event.tool_args:
            args_display = ", ".join(
                f"{k}={repr(v)[:50]}" for k, v in event.tool_args.items()
            )
            print(
                f"{self._indent}   {self.colors.DIM}Args: {args_display}{self.colors.RESET}"
            )

    @after_tool_call
    def handle_tool_result(self, event: ToolResultEvent):
        """Display tool result summary."""
        result_preview = str(event.result)[:100]
        if len(str(event.result)) > 100:
            result_preview += "..."
        print(
            f"{self._indent}   {self.colors.DIM}Result: {result_preview}{self.colors.RESET}"
        )

    @before_model_call
    def handle_before_model(self, event: ModelCallEvent):
        """Display which agent is currently thinking."""
        print(
            f"\n{self._indent}{self.colors.MAGENTA}🤖 Agent:{self.colors.RESET} "
            f"{self.colors.BOLD}{event.agent_name}{self.colors.RESET} "
            f"{self.colors.DIM}(processing {event.message_count} messages){self.colors.RESET}"
        )

    @on_delegation
    def handle_delegation(self, event: DelegationEvent):
        """Display when delegation happens between agents."""
        print(
            f"\n{self._indent}{self.colors.CYAN}🔀 Delegation:{self.colors.RESET} "
            f"{self.colors.BOLD}{event.from_agent}{self.colors.RESET} → "
            f"{self.colors.BOLD}{event.to_agent}{self.colors.RESET}"
        )
        task_preview = event.task[:80] + "..." if len(event.task) > 80 else event.task
        print(
            f"{self._indent}   {self.colors.DIM}Task: {task_preview}{self.colors.RESET}"
        )

    @on_todo_change
    def handle_todo_change(self, event: TodoChangeEvent):
        """Display todo list changes."""
        if event.action == "add":
            icon = "➕ "
            action_text = f"Added todo #{event.todo_id}"
            details = f'"{event.title}"'
        elif event.action == "update":
            icon = "✏️ "
            if event.old_state and event.old_state != event.state:
                action_text = f"Updated todo #{event.todo_id}"
                details = f"{event.old_state} → {event.state}"
            else:
                action_text = f"Updated todo #{event.todo_id}"
                details = f'"{event.title}"' if event.title else ""
        elif event.action == "delete":
            icon = "🗑️ "
            action_text = f"Deleted todo #{event.todo_id}"
            details = f'"{event.title}"' if event.title else ""
        else:
            return

        print(
            f"{self._indent}{self.colors.GREEN}{icon} {action_text}{self.colors.RESET} "
            f"{self.colors.DIM}{details}{self.colors.RESET}"
        )


class CLI:
    """Interactive CLI for multi-turn conversations with agents."""

    COMMANDS = {
        "/help": "Show this help message",
        "/clear": "Clear conversation history and start fresh",
        "/history": "Show conversation history",
        "/exit": "Exit the CLI (also: /quit, /q)",
        "/model": "Show current model information",
    }

    def __init__(self):
        self.context: Optional[Context] = None
        self.agent = None
        self.model = None
        self.conversation_start: Optional[datetime] = None
        self._hooks_handler = CLIHooksHandler(Colors)
        self._hooks = HookRegistry()
        self._hooks.register_hooks_from_object(self._hooks_handler)
        self._setup_readline()

    def _setup_readline(self):
        """Configure readline for better input handling."""
        # Enable tab completion (empty for now)
        readline.parse_and_bind("tab: complete")
        # Set history file
        histfile = os.path.expanduser("~/.graphent_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        import atexit

        atexit.register(readline.write_history_file, histfile)

    def _print_header(self):
        """Print the CLI header/banner."""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  {Colors.GREEN} ██████  ██████   █████  ██████  ██   ██ ███████ ███    ██ ████████ {Colors.CYAN}  ║
║  {Colors.GREEN}██       ██   ██ ██   ██ ██   ██ ██   ██ ██      ████   ██    ██    {Colors.CYAN}  ║
║  {Colors.GREEN}██   ███ ██████  ███████ ██████  ███████ █████   ██ ██  ██    ██    {Colors.CYAN}  ║
║  {Colors.GREEN}██    ██ ██   ██ ██   ██ ██      ██   ██ ██      ██  ██ ██    ██    {Colors.CYAN}  ║
║  {Colors.GREEN} ██████  ██   ██ ██   ██ ██      ██   ██ ███████ ██   ████    ██    {Colors.CYAN}  ║
║                                                                        ║
║              {Colors.YELLOW}Multi-turn Agent Conversation Interface{Colors.CYAN}                   ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(banner)

    def _print_welcome(self):
        """Print welcome message with instructions."""
        print(f"\n{Colors.DIM}Type your message to chat with the agent.{Colors.RESET}")
        print(
            f"{Colors.DIM}Type {Colors.YELLOW}/help{Colors.DIM} for available commands.{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}Press {Colors.YELLOW}Ctrl+C{Colors.DIM} or type {Colors.YELLOW}/exit{Colors.DIM} to quit.{Colors.RESET}\n"
        )

    def _print_help(self):
        """Print help message with available commands."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Commands:{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {Colors.YELLOW}{cmd:<12}{Colors.RESET} {desc}")
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}\n")

    def _print_history(self):
        """Print the conversation history."""
        if not self.context or not self.context.get_messages():
            print(f"\n{Colors.DIM}No conversation history yet.{Colors.RESET}\n")
            return

        print(f"\n{Colors.BOLD}{Colors.CYAN}Conversation History:{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

        for i, msg in enumerate(self.context.get_messages(), 1):
            if isinstance(msg, HumanMessage):
                role = f"{Colors.GREEN}You{Colors.RESET}"
                content = msg.content
            elif isinstance(msg, AIMessage):
                role = f"{Colors.BLUE}Agent{Colors.RESET}"
                content = msg.content if msg.content else "[Tool call]"
            elif isinstance(msg, ToolMessage):
                role = f"{Colors.YELLOW}Tool{Colors.RESET}"
                content = (
                    str(msg.content)[:100] + "..."
                    if len(str(msg.content)) > 100
                    else msg.content
                )
            elif isinstance(msg, SystemMessage):
                role = f"{Colors.DIM}System{Colors.RESET}"
                content = (
                    msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                )
            else:
                role = f"{Colors.DIM}Unknown{Colors.RESET}"
                content = str(msg)

            print(f"  {Colors.DIM}[{i}]{Colors.RESET} {role}: {content}")

        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}\n")

    def _print_model_info(self):
        """Print current model information."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Model Information:{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
        if self.model:
            print(f"  {Colors.YELLOW}Model:{Colors.RESET} {self.model.model_name}")
            print(
                f"  {Colors.YELLOW}Provider:{Colors.RESET} {self.model.openai_api_base or 'OpenAI'}"
            )
            print(
                f"  {Colors.YELLOW}Temperature:{Colors.RESET} {self.model.temperature}"
            )
        else:
            print(f"  {Colors.DIM}No model initialized{Colors.RESET}")
        if self.conversation_start:
            duration = datetime.now() - self.conversation_start
            print(
                f"  {Colors.YELLOW}Session Duration:{Colors.RESET} {str(duration).split('.')[0]}"
            )
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}\n")

    def _clear_conversation(self):
        """Clear conversation history and start fresh."""
        self.context = Context(hooks=self._hooks)
        self.conversation_start = datetime.now()
        print(
            f"\n{Colors.GREEN}✓ Conversation cleared. Starting fresh!{Colors.RESET}\n"
        )

    def _format_user_prompt(self) -> str:
        """Format the user input prompt."""
        return f"{Colors.GREEN}{Colors.BOLD}You ❯{Colors.RESET} "

    def _format_agent_response(self, response: str) -> str:
        """Format the agent's response for display."""
        return f"\n{Colors.BLUE}{Colors.BOLD}Agent ❯{Colors.RESET} {response}\n"

    def _setup_agent(self):
        """Initialize the agent and model."""
        # Disable console logging - use a file instead for debugging
        log_file = os.path.expanduser("~/.graphent_debug.log")
        AgentLoggerConfig.setup(log_file=log_file)

        self.model = ChatOpenAI(
            model="z-ai/glm-4.6:exacto",
            temperature=0.3,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )

        # Build the weather agent with hooks
        weather_agent = (
            AgentBuilder()
            .with_name("Weather Agent")
            .with_description("Agent that can get the weather at a location.")
            .with_model(self.model)
            .with_system_prompt(
                "Answer questions about the weather using the get_weather tool."
            )
            .add_tool(get_weather)
            .add_tool(get_coords)
            .with_hooks(self._hooks)
            .build()
        )

        # Build the main agent with hooks
        main_agent_builder = (
            AgentBuilder()
            .with_name("Main Agent")
            .with_model(self.model)
            .with_system_prompt(
                "You are qwarki, a helpful agent. Be concise but friendly in your responses."
            )
            .with_description("The main agent can call other agents.")
            .with_hooks(self._hooks)
        )

        # Create context with hooks for todo change events
        self.context = Context(hooks=self._hooks)

        # Create todo tools bound to self.context (uses lambda to always get current context)
        self.agent = (
            main_agent_builder.add_agent(weather_agent)
            .add_tools(create_todo_tools(lambda: self.context))
            .build()
        )

        self.conversation_start = datetime.now()

    def _handle_command(self, command: str) -> bool:
        """
        Handle CLI commands.
        Returns True if should continue, False if should exit.
        """
        command = command.strip().lower()

        if command in ["/exit", "/quit", "/q"]:
            return False
        elif command == "/help":
            self._print_help()
        elif command == "/clear":
            self._clear_conversation()
        elif command == "/history":
            self._print_history()
        elif command == "/model":
            self._print_model_info()
        else:
            print(f"\n{Colors.RED}Unknown command: {command}{Colors.RESET}")
            print(f"{Colors.DIM}Type /help for available commands.{Colors.RESET}\n")

        return True

    def _process_message(self, user_input: str):
        """Process a user message and get agent response."""
        try:
            self.context.add_message(HumanMessage(content=user_input))
            self.context = self.agent.invoke(self.context)

            # Get the last AI message
            response = self.context.get_messages()[-1].content
            print(self._format_agent_response(response))

        except Exception as e:
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}\n")

    def run(self):
        """Run the main CLI loop."""
        # Check for API key
        if not os.environ.get("OPENROUTER_API_KEY"):
            print(
                f"\n{Colors.RED}Error: OPENROUTER_API_KEY environment variable not set.{Colors.RESET}"
            )
            print(f"{Colors.DIM}Please set it before running the CLI:{Colors.RESET}")
            print(
                f"{Colors.YELLOW}  export OPENROUTER_API_KEY=your_api_key{Colors.RESET}\n"
            )
            sys.exit(1)

        self._print_header()

        print(f"{Colors.DIM}Initializing agent...{Colors.RESET}", end="", flush=True)
        try:
            self._setup_agent()
            print(
                f"\r{Colors.GREEN}✓ Agent initialized successfully!{Colors.RESET}     "
            )
        except Exception as e:
            print(f"\r{Colors.RED}✗ Failed to initialize agent: {e}{Colors.RESET}")
            sys.exit(1)

        self._print_welcome()

        try:
            while True:
                try:
                    user_input = input(self._format_user_prompt())

                    # Skip empty input
                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.strip().startswith("/"):
                        if not self._handle_command(user_input):
                            break
                        continue

                    # Process regular message
                    self._process_message(user_input)

                except EOFError:
                    # Handle Ctrl+D
                    print()
                    break

        except KeyboardInterrupt:
            pass

        # Goodbye message
        print(f"\n{Colors.CYAN}{'─' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}Thanks for using Graphent! Goodbye! 👋{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}\n")


def main():
    """Entry point for the CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
