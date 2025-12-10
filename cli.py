#!/usr/bin/env python3
"""
Graphent CLI - A beautiful command-line interface for multi-turn agent conversations.
"""

import os
import sys
import readline  # Enables arrow key navigation and history in input
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from lib import AgentBuilder, Context, AgentLoggerConfig
from lib.tools import get_coords, get_weather, create_todo_tools


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
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   {Colors.GREEN}▄████  ██▀███   ▄▄▄       ██▓███   ██░ ██ ▓█████  ███▄    █ ▄▄▄█████▓{Colors.CYAN}║
║   {Colors.GREEN}██▒ ▀█▒▓██ ▒ ██▒▒████▄    ▓██░  ██▒▓██░ ██▒▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒{Colors.CYAN}║
║   {Colors.GREEN}▒██░▄▄▄░▓██ ░▄█ ▒▒██  ▀█▄  ▓██░ ██▓▒▒██▀▀██░▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░{Colors.CYAN}║
║   {Colors.GREEN}░▓█  ██▓▒██▀▀█▄  ░██▄▄▄▄██ ▒██▄█▓▒ ▒░▓█ ░██ ▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░ {Colors.CYAN}║
║   {Colors.GREEN}░▒▓███▀▒░██▓ ▒██▒ ▓█   ▓██▒▒██▒ ░  ░░▓█▒░██▓░▒████▒▒██░   ▓██░  ▒██▒ ░ {Colors.CYAN}║
║   {Colors.GREEN} ░▒   ▒ ░ ▒▓ ░▒▓░ ▒▒   ▓▒█░▒▓▒░ ░  ░ ▒ ░░▒░▒░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░   {Colors.CYAN}║
║                                                              ║
║   {Colors.YELLOW}Multi-turn Agent Conversation Interface{Colors.CYAN}                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
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
        print(
            f"  {Colors.YELLOW}Model:{Colors.RESET} google/gemini-2.5-flash-preview-09-2025"
        )
        print(f"  {Colors.YELLOW}Provider:{Colors.RESET} OpenRouter")
        print(f"  {Colors.YELLOW}Temperature:{Colors.RESET} 0")
        if self.conversation_start:
            duration = datetime.now() - self.conversation_start
            print(
                f"  {Colors.YELLOW}Session Duration:{Colors.RESET} {str(duration).split('.')[0]}"
            )
        print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}\n")

    def _clear_conversation(self):
        """Clear conversation history and start fresh."""
        self.context = Context()
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

    def _print_thinking(self):
        """Print a thinking indicator."""
        print(
            f"\n{Colors.DIM}{Colors.ITALIC}Agent is thinking...{Colors.RESET}",
            end="",
            flush=True,
        )

    def _clear_thinking(self):
        """Clear the thinking indicator."""
        # Move cursor to beginning of line and clear
        print("\r" + " " * 30 + "\r", end="", flush=True)

    def _setup_agent(self):
        """Initialize the agent and model."""
        # Suppress logging noise during setup
        AgentLoggerConfig.setup()

        self.model = ChatOpenAI(
            model="z-ai/glm-4.6:exacto",
            temperature=0.2,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )

        # Build the weather agent
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
            .build()
        )

        # Build the main agent
        main_agent_builder = (
            AgentBuilder()
            .with_name("Main Agent")
            .with_model(self.model)
            .with_system_prompt(
                "You are qwarki, a helpful agent. Be concise but friendly in your responses."
            )
            .with_description("The main agent can call other agents.")
        )

        # Create todo tools bound to self.context (uses lambda to always get current context)
        self.agent = (
            main_agent_builder.add_agent(weather_agent)
            .add_tools(create_todo_tools(lambda: self.context))
            .build()
        )

        self.context = Context()
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
        self._print_thinking()

        try:
            self.context.add_message(HumanMessage(content=user_input))
            self.context = self.agent.invoke(self.context)

            self._clear_thinking()

            # Get the last AI message
            response = self.context.get_messages()[-1].content
            print(self._format_agent_response(response))

        except Exception as e:
            self._clear_thinking()
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
