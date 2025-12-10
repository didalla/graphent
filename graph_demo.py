"""Demo CLI for Graphent's graph-based architecture.

This script demonstrates the new graph features:
- ActionNode: Simple single-call nodes
- AgentNode: Full agent with tool support
- ClassifierNode: Branching based on LLM classification
- Graph execution with hooks and streaming

Usage:
    python graph_demo.py

Requires OPENROUTER_API_KEY environment variable.
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from lib.graph import (
    GraphBuilder,
    GraphHookRegistry,
    GraphHookType,
    NodeEnterEvent,
    NodeExitEvent,
    ClassificationEvent,
)
from lib.Context import Context
from lib import AgentBuilder

console = Console()


def get_model(streaming: bool = False) -> ChatOpenAI:
    """Get the configured LLM model."""
    return ChatOpenAI(
        model="z-ai/glm-4.6:exacto",
        temperature=0.3,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        streaming=streaming,
    )


def create_demo_hooks() -> GraphHookRegistry:
    """Create hooks that log graph execution."""
    hooks = GraphHookRegistry()

    def on_enter(event: NodeEnterEvent):
        console.print(
            f"  [dim]→ Entering node:[/dim] [cyan]{event.node_name}[/cyan] ({event.node_type})"
        )

    def on_exit(event: NodeExitEvent):
        console.print(f"  [dim]← Exited node:[/dim] [cyan]{event.node_name}[/cyan]")

    def on_classify(event: ClassificationEvent):
        console.print(
            f"  [dim]⚡ Classification:[/dim] [yellow]{event.classification}[/yellow] → [green]{event.target_node}[/green]"
        )

    hooks.register(GraphHookType.ON_NODE_ENTER, on_enter)
    hooks.register(GraphHookType.ON_NODE_EXIT, on_exit)
    hooks.register(GraphHookType.ON_CLASSIFICATION, on_classify)

    return hooks


def demo_simple_chain():
    """Demo 1: Simple linear chain of action nodes."""
    console.print(
        Panel.fit(
            "[bold]Demo 1: Simple Chain[/bold]\n"
            "Two action nodes in sequence: Summarize → Format",
            border_style="blue",
        )
    )

    model = get_model()

    graph = (
        GraphBuilder()
        .add_action_node(
            "summarizer",
            model,
            "You are a summarizer. Summarize the user's input in 2-3 sentences.",
        )
        .add_action_node(
            "formatter",
            model,
            "You are a formatter. Take the previous summary and format it as a bullet-point list with emojis.",
        )
        .connect("summarizer", "formatter")
        .set_entry("summarizer")
        .set_finish("formatter")
        .with_hooks(create_demo_hooks())
        .build()
    )

    user_input = "Python is a high-level programming language known for its readability. It was created by Guido van Rossum and released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."

    console.print(f"\n[bold]Input:[/bold] {user_input[:80]}...")
    console.print("\n[bold]Execution:[/bold]")

    context = Context().add_message(HumanMessage(content=user_input))
    result = graph.invoke(context)

    final_message = result.get_messages()[-1].content
    console.print(f"\n[bold]Output:[/bold]")
    console.print(Panel(Markdown(final_message), border_style="green"))


def demo_classifier_branching():
    """Demo 2: Classifier node with conditional branching."""
    console.print(
        Panel.fit(
            "[bold]Demo 2: Classifier Branching[/bold]\n"
            "Router classifies input, then routes to technical or creative handler",
            border_style="magenta",
        )
    )

    model = get_model()

    graph = (
        GraphBuilder()
        .add_classifier_node(
            "router", model, classes=["technical", "creative", "general"]
        )
        .add_action_node(
            "tech_handler",
            model,
            "You are a technical expert. Provide a detailed, technical response with code examples if relevant.",
        )
        .add_action_node(
            "creative_handler",
            model,
            "You are a creative writer. Respond with imagination, using metaphors and storytelling.",
        )
        .add_action_node(
            "general_handler",
            model,
            "You are a helpful assistant. Provide a clear, balanced response.",
        )
        .branch(
            "router",
            {
                "technical": "tech_handler",
                "creative": "creative_handler",
                "general": "general_handler",
            },
        )
        .set_entry("router")
        .with_hooks(create_demo_hooks())
        .build()
    )

    # Test with a technical question
    user_input = "How do I implement a binary search tree in Python?"

    console.print(f"\n[bold]Input:[/bold] {user_input}")
    console.print("\n[bold]Execution:[/bold]")

    context = Context().add_message(HumanMessage(content=user_input))
    result = graph.invoke(context)

    final_message = result.get_messages()[-1].content
    console.print(f"\n[bold]Output:[/bold]")
    console.print(
        Panel(
            Markdown(
                final_message[:1500] + "..."
                if len(final_message) > 1500
                else final_message
            ),
            border_style="green",
        )
    )


def demo_streaming():
    """Demo 3: Streaming graph execution."""
    console.print(
        Panel.fit(
            "[bold]Demo 3: Streaming Output[/bold]\n"
            "Watch the graph stream its response in real-time",
            border_style="yellow",
        )
    )

    model = get_model(streaming=True)

    graph = (
        GraphBuilder()
        .add_action_node(
            "storyteller",
            model,
            "You are a storyteller. Tell a very short story (4-5 sentences) about the topic the user provides.",
        )
        .set_entry("storyteller")
        .set_finish("storyteller")
        .build()
    )

    user_input = "a robot learning to paint"

    console.print(f"\n[bold]Input:[/bold] {user_input}")
    console.print("\n[bold]Streaming output:[/bold]")

    context = Context().add_message(HumanMessage(content=user_input))

    # Stream the response
    output = Text()
    with Live(Panel(output, border_style="green"), refresh_per_second=10) as live:
        for chunk in graph.stream(context):
            output.append(chunk)
            live.update(Panel(output, border_style="green"))


def demo_agent_node():
    """Demo 4: AgentNode with tool access."""
    console.print(
        Panel.fit(
            "[bold]Demo 4: Agent Node with Tools[/bold]\n"
            "An agent node that can use the calculator tool",
            border_style="cyan",
        )
    )

    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression. Use Python syntax."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    model = get_model()

    # Create an agent with the calculator tool
    calc_agent = (
        AgentBuilder()
        .with_name("Calculator Agent")
        .with_model(model)
        .with_system_prompt(
            "You are a math assistant. Use the calculator tool to solve math problems. Always show your work."
        )
        .with_description("Math assistant with calculator")
        .add_tool(calculator)
        .build()
    )

    graph = (
        GraphBuilder()
        .add_agent_node("math_solver", calc_agent)
        .set_entry("math_solver")
        .set_finish("math_solver")
        .with_hooks(create_demo_hooks())
        .build()
    )

    user_input = "What is 1234 * 5678?"

    console.print(f"\n[bold]Input:[/bold] {user_input}")
    console.print("\n[bold]Execution:[/bold]")

    context = Context().add_message(HumanMessage(content=user_input))
    result = graph.invoke(context)

    final_message = result.get_messages()[-1].content
    console.print(f"\n[bold]Output:[/bold]")
    console.print(Panel(final_message, border_style="green"))


def main():
    """Run all demos."""
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print(
            "[red]Error: OPENROUTER_API_KEY environment variable not set[/red]"
        )
        console.print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold blue]🔷 Graphent Graph Architecture Demo 🔷[/bold blue]\n\n"
            "This demo showcases the new graph-based workflow features:\n"
            "• ActionNode - Simple single LLM calls\n"
            "• AgentNode - Full agents with tool support\n"
            "• ClassifierNode - Conditional branching\n"
            "• Streaming - Real-time output",
            border_style="blue",
        )
    )

    demos = [
        ("Simple Chain", demo_simple_chain),
        ("Classifier Branching", demo_classifier_branching),
        ("Streaming", demo_streaming),
        ("Agent Node with Tools", demo_agent_node),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        console.print(f"\n{'=' * 60}\n")
        try:
            demo_func()
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")

        if i < len(demos):
            console.print(
                "\n[dim]Press Enter for next demo, or Ctrl+C to exit...[/dim]"
            )
            try:
                input()
            except KeyboardInterrupt:
                break

    console.print("\n[bold green]✓ Demo complete![/bold green]\n")


if __name__ == "__main__":
    main()
