# Multi-Agent Systems

Graphent supports building multi-agent systems where agents can delegate tasks to specialized sub-agents. This enables complex workflows where different agents handle different domains.

## Overview

Multi-agent systems in Graphent:

- Agents can have sub-agents that handle specific tasks
- Delegation happens automatically via the `hand_off_to_subagent` tool
- Sub-agents inherit the conversation context
- Results are returned to the parent agent

## Creating Sub-Agents

First, create specialized agents:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from lib import AgentBuilder

model = ChatOpenAI(model="gpt-4")

# Create specialized tools
@tool
def get_coords(location: str) -> str:
    """Get coordinates for a location."""
    # Implementation here
    return "52.52, 13.405"

@tool
def get_weather(coords: str) -> str:
    """Get weather for coordinates."""
    # Implementation here
    return "Sunny, 22°C"

# Create a weather specialist agent
weather_agent = (AgentBuilder()
    .with_name("Weather Agent")
    .with_model(model)
    .with_system_prompt("You answer weather questions using the available tools.")
    .with_description("Gets weather information for locations")
    .add_tool(get_coords)
    .add_tool(get_weather)
    .build())
```

## Adding Sub-Agents

Use `add_agent()` to add sub-agents to a parent agent:

```python
# Create the main orchestrator agent
main_agent = (AgentBuilder()
    .with_name("Main Agent")
    .with_model(model)
    .with_system_prompt("""You are a helpful assistant. 
    When asked about weather, delegate to the Weather Agent.""")
    .with_description("Main orchestrator agent")
    .add_agent(weather_agent)
    .build())
```

## How Delegation Works

When you add sub-agents, Graphent automatically creates a `hand_off_to_subagent` tool that the parent agent can use:

```python
from langchain_core.messages import HumanMessage
from lib import Context

context = Context().add_message(
    HumanMessage(content="What's the weather in Berlin?")
)

# The main agent will:
# 1. Recognize this is a weather question
# 2. Call hand_off_to_subagent with target="Weather Agent"
# 3. Weather Agent processes the request
# 4. Result is returned to main agent
result = main_agent.invoke(context)
```

## Multiple Sub-Agents

You can add multiple specialized sub-agents:

```python
# Create multiple specialists
weather_agent = (AgentBuilder()
    .with_name("Weather Agent")
    .with_model(model)
    .with_system_prompt("You handle weather queries.")
    .with_description("Gets weather information")
    .add_tool(get_weather)
    .build())

math_agent = (AgentBuilder()
    .with_name("Math Agent")
    .with_model(model)
    .with_system_prompt("You solve mathematical problems.")
    .with_description("Solves math problems and calculations")
    .add_tool(calculate)
    .build())

search_agent = (AgentBuilder()
    .with_name("Search Agent")
    .with_model(model)
    .with_system_prompt("You search for information.")
    .with_description("Searches the web for information")
    .add_tool(web_search)
    .build())

# Create orchestrator with all sub-agents
orchestrator = (AgentBuilder()
    .with_name("Orchestrator")
    .with_model(model)
    .with_system_prompt("""You are a helpful assistant that coordinates 
    specialized agents. Route queries to the appropriate specialist.""")
    .with_description("Main orchestrator")
    .add_agent(weather_agent)
    .add_agent(math_agent)
    .add_agent(search_agent)
    .build())
```

## Monitoring Delegation

Use the `@on_delegation` hook to monitor delegation events:

```python
from lib import on_delegation, DelegationEvent

class DelegationMonitor:
    @on_delegation
    def log_delegation(self, event: DelegationEvent):
        print(f"{event.agent_name} delegating to {event.target_agent}")
        print(f"Task: {event.task}")

orchestrator = (AgentBuilder()
    .with_name("Orchestrator")
    .with_model(model)
    .with_system_prompt("You coordinate specialists.")
    .with_description("Orchestrator")
    .add_agent(weather_agent)
    .add_hooks_from_object(DelegationMonitor())
    .build())
```

## Best Practices

!!! tip "Agent Descriptions Matter"
    Write clear, specific descriptions for sub-agents. The parent agent uses these descriptions to decide which sub-agent to delegate to.

!!! tip "Keep Agents Focused"
    Each sub-agent should have a clear, focused responsibility. This makes delegation decisions easier for the orchestrator.

!!! warning "Avoid Deep Nesting"
    While sub-agents can have their own sub-agents, deep nesting can make debugging difficult. Try to keep hierarchies shallow.
