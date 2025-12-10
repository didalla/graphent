# YAML Configuration

Graphent supports loading agents and graphs from YAML configuration files, making it easy to define and modify agent configurations without changing code.

## Overview

YAML configuration allows you to:

- Define agents declaratively
- Configure models, prompts, and tools
- Build graph workflows from YAML
- Share configurations across projects

## Loading Agents from YAML

```python
from lib import load_agent_from_yaml

# Load an agent from a YAML file
agent = load_agent_from_yaml("config/my_agent.yaml")

# Use the agent normally
result = agent.invoke(context)
```

## Agent YAML Schema

```yaml
# my_agent.yaml
name: Assistant
description: A helpful assistant

model:
  provider: openai
  model_name: gpt-4
  temperature: 0.7

system_prompt: |
  You are a helpful assistant that answers questions
  clearly and concisely.

tools:
  - name: get_weather
    module: my_tools
  - name: search_web
    module: my_tools
```

### Model Configuration

```yaml
model:
  provider: openai          # openai, anthropic, etc.
  model_name: gpt-4         # Model name
  temperature: 0.7          # Optional: temperature
  api_key: ${OPENAI_API_KEY}  # Supports env variables
  base_url: https://api.openai.com/v1  # Optional: custom endpoint
```

### Inline Model Definition

You can define the model inline or reference an external configuration:

```yaml
# Inline definition
model:
  provider: openai
  model_name: gpt-4

# Or reference external config
model: ./models/gpt4.yaml
```

## Sub-Agents in YAML

Define sub-agents inline or by reference:

```yaml
name: Orchestrator
description: Main orchestrator agent

model:
  provider: openai
  model_name: gpt-4

system_prompt: You coordinate specialized agents.

sub_agents:
  # Inline sub-agent
  - name: Weather Agent
    description: Handles weather queries
    model:
      provider: openai
      model_name: gpt-4
    system_prompt: You answer weather questions.
    tools:
      - name: get_weather
        module: my_tools
  
  # Reference to external file
  - file: ./agents/math_agent.yaml
```

## Tool Registration

Tools can be loaded from Python modules:

```yaml
tools:
  - name: my_tool
    module: my_package.tools
  
  # Or from a registry
  - registry: default
    name: web_search
```

### Creating a Tool Registry

```python
from lib.yaml_loader import ToolRegistry

# Create and populate a registry
registry = ToolRegistry()
registry.register("web_search", my_web_search_tool)
registry.register("calculator", my_calculator_tool)

# Load agent with registry
agent = load_agent_from_yaml("config/agent.yaml", tool_registry=registry)
```

## Loading Graphs from YAML

```python
from lib import load_graph_from_yaml

# Load a graph from YAML
graph = load_graph_from_yaml("config/my_graph.yaml")

# Run the graph
result = graph.run(context)
```

## Graph YAML Schema

```yaml
# my_graph.yaml
name: Intent Pipeline

nodes:
  - type: classifier
    name: intent_classifier
    model:
      provider: openai
      model_name: gpt-4
    categories:
      - question
      - command
      - greeting
    
  - type: action
    name: handle_question
    model:
      provider: openai
      model_name: gpt-4
    prompt: Answer the user's question.
    
  - type: agent
    name: task_executor
    agent:
      file: ./agents/task_agent.yaml

edges:
  - from: __start__
    to: intent_classifier
    
  - from: intent_classifier
    type: conditional
    routes:
      question: handle_question
      command: task_executor
      greeting: handle_greeting
      
  - from: handle_question
    to: __end__
```

## Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
model:
  provider: openai
  model_name: gpt-4
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_BASE_URL}
```

## Complete Example

Here's a complete example with all features:

```yaml
# orchestrator.yaml
name: Smart Orchestrator
description: Routes queries to specialized agents

model:
  provider: openai
  model_name: gpt-4
  temperature: 0.3
  api_key: ${OPENAI_API_KEY}

system_prompt: |
  You are an intelligent orchestrator that routes user queries
  to the most appropriate specialized agent.
  
  Available specialists:
  - Weather Agent: For weather queries
  - Math Agent: For calculations
  - Search Agent: For web searches

sub_agents:
  - name: Weather Agent
    description: Gets weather information for any location
    model:
      provider: openai
      model_name: gpt-4
    system_prompt: You answer weather questions using available tools.
    tools:
      - name: get_coords
        module: my_tools.weather
      - name: get_weather
        module: my_tools.weather
  
  - file: ./agents/math_agent.yaml
  - file: ./agents/search_agent.yaml
```

For complete API documentation, see the [API Reference](../reference/index.md#yaml-loading).
