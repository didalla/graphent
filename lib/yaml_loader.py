"""YAML configuration loader for Graphent agents and graphs.

This module provides functions to load agents and graphs from YAML
configuration files, enabling easy declarative configuration.

Example:
    >>> from lib.yaml_loader import load_graph_from_yaml
    >>> graph = load_graph_from_yaml("my_graph.yaml", tools_registry={"calculator": calc_tool})
    >>> result = graph.invoke(context)
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from lib.Agent import Agent
from lib.AgentBuilder import AgentBuilder
from lib.graph import Graph, GraphBuilder
from lib.exceptions import AgentConfigurationError


# Environment variable pattern: ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variable references in a value.

    Args:
        value: A string potentially containing ${VAR_NAME} references,
               or a non-string value to return unchanged.

    Returns:
        The value with environment variables resolved.
    """
    if not isinstance(value, str):
        return value

    def replace_env(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name, "")
        if not env_value:
            raise AgentConfigurationError(
                f"Environment variable '{var_name}' is not set"
            )
        return env_value

    return ENV_VAR_PATTERN.sub(replace_env, value)


def _create_model_from_config(config: dict[str, Any]) -> BaseChatModel:
    """Create a LangChain chat model from configuration.

    Args:
        config: Model configuration dict with provider, name, temperature, etc.

    Returns:
        A configured BaseChatModel instance.

    Raises:
        AgentConfigurationError: If provider is unsupported or config is invalid.
    """
    provider = config.get("provider", "openai").lower()
    model_name = config.get("name", "gpt-4")
    temperature = config.get("temperature", 0.7)
    api_key = _resolve_env_vars(config.get("api_key", ""))
    base_url = config.get("base_url")
    streaming = config.get("streaming", False)

    if provider in ("openai", "openrouter", "azure"):
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key,
            "streaming": streaming,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    else:
        raise AgentConfigurationError(
            f"Unsupported model provider: {provider}. "
            "Supported providers: openai, openrouter, azure"
        )


def _resolve_model(
    model_ref: Union[str, dict[str, Any]], models_registry: dict[str, dict[str, Any]]
) -> BaseChatModel:
    """Resolve a model reference to a model instance.

    Args:
        model_ref: Either a string name referencing models_registry,
                   or an inline model config dict.
        models_registry: Dictionary of named model configurations.

    Returns:
        A configured BaseChatModel instance.

    Raises:
        AgentConfigurationError: If model reference cannot be resolved.
    """
    if isinstance(model_ref, str):
        # Reference to named model
        if model_ref not in models_registry:
            raise AgentConfigurationError(
                f"Model '{model_ref}' not found in models registry. "
                f"Available models: {list(models_registry.keys())}"
            )
        return _create_model_from_config(models_registry[model_ref])
    elif isinstance(model_ref, dict):
        # Inline model config
        return _create_model_from_config(model_ref)
    else:
        raise AgentConfigurationError(
            f"Invalid model reference type: {type(model_ref)}. "
            "Expected string name or dict config."
        )


def _build_agent_from_config(
    config: dict[str, Any],
    models_registry: dict[str, dict[str, Any]],
    agents_registry: dict[str, Agent],
    tools_registry: dict[str, BaseTool],
) -> Agent:
    """Build an Agent from a configuration dict.

    Args:
        config: Agent configuration with name, description, system_prompt, etc.
        models_registry: Named model configurations.
        agents_registry: Named agent instances for sub_agents lookup.
        tools_registry: Named tool instances.

    Returns:
        A configured Agent instance.
    """
    name = config.get("name")
    description = config.get("description")
    system_prompt = config.get("system_prompt")

    if not all([name, description, system_prompt]):
        raise AgentConfigurationError(
            "Agent config requires 'name', 'description', and 'system_prompt'"
        )

    # Resolve model
    model_ref = config.get("model")
    if not model_ref:
        raise AgentConfigurationError(
            f"Agent '{name}' requires a 'model' configuration"
        )
    model = _resolve_model(model_ref, models_registry)

    # Build agent
    builder = (
        AgentBuilder()
        .with_name(name)
        .with_description(description)
        .with_system_prompt(system_prompt)
        .with_model(model)
    )

    # Add tools
    tool_names = config.get("tools", [])
    for tool_name in tool_names:
        if tool_name not in tools_registry:
            raise AgentConfigurationError(
                f"Tool '{tool_name}' not found in tools registry for agent '{name}'"
            )
        builder.add_tool(tools_registry[tool_name])

    # Add sub-agents
    sub_agents_config = config.get("sub_agents", [])
    for sub_agent_ref in sub_agents_config:
        if isinstance(sub_agent_ref, str):
            # Reference by name - look in agents_registry first, then local agents
            if sub_agent_ref in agents_registry:
                builder.add_agent(agents_registry[sub_agent_ref])
            else:
                raise AgentConfigurationError(
                    f"Sub-agent '{sub_agent_ref}' not found in agents section or registry"
                )
        elif isinstance(sub_agent_ref, dict):
            # Inline agent definition
            sub_agent = _build_agent_from_config(
                sub_agent_ref, models_registry, agents_registry, tools_registry
            )
            builder.add_agent(sub_agent)
        else:
            raise AgentConfigurationError(
                f"Invalid sub_agent reference type: {type(sub_agent_ref)}"
            )

    return builder.build()


def load_agent_from_yaml(
    path: Union[str, Path],
    tools_registry: Optional[dict[str, BaseTool]] = None,
    agents_registry: Optional[dict[str, Agent]] = None,
) -> Agent:
    """Load an Agent from a YAML configuration file.

    Args:
        path: Path to the YAML file.
        tools_registry: Optional dict mapping tool names to BaseTool instances.
        agents_registry: Optional dict mapping agent names to Agent instances
                        for sub-agent references.

    Returns:
        A configured Agent instance.

    Raises:
        AgentConfigurationError: If configuration is invalid.
        FileNotFoundError: If the YAML file doesn't exist.

    Example:
        >>> agent = load_agent_from_yaml(
        ...     "my_agent.yaml",
        ...     tools_registry={"calculator": calc_tool}
        ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    tools_registry = tools_registry or {}
    agents_registry = agents_registry or {}

    # Build models registry from config
    models_registry = {}
    if "model" in config and isinstance(config["model"], dict):
        # Single model definition at root level
        models_registry["default"] = config["model"]
    if "models" in config:
        models_registry.update(config["models"])

    return _build_agent_from_config(
        config, models_registry, agents_registry, tools_registry
    )


def load_graph_from_yaml(
    path: Union[str, Path],
    tools_registry: Optional[dict[str, BaseTool]] = None,
    agents_registry: Optional[dict[str, Agent]] = None,
) -> Graph:
    """Load a Graph from a YAML configuration file.

    Args:
        path: Path to the YAML file.
        tools_registry: Optional dict mapping tool names to BaseTool instances.
        agents_registry: Optional dict mapping agent names to Agent instances.

    Returns:
        A configured Graph instance.

    Raises:
        AgentConfigurationError: If configuration is invalid.
        FileNotFoundError: If the YAML file doesn't exist.

    Example:
        >>> graph = load_graph_from_yaml(
        ...     "my_graph.yaml",
        ...     tools_registry={"calculator": calc_tool},
        ...     agents_registry={"helper": helper_agent}
        ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    tools_registry = tools_registry or {}
    agents_registry = dict(agents_registry) if agents_registry else {}

    # Build models registry
    models_registry: dict[str, dict[str, Any]] = {}
    if "models" in config:
        models_registry.update(config["models"])

    # Get default model name (first in models or "default")
    default_model_name = (
        next(iter(models_registry.keys()), None) if models_registry else None
    )

    # Build inline agents and add to agents_registry
    inline_agents = config.get("agents", {})
    for agent_key, agent_config in inline_agents.items():
        # Use the key as name if not specified in config
        if "name" not in agent_config:
            agent_config["name"] = agent_key
        # Use default model if not specified
        if "model" not in agent_config and default_model_name:
            agent_config["model"] = default_model_name

        agent = _build_agent_from_config(
            agent_config, models_registry, agents_registry, tools_registry
        )
        agents_registry[agent_key] = agent

    # Build graph
    builder = GraphBuilder()

    # Add nodes
    nodes = config.get("nodes", [])
    for node_config in nodes:
        node_name = node_config.get("name")
        node_type = node_config.get("type", "action")

        if not node_name:
            raise AgentConfigurationError("Node requires a 'name'")

        # Get model for this node
        model_ref = node_config.get("model", default_model_name)

        if node_type == "action":
            system_prompt = node_config.get("system_prompt", "")
            if not model_ref:
                raise AgentConfigurationError(
                    f"Action node '{node_name}' requires a model"
                )
            model = _resolve_model(model_ref, models_registry)
            builder.add_action_node(node_name, model, system_prompt)

        elif node_type == "classifier":
            classes = node_config.get("classes", [])
            if not classes:
                raise AgentConfigurationError(
                    f"Classifier node '{node_name}' requires 'classes'"
                )
            if not model_ref:
                raise AgentConfigurationError(
                    f"Classifier node '{node_name}' requires a model"
                )
            model = _resolve_model(model_ref, models_registry)
            system_prompt = node_config.get("system_prompt")
            builder.add_classifier_node(node_name, model, classes, system_prompt)

        elif node_type == "agent":
            agent_ref = node_config.get("agent")
            if not agent_ref:
                raise AgentConfigurationError(
                    f"Agent node '{node_name}' requires 'agent' reference"
                )
            if agent_ref not in agents_registry:
                raise AgentConfigurationError(
                    f"Agent '{agent_ref}' not found for node '{node_name}'. "
                    f"Available agents: {list(agents_registry.keys())}"
                )
            builder.add_agent_node(node_name, agents_registry[agent_ref])

        else:
            raise AgentConfigurationError(
                f"Unknown node type '{node_type}' for node '{node_name}'"
            )

    # Add edges
    edges = config.get("edges", [])
    for edge_config in edges:
        source = edge_config.get("from")
        target = edge_config.get("to")
        if not source or not target:
            raise AgentConfigurationError("Edge requires both 'from' and 'to' fields")
        builder.connect(source, target)

    # Add branches (conditional edges)
    branches = config.get("branches", {})
    for source, mapping in branches.items():
        builder.branch(source, mapping)

    # Set entry and finish points
    entry = config.get("entry")
    if not entry:
        raise AgentConfigurationError("Graph requires an 'entry' node")
    builder.set_entry(entry)

    finish = config.get("finish")
    if finish:
        builder.set_finish(finish)

    return builder.build()
