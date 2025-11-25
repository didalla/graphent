"""AgentBuilder module for fluent Agent construction.

This module provides a builder pattern implementation for creating
Agent instances with a clean, fluent API.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from lib.Agent import Agent


class AgentBuilder:
    """A fluent builder for constructing Agent instances.
    
    The AgentBuilder provides a clean, chainable API for configuring
    and creating Agent objects. All configuration methods return self
    to enable method chaining.
    
    Example:
        >>> agent = (AgentBuilder()
        ...     .with_name("My Agent")
        ...     .with_model(my_llm)
        ...     .with_system_prompt("You are helpful.")
        ...     .with_description("A helpful agent")
        ...     .add_tool(my_tool)
        ...     .build())
    
    Attributes:
        _name: The agent's name (required).
        _model: The language model (required).
        _system_prompt: The system prompt (required).
        _description: The agent description (required).
        _tools: List of tools for the agent.
        _callable_agents: List of sub-agents for delegation.
    """
    
    def __init__(self):
        """Initialize a new AgentBuilder with default values."""
        self._name: str | None = None
        self._model: BaseChatModel | None = None
        self._system_prompt: str | None = None
        self._description: str | None = None
        self._tools: list[BaseTool] = []
        self._callable_agents: list[Agent] = []

    def with_name(self, name: str) -> 'AgentBuilder':
        """Set the agent's name.
        
        Args:
            name: The display name for the agent.
            
        Returns:
            Self for method chaining.
        """
        self._name = name
        return self

    def with_model(self, model: BaseChatModel) -> 'AgentBuilder':
        """Set the language model for the agent.
        
        Args:
            model: A LangChain BaseChatModel instance.
            
        Returns:
            Self for method chaining.
        """
        self._model = model
        return self

    def with_system_prompt(self, system_prompt: str) -> 'AgentBuilder':
        """Set the system prompt for the agent.
        
        Args:
            system_prompt: The system prompt defining agent behavior.
            
        Returns:
            Self for method chaining.
        """
        self._system_prompt = system_prompt
        return self

    def with_description(self, description: str) -> 'AgentBuilder':
        """Set the agent's description.
        
        The description is used by parent agents when this agent
        is available as a callable sub-agent.
        
        Args:
            description: A description of the agent's capabilities.
            
        Returns:
            Self for method chaining.
        """
        self._description = description
        return self

    def add_tool(self, tool: BaseTool) -> 'AgentBuilder':
        """Add a single tool to the agent.
        
        Args:
            tool: A LangChain BaseTool instance.
            
        Returns:
            Self for method chaining.
        """
        self._tools.append(tool)
        return self

    def add_tools(self, tools: list[BaseTool]) -> 'AgentBuilder':
        """Add multiple tools to the agent.
        
        Args:
            tools: A list of LangChain BaseTool instances.
            
        Returns:
            Self for method chaining.
        """
        self._tools.extend(tools)
        return self

    def add_agent(self, agent: 'Agent') -> 'AgentBuilder':
        """Add a sub-agent that this agent can delegate to.
        
        Args:
            agent: An Agent instance to add as a callable sub-agent.
            
        Returns:
            Self for method chaining.
        """
        self._callable_agents.append(agent)
        return self

    def add_agents(self, agents: list['Agent']) -> 'AgentBuilder':
        """Add multiple sub-agents that this agent can delegate to.
        
        Args:
            agents: A list of Agent instances.
            
        Returns:
            Self for method chaining.
        """
        self._callable_agents.extend(agents)
        return self

    def build(self) -> 'Agent':
        """Build and return the configured Agent.
        
        Returns:
            A new Agent instance with the configured settings.
            
        Raises:
            ValueError: If any required field (name, model, system_prompt,
                description) is not set.
        """
        if self._name is None:
            raise ValueError("Agent name is required")
        if self._model is None:
            raise ValueError("Agent model is required")
        if self._system_prompt is None:
            raise ValueError("Agent system prompt is required")
        if self._description is None:
            raise ValueError("Agent description is required")

        return Agent(
            name=self._name,
            model=self._model,
            system_prompt=self._system_prompt,
            description=self._description,
            tools=self._tools if self._tools else None,
            callable_agents=self._callable_agents if self._callable_agents else None
        )