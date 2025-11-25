"""Agent module for the Graphent multi-agent framework.

This module provides the core Agent class that can process conversations,
use tools, and delegate tasks to sub-agents.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool, BaseTool

from lib.Context import Context
from lib.logging_utils import log_agent_activity

from pydantic import BaseModel, Field


class AgentHandOff(BaseModel):
    """Schema for agent hand-off tool arguments.
    
    This Pydantic model defines the structure for delegating tasks
    between agents in the multi-agent system.
    
    Attributes:
        agent_name: The name of the target agent to delegate to.
        task: A detailed description of the subtask to be performed.
    """
    agent_name: str = Field(..., description="Name of the agent to be invoked, with a subtask")
    task: str = Field(..., description="A detailed description of the subtask to be performed by the invoked agent, with all necessary information.")


class Agent:
    """An AI agent capable of using tools and delegating to sub-agents.
    
    The Agent class is the core component of the Graphent framework. It wraps
    a language model and provides capabilities for tool use and multi-agent
    collaboration through task delegation.
    
    Attributes:
        name: The display name of the agent.
        system_prompt: The system prompt that configures the agent's behavior.
        description: A description used when this agent is callable by parent agents.
        tools: List of tools available to this agent.
        callable_agents: List of sub-agents this agent can delegate tasks to.
        model: The language model (with tools bound if applicable).
    
    Example:
        >>> agent = Agent(
        ...     name="Assistant",
        ...     model=my_llm,
        ...     system_prompt="You are a helpful assistant.",
        ...     description="General purpose assistant",
        ...     tools=[search_tool, calculator_tool]
        ... )
        >>> context = Context().add_message(HumanMessage(content="Hello!"))
        >>> result = agent.invoke(context)
    """
    
    def __init__(self,
                 name: str,
                 model: BaseChatModel,
                 system_prompt: str,
                 description: str,
                 tools: Optional[list[BaseTool]] = None,
                 callable_agents: Optional[list['Agent']] = None):
        """Initialize a new Agent.
        
        Args:
            name: The display name of the agent.
            model: The LangChain chat model to use for generating responses.
            system_prompt: The system prompt that defines the agent's behavior.
            description: A description of the agent's capabilities (used by parent agents).
            tools: Optional list of tools the agent can use.
            callable_agents: Optional list of sub-agents this agent can delegate to.
        """
        self.name = name
        self.system_prompt = Agent._set_up_system_prompt(system_prompt, callable_agents)
        self.description = description
        self.callable_agents = callable_agents

        if callable_agents is not None:
            if tools is None:
                tools = []
            hand_off_tool = StructuredTool.from_function(
                func=self._hand_off_to_subagent,
                name="hand_off_to_subagent",
                description="Allows to delegate subtasks to other specialized agents",
                args_schema=AgentHandOff
            )
            self.tools = tools + [hand_off_tool]
        else:
            self.tools = tools

        if self.tools is not None:
            self.model = model.bind_tools(self.tools)
        else:
            self.model = model

    @staticmethod
    def _set_up_system_prompt(system_prompt: str, callable_agents: Optional[list['Agent']] = None) -> str:
        """Enhance the system prompt with sub-agent information.
        
        If the agent has callable sub-agents, this method appends instructions
        for delegating tasks to them.
        
        Args:
            system_prompt: The base system prompt.
            callable_agents: Optional list of sub-agents to include in the prompt.
            
        Returns:
            The enhanced system prompt with delegation instructions.
        """
        if callable_agents is not None:
            system_prompt += """
            Planning if you don't have information for a tool call, check if you can use a tool or a subagent to get the information you need.
            You can use a tool and then react to its output, or you can call a subagent to perform a specific task.
            
            # Important: You can delegate subtasks to other agents using the tool 'hand_off_to_subagent'.
            Group tasks that are for a single agent when calling other agents.
            You have access to the following agents:
            """
            for agent in callable_agents:
                system_prompt += f"- {agent.name}\n"
                system_prompt += f"  Description: {agent.description}\n"
        return system_prompt

    def _hand_off_to_subagent(self, agent_name: str, task: str) -> str:
        """Invoke a sub-agent with a delegated subtask.
        
        Used for delegating subtasks to other specialized agents in the
        multi-agent system.
        
        Args:
            agent_name: Name of the agent to be invoked.
            task: A detailed description of the subtask to be performed
                by the invoked agent, with all necessary information.
                
        Returns:
            The result of the subtask performed by the invoked agent as a string.
        """
        if self.callable_agents is None:
            return f"Agent {agent_name} is not available"

        usable_agents = [agent for agent in self.callable_agents if agent.name == agent_name]

        if len(usable_agents) == 0:
            return f"Agent {agent_name} is not available"
        elif len(usable_agents) > 1:
            return f"More than one agent named {agent_name} is available"
        else:
            result = usable_agents[0].invoke(Context().add_message(AIMessage(content=task))).get_messages(last_n=1)[0]
            content = result.content if hasattr(result, 'content') else str(result)
            # Handle case where content might be a list
            if isinstance(content, list):
                return str(content)
            return content

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"Agent: {self.name}, using model: {self.model}, with tools: {self.tools}"

    @log_agent_activity
    def invoke(self, context: Context, max_iterations: int = 10, _current_iteration: int = 0) -> Context:
        """Invoke the agent with the given context.
        
        Processes the conversation context through the language model,
        handling any tool calls recursively until a final response is generated.
        
        Args:
            context: The conversation context containing messages.
            max_iterations: Maximum number of tool-use iterations to prevent
                infinite loops. Defaults to 10.
            _current_iteration: Internal counter for tracking recursion depth.
                Do not set this manually.
                
        Returns:
            The updated context with the agent's response and any tool results.
        """
        if _current_iteration >= max_iterations:
            context.add_message(AIMessage(
                content="I've reached the maximum number of steps for this request. Please try breaking down your request into smaller parts."
            ))
            return context
            
        chat_history = [SystemMessage(content=self.system_prompt), *context.get_messages()]

        response = self.model.invoke(chat_history)
        
        # Add AI response to context (preserves tool call information)
        context.add_message(response)
        
        if not response.tool_calls:
            return context

        # Guard against None tools (shouldn't happen if tool_calls exist)
        if self.tools is None:
            return context

        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id', '')
            fitting_tools = [tool for tool in self.tools if tool.name == tool_call['name']]
            if len(fitting_tools) == 0:
                context.add_message(ToolMessage(
                    content=f"Tool {tool_call['name']} is not available",
                    tool_call_id=tool_call_id
                ))
            elif len(fitting_tools) > 1:
                context.add_message(ToolMessage(
                    content=f"More than one tool named {tool_call['name']} is available",
                    tool_call_id=tool_call_id
                ))
            else:
                tool_result = fitting_tools[0].invoke(tool_call)
                context.add_message(tool_result)

        return self.invoke(context, max_iterations, _current_iteration + 1)
