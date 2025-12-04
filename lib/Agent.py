"""Agent module for the Graphent multi-agent framework.

This module provides the core Agent class that can process conversations,
use tools, and delegate tasks to sub-agents.
"""

from typing import Optional, Generator, AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool, BaseTool

from lib.Context import Context
from lib.logging_utils import log_agent_activity
from lib.hooks import (
    HookRegistry, HookType, HookCallback,
    ToolCallEvent, ToolResultEvent, ResponseEvent,
    ModelCallEvent, ModelResultEvent, DelegationEvent
)

from pydantic import BaseModel, Field


# ============================================================================
# System Prompt Templates
# ============================================================================

AGENT_DELEGATION_PROMPT_HEADER = """
Planning: If you don't have information for a tool call, check if you can use a tool or a subagent to get the information you need.
You can use a tool and then react to its output, or you can call a subagent to perform a specific task.

# Important: You can delegate subtasks to other agents using the tool 'hand_off_to_subagent'.
Group tasks that are for a single agent when calling other agents.
You have access to the following agents:
"""

AGENT_DELEGATION_AGENT_TEMPLATE = "- {name}\n  Description: {description}\n"


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
                 callable_agents: Optional[list['Agent']] = None,
                 hooks: Optional[HookRegistry] = None):
        """Initialize a new Agent.

        Args:
            name: The display name of the agent.
            model: The LangChain chat model to use for generating responses.
            system_prompt: The system prompt that defines the agent's behavior.
            description: A description of the agent's capabilities (used by parent agents).
            tools: Optional list of tools the agent can use.
            callable_agents: Optional list of sub-agents this agent can delegate to.
            hooks: Optional HookRegistry for event callbacks.
        """
        self.name = name
        self.system_prompt = Agent._set_up_system_prompt(system_prompt, callable_agents)
        self.description = description
        self.callable_agents = callable_agents
        self._hooks = hooks or HookRegistry()

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
            system_prompt += AGENT_DELEGATION_PROMPT_HEADER
            for agent in callable_agents:
                system_prompt += AGENT_DELEGATION_AGENT_TEMPLATE.format(
                    name=agent.name,
                    description=agent.description
                )
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
            # Trigger delegation hook
            self._hooks.trigger(HookType.ON_DELEGATION, DelegationEvent(
                from_agent=self.name,
                to_agent=agent_name,
                task=task
            ))
            result = usable_agents[0].invoke(Context().add_message(AIMessage(content=task))).get_messages(last_n=1)[0]
            content = result.content if hasattr(result, 'content') else str(result)
            # Handle case where content might be a list
            if isinstance(content, list):
                return str(content)
            return content

    async def _ahand_off_to_subagent(self, agent_name: str, task: str) -> str:
        """Asynchronously invoke a sub-agent with a delegated subtask.

        Async variant of _hand_off_to_subagent for delegating subtasks
        to other specialized agents in the multi-agent system.

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
            # Trigger delegation hook
            await self._hooks.atrigger(HookType.ON_DELEGATION, DelegationEvent(
                from_agent=self.name,
                to_agent=agent_name,
                task=task
            ))
            result_context = await usable_agents[0].ainvoke(Context().add_message(AIMessage(content=task)))
            result = result_context.get_messages(last_n=1)[0]
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

        # Trigger before_model_call hook
        self._hooks.trigger(HookType.BEFORE_MODEL_CALL, ModelCallEvent(
            agent_name=self.name,
            message_count=len(chat_history),
            system_prompt=self.system_prompt
        ))

        response = self.model.invoke(chat_history)

        # Trigger after_model_call hook
        response_content = response.content if hasattr(response, 'content') else ''
        if isinstance(response_content, list):
            response_content = str(response_content)
        self._hooks.trigger(HookType.AFTER_MODEL_CALL, ModelResultEvent(
            agent_name=self.name,
            response_content=response_content,
            has_tool_calls=bool(response.tool_calls),
            tool_calls=response.tool_calls if response.tool_calls else []
        ))

        # Add AI response to context (preserves tool call information)
        context.add_message(response)

        self._last_response = response # For wrapper
        if not response.tool_calls:
            # Trigger on_response hook for final responses
            self._hooks.trigger(HookType.ON_RESPONSE, ResponseEvent(
                content=response_content,
                agent_name=self.name,
                has_tool_calls=False,
                tool_calls=[]
            ))
            return context

        # Guard against None tools (shouldn't happen if tool_calls exist)
        if self.tools is None:
            return context

        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id', '')
            tool_args = tool_call.get('args', {})
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
                # Trigger on_tool_call hook
                self._hooks.trigger(HookType.ON_TOOL_CALL, ToolCallEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name
                ))
                tool_result = fitting_tools[0].invoke(tool_call)
                # Trigger after_tool_call hook
                result_content = tool_result.content if hasattr(tool_result, 'content') else str(tool_result)
                self._hooks.trigger(HookType.AFTER_TOOL_CALL, ToolResultEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name,
                    result=result_content
                ))
                context.add_message(tool_result)

        return self.invoke(context, max_iterations, _current_iteration + 1)

    @log_agent_activity
    async def ainvoke(self, context: Context, max_iterations: int = 10, _current_iteration: int = 0) -> Context:
        """Asynchronously invoke the agent with the given context.

        Async variant of invoke. Processes the conversation context through
        the language model, handling any tool calls recursively until a
        final response is generated.

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

        # Trigger before_model_call hook
        await self._hooks.atrigger(HookType.BEFORE_MODEL_CALL, ModelCallEvent(
            agent_name=self.name,
            message_count=len(chat_history),
            system_prompt=self.system_prompt
        ))

        response = await self.model.ainvoke(chat_history)

        # Trigger after_model_call hook
        response_content = response.content if hasattr(response, 'content') else ''
        if isinstance(response_content, list):
            response_content = str(response_content)
        await self._hooks.atrigger(HookType.AFTER_MODEL_CALL, ModelResultEvent(
            agent_name=self.name,
            response_content=response_content,
            has_tool_calls=bool(response.tool_calls),
            tool_calls=response.tool_calls if response.tool_calls else []
        ))

        # Add AI response to context (preserves tool call information)
        context.add_message(response)

        self._last_response = response  # For wrapper
        if not response.tool_calls:
            # Trigger on_response hook for final responses
            await self._hooks.atrigger(HookType.ON_RESPONSE, ResponseEvent(
                content=response_content,
                agent_name=self.name,
                has_tool_calls=False,
                tool_calls=[]
            ))
            return context

        # Guard against None tools (shouldn't happen if tool_calls exist)
        if self.tools is None:
            return context

        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id', '')
            tool_args = tool_call.get('args', {})
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
                tool = fitting_tools[0]
                # Trigger on_tool_call hook
                await self._hooks.atrigger(HookType.ON_TOOL_CALL, ToolCallEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name
                ))
                # Check if tool has async invoke method and use it
                if hasattr(tool, 'ainvoke'):
                    tool_result = await tool.ainvoke(tool_call)
                else:
                    tool_result = tool.invoke(tool_call)
                # Trigger after_tool_call hook
                result_content = tool_result.content if hasattr(tool_result, 'content') else str(tool_result)
                await self._hooks.atrigger(HookType.AFTER_TOOL_CALL, ToolResultEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name,
                    result=result_content
                ))
                context.add_message(tool_result)

        return await self.ainvoke(context, max_iterations, _current_iteration + 1)

    @log_agent_activity
    def stream(self, context: Context, max_iterations: int = 10, _current_iteration: int = 0) -> Generator[str, None, Context]:
        """Stream the agent's response chunk by chunk.

        Processes the conversation context through the language model using
        streaming, yielding response chunks as they become available. Handles
        tool calls recursively like invoke().

        Args:
            context: The conversation context containing messages.
            max_iterations: Maximum number of tool-use iterations to prevent
                infinite loops. Defaults to 10.
            _current_iteration: Internal counter for tracking recursion depth.
                Do not set this manually.

        Yields:
            String chunks of the response as they are generated.

        Returns:
            The updated context with the agent's response and any tool results.

        Example:
            >>> context = Context().add_message(HumanMessage(content="Hello"))
            >>> for chunk in agent.stream(context):
            ...     print(chunk, end="", flush=True)
        """
        if _current_iteration >= max_iterations:
            max_iter_message = "I've reached the maximum number of steps for this request. Please try breaking down your request into smaller parts."
            context.add_message(AIMessage(content=max_iter_message))
            yield max_iter_message
            return context

        chat_history = [SystemMessage(content=self.system_prompt), *context.get_messages()]

        # Trigger before_model_call hook
        self._hooks.trigger(HookType.BEFORE_MODEL_CALL, ModelCallEvent(
            agent_name=self.name,
            message_count=len(chat_history),
            system_prompt=self.system_prompt
        ))

        # Collect the full response while streaming
        full_content = ""
        tool_calls = []

        for chunk in self.model.stream(chat_history):
            # Accumulate content
            if hasattr(chunk, 'content') and chunk.content:
                full_content += chunk.content
                yield chunk.content

            # Accumulate tool calls from chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_call_chunk in chunk.tool_call_chunks:
                    # Find or create tool call entry
                    tc_index = tool_call_chunk.get('index', 0)
                    while len(tool_calls) <= tc_index:
                        tool_calls.append({'id': '', 'name': '', 'args': ''})
                    
                    if tool_call_chunk.get('id'):
                        tool_calls[tc_index]['id'] = tool_call_chunk['id']
                    if tool_call_chunk.get('name'):
                        tool_calls[tc_index]['name'] = tool_call_chunk['name']
                    if tool_call_chunk.get('args'):
                        tool_calls[tc_index]['args'] += tool_call_chunk['args']

        # Build the complete response message
        response = AIMessage(content=full_content)
        
        # Parse accumulated tool calls if any
        if tool_calls and tool_calls[0].get('name'):
            import json
            parsed_tool_calls = []
            for tc in tool_calls:
                if tc.get('name'):
                    try:
                        args = json.loads(tc['args']) if tc['args'] else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_tool_calls.append({
                        'id': tc['id'],
                        'name': tc['name'],
                        'args': args
                    })
            response.tool_calls = parsed_tool_calls

        # Add AI response to context
        context.add_message(response)
        self._last_response = response

        # Trigger after_model_call hook
        self._hooks.trigger(HookType.AFTER_MODEL_CALL, ModelResultEvent(
            agent_name=self.name,
            response_content=full_content,
            has_tool_calls=bool(response.tool_calls),
            tool_calls=response.tool_calls if response.tool_calls else []
        ))

        if not response.tool_calls:
            # Trigger on_response hook for final responses
            self._hooks.trigger(HookType.ON_RESPONSE, ResponseEvent(
                content=full_content,
                agent_name=self.name,
                has_tool_calls=False,
                tool_calls=[]
            ))
            return context

        # Guard against None tools
        if self.tools is None:
            return context

        # Process tool calls
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id', '')
            tool_args = tool_call.get('args', {})
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
                # Trigger on_tool_call hook
                self._hooks.trigger(HookType.ON_TOOL_CALL, ToolCallEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name
                ))
                tool_result = fitting_tools[0].invoke(tool_call)
                # Trigger after_tool_call hook
                result_content = tool_result.content if hasattr(tool_result, 'content') else str(tool_result)
                self._hooks.trigger(HookType.AFTER_TOOL_CALL, ToolResultEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name,
                    result=result_content
                ))
                context.add_message(tool_result)

        # Recursively continue streaming after tool calls
        yield from self.stream(context, max_iterations, _current_iteration + 1)
        return context

    @log_agent_activity
    async def astream(self, context: Context, max_iterations: int = 10, _current_iteration: int = 0) -> AsyncGenerator[str, None]:
        """Asynchronously stream the agent's response chunk by chunk.

        Async variant of stream(). Processes the conversation context through
        the language model using async streaming, yielding response chunks
        as they become available.

        Args:
            context: The conversation context containing messages.
            max_iterations: Maximum number of tool-use iterations to prevent
                infinite loops. Defaults to 10.
            _current_iteration: Internal counter for tracking recursion depth.
                Do not set this manually.

        Yields:
            String chunks of the response as they are generated.

        Example:
            >>> context = Context().add_message(HumanMessage(content="Hello"))
            >>> async for chunk in agent.astream(context):
            ...     print(chunk, end="", flush=True)
        """
        if _current_iteration >= max_iterations:
            max_iter_message = "I've reached the maximum number of steps for this request. Please try breaking down your request into smaller parts."
            context.add_message(AIMessage(content=max_iter_message))
            yield max_iter_message
            return

        chat_history = [SystemMessage(content=self.system_prompt), *context.get_messages()]

        # Trigger before_model_call hook
        await self._hooks.atrigger(HookType.BEFORE_MODEL_CALL, ModelCallEvent(
            agent_name=self.name,
            message_count=len(chat_history),
            system_prompt=self.system_prompt
        ))

        # Collect the full response while streaming
        full_content = ""
        tool_calls = []

        async for chunk in self.model.astream(chat_history):
            # Accumulate content
            if hasattr(chunk, 'content') and chunk.content:
                full_content += chunk.content
                yield chunk.content

            # Accumulate tool calls from chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_call_chunk in chunk.tool_call_chunks:
                    # Find or create tool call entry
                    tc_index = tool_call_chunk.get('index', 0)
                    while len(tool_calls) <= tc_index:
                        tool_calls.append({'id': '', 'name': '', 'args': ''})
                    
                    if tool_call_chunk.get('id'):
                        tool_calls[tc_index]['id'] = tool_call_chunk['id']
                    if tool_call_chunk.get('name'):
                        tool_calls[tc_index]['name'] = tool_call_chunk['name']
                    if tool_call_chunk.get('args'):
                        tool_calls[tc_index]['args'] += tool_call_chunk['args']

        # Build the complete response message
        response = AIMessage(content=full_content)
        
        # Parse accumulated tool calls if any
        if tool_calls and tool_calls[0].get('name'):
            import json
            parsed_tool_calls = []
            for tc in tool_calls:
                if tc.get('name'):
                    try:
                        args = json.loads(tc['args']) if tc['args'] else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_tool_calls.append({
                        'id': tc['id'],
                        'name': tc['name'],
                        'args': args
                    })
            response.tool_calls = parsed_tool_calls

        # Add AI response to context
        context.add_message(response)
        self._last_response = response

        # Trigger after_model_call hook
        await self._hooks.atrigger(HookType.AFTER_MODEL_CALL, ModelResultEvent(
            agent_name=self.name,
            response_content=full_content,
            has_tool_calls=bool(response.tool_calls),
            tool_calls=response.tool_calls if response.tool_calls else []
        ))

        if not response.tool_calls:
            # Trigger on_response hook for final responses
            await self._hooks.atrigger(HookType.ON_RESPONSE, ResponseEvent(
                content=full_content,
                agent_name=self.name,
                has_tool_calls=False,
                tool_calls=[]
            ))
            return

        # Guard against None tools
        if self.tools is None:
            return

        # Process tool calls
        for tool_call in response.tool_calls:
            tool_call_id = tool_call.get('id', '')
            tool_args = tool_call.get('args', {})
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
                tool = fitting_tools[0]
                # Trigger on_tool_call hook
                await self._hooks.atrigger(HookType.ON_TOOL_CALL, ToolCallEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name
                ))
                if hasattr(tool, 'ainvoke'):
                    tool_result = await tool.ainvoke(tool_call)
                else:
                    tool_result = tool.invoke(tool_call)
                # Trigger after_tool_call hook
                result_content = tool_result.content if hasattr(tool_result, 'content') else str(tool_result)
                await self._hooks.atrigger(HookType.AFTER_TOOL_CALL, ToolResultEvent(
                    tool_name=tool_call['name'],
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    agent_name=self.name,
                    result=result_content
                ))
                context.add_message(tool_result)

        # Recursively continue streaming after tool calls
        async for chunk in self.astream(context, max_iterations, _current_iteration + 1):
            yield chunk
