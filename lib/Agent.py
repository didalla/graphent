from langchain_core.language_models import BaseChatModel
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool, BaseTool

from lib.Context import Context
from lib.logging_utils import log_agent_activity

from pydantic import BaseModel, Field

class AgentHandOf(BaseModel):
    agent_name: str = Field(..., description="Name of the agent to be invoked, with a subtask")
    task: str = Field(..., description="A detailed description of the subtask to be performed by the invoked agent, with all necessary information.")

class Agent:
    def __init__(self,
                 name: str,
                 model: BaseChatModel,
                 system_prompt: str,
                 description: str,
                 tools: list[BaseTool] = None,
                 callable_agents: list['Agent'] = None):
        self.name = name
        self.system_prompt = Agent._set_up_system_prompt(system_prompt, callable_agents)
        self.description = description
        self.callable_agents = callable_agents

        if callable_agents is not None:
            if tools is None:
                tools = []
            hand_of_tool = StructuredTool.from_function(
                func=self._hand_of_to_subagent,
                name="hand_of_to_subagent",
                description="Allows to delegate subtasks to other specialized agents",
                args_schema=AgentHandOf
            )
            self.tools = tools + [hand_of_tool]
        else:
            self.tools = tools

        if self.tools is not None:
            self.model = model.bind_tools(self.tools)
        else:
            self.model = model

    @staticmethod
    def _set_up_system_prompt(system_prompt: str, callable_agents: list['Agent'] = None):
        if callable_agents is not None:
            system_prompt += """
            Planning if you don't have information for a tool call, check if you can use a tool or a subagent to get the information you need.
            You can use a tool and then react to its output, or you can call a subagent to perform a specific task.
            
            # Important: You can delegate subtasks to other agents using the tool 'hand_of_to_subagent'.
            Group tasks that are for a single agent when calling other agents.
            You have access to the following agents:
            """
            for agent in callable_agents:
                system_prompt += f"- {agent.name}\n"
                system_prompt += f"  Description: {agent.description}\n"
        return system_prompt

    def _hand_of_to_subagent(self, agent_name: str, task: str) -> BaseMessage:
        """
        Invokes an agent with a subtask. Used for delegating subtasks to other specialized agents.

        :param agent_name: Name of the agent to be invoked, with a subtask
        :param task: A detailed description of the subtask to be performed by the invoked agent, with all necessary information.
        :return: The result of the subtask performed by the invoked agent.
        """
        if self.callable_agents is None:
            return ToolMessage(content=f"Agent {agent_name} is not available")

        usable_agents = [agent for agent in self.callable_agents if agent.name == agent_name]

        if len(usable_agents) == 0:
            return ToolMessage(content=f"Agent {agent_name} is not available")
        elif len(usable_agents) > 1:
            return ToolMessage(content=f"More than one agent named {agent_name} is available")
        else:
            return usable_agents[0].invoke(Context().add_message(AIMessage(content=task))).get_messages(last_n=1)[0]

    def __str__(self):
        return f"Agent: {self.name}, using model: {self.model}, with tools: {self.tools}"

    @log_agent_activity
    def invoke(self, context: Context) -> Context:
        chat_history = [SystemMessage(content=self.system_prompt), *context.get_messages()]

        response = self.model.invoke(chat_history)
        if not response.tool_calls:
            return Context().add_message(AIMessage(content=response.content))

        for tool_call in response.tool_calls:
            fitting_tools = [tool for tool in self.tools if tool.name == tool_call['name']]
            if len(fitting_tools) == 0:
                context.add_message(ToolMessage(content=f"Tool {tool_call['name']} is not available"))
            elif len(fitting_tools) > 1:
                context.add_message(
                    ToolMessage(content=f"More than one tool named {tool_call['name']} is available"))
            else:
                tool_result = fitting_tools[0].invoke(tool_call)
                context.add_message(tool_result)

        return self.invoke(context)
