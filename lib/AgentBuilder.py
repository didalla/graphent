from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from lib.Agent import Agent


class AgentBuilder:
    def __init__(self):
        self._name: str | None = None
        self._model: BaseChatModel | None = None
        self._system_prompt: str | None = None
        self._description: str | None = None
        self._tools: list[BaseTool] = []
        self._callable_agents: list[Agent] = []

    def with_name(self, name: str) -> 'AgentBuilder':
        self._name = name
        return self

    def with_model(self, model: BaseChatModel) -> 'AgentBuilder':
        self._model = model
        return self

    def with_system_prompt(self, system_prompt: str) -> 'AgentBuilder':
        self._system_prompt = system_prompt
        return self

    def with_description(self, description: str) -> 'AgentBuilder':
        self._description = description
        return self

    def add_tool(self, tool: BaseTool) -> 'AgentBuilder':
        self._tools.append(tool)
        return self

    def add_tools(self, tools: list[BaseTool]) -> 'AgentBuilder':
        self._tools.extend(tools)
        return self

    def add_agent(self, agent: 'Agent') -> 'AgentBuilder':
        self._callable_agents.append(agent)
        return self

    def add_agents(self, agents: list['Agent']) -> 'AgentBuilder':
        self._callable_agents.extend(agents)
        return self

    def build(self) -> 'Agent':
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