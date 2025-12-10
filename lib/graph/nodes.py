"""Node implementations for graph-based workflows.

This module provides three core node types:
- ActionNode: Simple single-action node with one model call
- AgentNode: Reactive agent node wrapping a full Agent
- ClassifierNode: Conditional branching based on LLM classification
"""

from typing import Optional, Generator, AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from pydantic import BaseModel, Field

from lib.Context import Context
from lib.Agent import Agent
from lib.graph.base import BaseNode
from lib.graph.hooks import GraphHookRegistry


class ActionNode(BaseNode):
    """Simple single-action node.

    Performs exactly ONE model call without any tool loop.
    Takes the input context, makes a single LLM call, and passes
    the result to the next node.

    This is ideal for simple transformations, formatting, or
    single-step processing.

    Attributes:
        name: Unique identifier for this node.
        model: The language model to use.
        system_prompt: The system prompt for this action.

    Example:
        >>> node = ActionNode(
        ...     name="formatter",
        ...     model=my_llm,
        ...     system_prompt="Format the input as bullet points"
        ... )
        >>> result = node.invoke(context)
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        system_prompt: str,
        hooks: Optional[GraphHookRegistry] = None,
    ):
        """Initialize an ActionNode.

        Args:
            name: Unique identifier for this node.
            model: The language model to use.
            system_prompt: The system prompt defining the action behavior.
            hooks: Optional hook registry for events.
        """
        super().__init__(name, hooks)
        self.model = model
        self.system_prompt = system_prompt

    def invoke(self, context: Context) -> Context:
        """Execute a single model call.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context with the model's response appended.
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]
        response = self.model.invoke(chat_history)
        context.add_message(response)
        return context

    async def ainvoke(self, context: Context) -> Context:
        """Asynchronously execute a single model call.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context with the model's response appended.
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]
        response = await self.model.ainvoke(chat_history)
        context.add_message(response)
        return context

    def stream(self, context: Context) -> Generator[str, None, Context]:
        """Stream the model's response chunk by chunk.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.

        Returns:
            The updated context with the complete response.
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]

        full_content = ""
        for chunk in self.model.stream(chat_history):
            if hasattr(chunk, "content") and chunk.content:
                full_content += chunk.content
                yield chunk.content

        response = AIMessage(content=full_content)
        context.add_message(response)
        return context

    async def astream(self, context: Context) -> AsyncGenerator[str, None]:
        """Asynchronously stream the model's response.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]

        full_content = ""
        async for chunk in self.model.astream(chat_history):
            if hasattr(chunk, "content") and chunk.content:
                full_content += chunk.content
                yield chunk.content

        response = AIMessage(content=full_content)
        context.add_message(response)


class AgentNode(BaseNode):
    """Reactive agent node wrapping a full Agent.

    Wraps an existing Agent instance, enabling full tool-calling loops
    and sub-agent delegation within the graph.

    This is ideal for complex processing steps that require
    multiple tool calls or agent collaboration.

    Attributes:
        name: Unique identifier for this node.
        agent: The wrapped Agent instance.

    Example:
        >>> agent = AgentBuilder()...build()
        >>> node = AgentNode(name="researcher", agent=agent)
        >>> result = node.invoke(context)
    """

    def __init__(
        self, name: str, agent: Agent, hooks: Optional[GraphHookRegistry] = None
    ):
        """Initialize an AgentNode.

        Args:
            name: Unique identifier for this node.
            agent: The Agent instance to wrap.
            hooks: Optional hook registry for events.
        """
        super().__init__(name, hooks)
        self.agent = agent

    def invoke(self, context: Context) -> Context:
        """Execute the wrapped agent.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context after agent processing.
        """
        return self.agent.invoke(context)

    async def ainvoke(self, context: Context) -> Context:
        """Asynchronously execute the wrapped agent.

        Args:
            context: The conversation context to process.

        Returns:
            The updated context after agent processing.
        """
        return await self.agent.ainvoke(context)

    def stream(self, context: Context) -> Generator[str, None, Context]:
        """Stream the agent's response.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.

        Returns:
            The updated context after agent processing.
        """
        yield from self.agent.stream(context)
        return context

    async def astream(self, context: Context) -> AsyncGenerator[str, None]:
        """Asynchronously stream the agent's response.

        Args:
            context: The conversation context to process.

        Yields:
            String chunks of the response.
        """
        async for chunk in self.agent.astream(context):
            yield chunk


class ClassificationResult(BaseModel):
    """Schema for classifier output.

    Attributes:
        classification: The selected classification from available options.
        confidence: Confidence score between 0 and 1.
        reasoning: Brief explanation for the classification.
    """

    classification: str = Field(
        ...,
        description="The classification category selected from the available options",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        default="", description="Brief reasoning for the classification"
    )


class ClassifierNode(BaseNode):
    """Conditional branching node based on LLM classification.

    Uses the LLM to classify input and determine which edge to follow
    in the graph. Works with any LLM by parsing text output.

    The classification is stored in the node and used by the graph
    to select the appropriate next node.

    Attributes:
        name: Unique identifier for this node.
        model: The language model to use.
        classes: List of valid classification options.
        system_prompt: Optional custom system prompt.
        last_classification: The result of the last classification.

    Example:
        >>> classifier = ClassifierNode(
        ...     name="router",
        ...     model=my_llm,
        ...     classes=["technical", "creative", "general"]
        ... )
        >>> result = classifier.invoke(context)
        >>> print(classifier.last_classification)  # e.g., "technical"
    """

    DEFAULT_SYSTEM_PROMPT = """You are a classifier. Analyze the input and classify it into exactly ONE of the following categories:
{classes}

You MUST respond with ONLY the category name, nothing else. No explanation, no punctuation, just the category name."""

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        classes: list[str],
        system_prompt: Optional[str] = None,
        hooks: Optional[GraphHookRegistry] = None,
    ):
        """Initialize a ClassifierNode.

        Args:
            name: Unique identifier for this node.
            model: The language model to use for classification.
            classes: List of valid classification categories.
            system_prompt: Optional custom system prompt (uses default if None).
            hooks: Optional hook registry for events.
        """
        super().__init__(name, hooks)
        self.classes = classes
        self.model = model
        self.last_classification: Optional[str] = None
        self.last_confidence: Optional[float] = None

        # Build system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(
                classes="\n".join(f"- {c}" for c in classes)
            )

    def _parse_classification(self, response_text: str) -> str:
        """Parse the classification from text response.

        Args:
            response_text: The raw response from the model.

        Returns:
            The classification category.
        """
        # Clean up the response
        cleaned = response_text.strip().lower()

        # Try to match against known classes (case-insensitive)
        for cls in self.classes:
            if cls.lower() in cleaned or cleaned in cls.lower():
                return cls

        # If no match, return the first class as fallback
        return self.classes[0]

    def invoke(self, context: Context) -> Context:
        """Classify the input and store the result.

        Args:
            context: The conversation context to classify.

        Returns:
            The context (with classification stored in node).
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]

        response = self.model.invoke(chat_history)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        self.last_classification = self._parse_classification(response_text)
        self.last_confidence = 1.0  # Simple classification doesn't provide confidence

        # Add a message noting the classification for context
        context.add_message(
            AIMessage(content=f"[Classification: {self.last_classification}]")
        )

        return context

    async def ainvoke(self, context: Context) -> Context:
        """Asynchronously classify the input.

        Args:
            context: The conversation context to classify.

        Returns:
            The context (with classification message added).
        """
        chat_history = [
            SystemMessage(content=self.system_prompt),
            *context.get_messages(),
        ]

        response = await self.model.ainvoke(chat_history)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        self.last_classification = self._parse_classification(response_text)
        self.last_confidence = 1.0

        # Add a message noting the classification for context
        context.add_message(
            AIMessage(content=f"[Classification: {self.last_classification}]")
        )

        return context

    def get_classification(self) -> Optional[str]:
        """Get the last classification result.

        Returns:
            The classification string, or None if not yet classified.
        """
        return self.last_classification
