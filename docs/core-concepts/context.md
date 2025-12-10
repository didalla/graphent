# Context

The `Context` class manages conversation state and message history. It provides a clean interface for adding, retrieving, and manipulating messages in a conversation.

## Overview

A Context:

- Holds the conversation message history
- Provides methods to add and retrieve messages
- Is passed to agents for invocation
- Is updated with responses during processing

## Creating a Context

```python
from langchain_core.messages import HumanMessage, SystemMessage
from lib import Context

# Create an empty context
context = Context()

# Add messages
context = context.add_message(HumanMessage(content="Hello!"))
```

## Adding Messages

The `add_message` method returns a new Context with the message added:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

context = Context()

# Add different message types
context = context.add_message(SystemMessage(content="You are helpful."))
context = context.add_message(HumanMessage(content="What is 2+2?"))
context = context.add_message(AIMessage(content="2+2 equals 4."))
```

### Method Chaining

You can chain multiple `add_message` calls:

```python
context = (Context()
    .add_message(SystemMessage(content="You are helpful."))
    .add_message(HumanMessage(content="Hello!")))
```

## Retrieving Messages

### Get All Messages

```python
messages = context.get_messages()

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

### Get the Last Message

```python
last_message = context.get_messages()[-1]
print(last_message.content)
```

## Using Context with Agents

```python
from langchain_openai import ChatOpenAI
from lib import AgentBuilder, Context

agent = (AgentBuilder()
    .with_name("Assistant")
    .with_model(ChatOpenAI(model="gpt-4"))
    .with_system_prompt("You are helpful.")
    .with_description("An assistant")
    .build())

# Create context with user message
context = Context().add_message(HumanMessage(content="Hello!"))

# Invoke agent - returns updated context
result = agent.invoke(context)

# The result context now contains the agent's response
all_messages = result.get_messages()
```

## Multi-Turn Conversations

Context naturally supports multi-turn conversations:

```python
context = Context()

# First turn
context = context.add_message(HumanMessage(content="What is Python?"))
context = agent.invoke(context)

# Second turn (context includes previous exchange)
context = context.add_message(HumanMessage(content="What are its main features?"))
context = agent.invoke(context)

# Context now contains the full conversation history
```

For complete API documentation, see the [API Reference](../reference/index.md#context).
