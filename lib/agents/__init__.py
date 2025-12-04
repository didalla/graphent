"""Prebuilt agents for the Graphent agent framework.

This module provides ready-to-use agent configurations that can be used
directly or extended for specific use cases.

Prebuilt agents are factory functions that return configured Agent instances.
They require a model to be passed in, allowing flexibility in model choice.

Example:
    >>> from lib.agents import create_weather_agent
    >>> from langchain_openai import ChatOpenAI
    >>> model = ChatOpenAI(model="gpt-4")
    >>> weather_agent = create_weather_agent(model)
"""

# Prebuilt agents will be added here as the library grows
# Example pattern:
# from lib.agents.weather_agent import create_weather_agent

__all__ = [
    # Add prebuilt agent factory functions here
]

