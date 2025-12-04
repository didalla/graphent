"""Prebuilt tools for the Graphent agent framework.

This module provides ready-to-use tools that can be added to agents.
These tools demonstrate common patterns and provide useful functionality.

Example:
    >>> from lib.tools import get_coords, get_weather
    >>> agent = (AgentBuilder()
    ...     .with_name("Weather Agent")
    ...     .add_tool(get_coords)
    ...     .add_tool(get_weather)
    ...     .build())
"""

from lib.tools.map_tool import get_coords, get_weather

__all__ = [
    "get_coords",
    "get_weather",
]

