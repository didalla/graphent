from pprint import pprint

from langchain_core.tools import tool
import requests
from typing import Optional

@tool
def get_coords(city: str) -> str:
    """
    Get the geographical coordinates (latitude and longitude) of a given city using OpenStreetMap's Nominatim API.

    Args:
        city (str): The name of the city to look up.

    Returns:
        str: A string containing the latitude and longitude of the city, or an error message if
                the city is not found or an error occurs.
    """
    if not city or not city.strip():
        return "Bitte gib einen gültigen Stadtnamen an."

    url = f"https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        try:
            lat = data["results"][0]["latitude"]
            lon = data["results"][0]["longitude"]
            return f"The coordinates of {city} are Latitude: {lat}, Longitude: {lon}"
        except (KeyError, IndexError, TypeError):
            return f"Can't extract coordinates from the response: {data}"
    except requests.RequestException as e:
        return f"API Error:{e}"


@tool
def get_weather(lat: float, lon:float, forecast_duration: int = 1) -> str:
    """
    Get the current weather information for given geographical coordinates using Open-Meteo API.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        forecast_duration (int): Duration of the forecast in days (max 16). When no duration is specified, the default value is 1.

    Returns:
        str: A string containing the current weather information, or an error message if
             the data cannot be retrieved.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,weathercode",
        "forecast_days": forecast_duration,
        "timezone": "Europe/Berlin"
    }

    # weather codes
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    #restructure response to obj["time"] = {temperature: x, weathercode_translation: y}

    restructured_data = {}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extract hourly data
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temperatures = hourly.get("temperature_2m", [])
        weathercodes = hourly.get("weathercode", [])

        # Restructure data
        for i, time in enumerate(times):
            weathercode = weathercodes[i] if i < len(weathercodes) else 0
            restructured_data[time] = {
                "temperature": temperatures[i] if i < len(temperatures) else None,
                "weathercode_translation": weather_codes.get(weathercode, "Unknown")
            }

        return_value = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "config": {
                "forecast_duration": forecast_duration,
                "timezone": data.get("timezone"),
                "temperature_unit": "Celsius",
            },
            "data": restructured_data
        }

        return str(return_value)

    except requests.RequestException as e:
        return f"API Error: {e}"
    except (KeyError, IndexError, TypeError) as e:
        return f"Error parsing weather data: {e}"

#pprint(get_weather(49.489, 8.466, 3))

#print(get_coords("Berlin"))






