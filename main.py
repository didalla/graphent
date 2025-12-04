import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from lib import AgentBuilder, Context, AgentLoggerConfig
from lib.tools import get_coords, get_weather

AgentLoggerConfig.setup(log_file=os.environ.get("PATH_TO_FILE", None))

model = ChatOpenAI(
    model="google/gemini-2.5-flash-preview-09-2025",
    temperature=0,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

if __name__ == '__main__':
    context = Context()

    main_agent_builder = (AgentBuilder()
                          .with_name("Main Agent")
                          .with_model(model)
                          .with_system_prompt("You are qwarki a helpfull agent.")
                          .with_description("The main agent can call other agents.")
                          )

    weather_agent = (AgentBuilder()
                     .with_name("Weather Agent")
                     .with_description("Agent that can get the weather at a location.")
                     .with_model(model)
                     .with_system_prompt("Answer questions about the weather using the get_weather tool.")
                     .add_tool(get_weather)
                     .add_tool(get_coords)
                     .build())

    main_agent = (main_agent_builder
                  .add_agent(weather_agent)
                  .build())

    while True:
        user_input = input("Enter a message: ")
        if user_input == "exit":
            break
        context.add_message(HumanMessage(content=user_input))
        context = main_agent.invoke(context)
        print(context.get_messages()[-1].content)
