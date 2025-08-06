import requests
import json
import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API = os.getenv("GROQ_API")

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API,
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=2000
)

# Define the weather tool using LangChain's @tool decorator
@tool
def get_weather(lat: float, lon: float) -> str:
    """Get current weather and forecast for the next 5 hours based on latitude and longitude of provided city.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
    
    Returns:
        JSON string containing current weather and 5-hour forecast
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current_weather=true"
            "&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,rain"
            "&timezone=auto"
        )
        response = requests.get(url)
        data = response.json()
        current_weather = data.get("current_weather", {})
        hourly_data = data.get("hourly", {})

        if not current_weather:
            return "No current weather data available. Cannot provide weather information."
        
        result = {
            "current_weather": {
                "temperature": current_weather.get("temperature", "N/A"),
                "wind_speed": current_weather.get("windspeed", "N/A"),
                "time": current_weather.get("time", "N/A"),
            },
            "next_5_hours": [
                {
                    "time": hourly_data.get("time", [])[i] if i < len(hourly_data.get("time", [])) else "N/A",
                    "temperature": hourly_data.get("temperature_2m", [])[i] if i < len(hourly_data.get("temperature_2m", [])) else "N/A",
                    "apparent_temperature": hourly_data.get("apparent_temperature", [])[i] if i < len(hourly_data.get("apparent_temperature", [])) else "N/A",
                    "relative_humidity": hourly_data.get("relative_humidity_2m", [])[i] if i < len(hourly_data.get("relative_humidity_2m", [])) else "N/A",
                    "wind_speed": hourly_data.get("wind_speed_10m", [])[i] if i < len(hourly_data.get("wind_speed_10m", [])) else "N/A",
                    "rain": hourly_data.get("rain", [])[i] if i < len(hourly_data.get("rain", [])) else "N/A",
                }
                for i in range(5)
            ]
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Error fetching weather data: {str(e)}"})

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides helpful answers to user queries."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create list of tools
tools = [get_weather]

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # Set to True to see the agent's reasoning process
    handle_parsing_errors=True
)

def run_agent(user_prompt: str) -> str:
    """Run the agent with the given user prompt."""
    try:
        result = agent_executor.invoke({"input": user_prompt})
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"

# Alternative implementation using memory for conversation history
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

def run_agent_with_memory(user_prompt: str) -> str:
    """Run agent with conversation memory."""
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create agent with memory
    agent_with_memory = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    try:
        result = agent_with_memory.run(user_prompt)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Choose implementation:")
    print("1. Basic agent")
    print("2. Agent with memory")
    
    choice = input("Enter choice (1 or 2): ").strip()
    user_query = input("Ask me anything: ")
    
    if choice == "2":
        answer = run_agent_with_memory(user_query)
    else:
        answer = run_agent(user_query)
    
    print("Final LLM answer:", answer)

# Example usage:
# python weather_agent_langchain.py
# Ask: "What's the weather like in Kathmandu?" (lat: 27.6992, lon: 85.3567)