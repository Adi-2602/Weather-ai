import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
# Load environment variables from .env file


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
search_tool = TavilySearchResults(max_results=5, api_key=TAVILY_API_KEY)
@tool 
def get_weather_data(city : str) -> str:
    """
    Fetch current weather information for a given city 
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" in data:
        return f"Could not fetch weather data for {city}. Please check the city name and try again."

    return (
        
    )
search_tool.invoke("Give me a latest news about AI and its impact on society.")
#============================================
#LLM
#=============================================

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)
response =llm.invoke("Write a short story about love.")
print(response.content)
#============================================
#Prompt
#============================================
prompt = hub.pull("hwchase17/react")
prompt
#============================================
#TOOLS
#============================================
tools = [search_tool, get_weather_data]
#tools = []
#============================================
#CREATE AGENT
#============================================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
#============================================
#Executor
#============================================
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False
)
#============================================
#Run
#============================================

response = agent_executor.invoke({
    "input": ( 
        "Find the capital of India"
        "and then find its current weather"
        )
    })
print(response["output"])