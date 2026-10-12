
import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_classic import hub
from langchain_core.prompts import PromptTemplate   # ✅
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.caption("Ask questions, search the web, and check current weather.")

# ==========================================
# CHECK API KEYS
# ==========================================

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing from your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing from your .env file.")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.error("WEATHERSTACK_API_KEY is missing from your .env file.")
    st.stop()


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """Get current weather, temperature, humidity and wind
    information for a given city."""
    try:
        url = "https://api.weatherstack.com/current"

        response = requests.get(
            url,
            params={
                "access_key": WEATHERSTACK_API_KEY,
                "query": city
            },
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            error = data.get("error", {}).get(
                "info", "Unknown weather API error."
            )
            return f"Could not fetch weather for {city}: {error}"

        current = data["current"]
        location = data["location"]

        description = ", ".join(
            current.get("weather_descriptions", [])
        )

        return (
            f"Weather in {location.get('name', city)}, "
            f"{location.get('country', '')}:\n"
            f"Condition: {description}\n"
            f"Temperature: {current.get('temperature')} °C\n"
            f"Feels like: {current.get('feelslike')} °C\n"
            f"Humidity: {current.get('humidity')}%\n"
            f"Wind speed: {current.get('wind_speed')} km/h"
        )

    except requests.RequestException as e:
        return f"Weather service request failed: {e}"


# ==========================================
# CREATE AI AGENT
# ==========================================
REACT_PROMPT = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

@st.cache_resource
def create_agent():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    search_tool = TavilySearchResults(
        max_results=5,
        api_key=TAVILY_API_KEY
    )

    tools = [search_tool, get_weather_data]

    prompt = REACT_PROMPT 

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6
    )

    return executor


# ==========================================
# CHAT INTERFACE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Suggested questions
with st.sidebar:
    st.header("💡 Try asking")
    st.markdown("""
    - What is the capital of India?
    - Find the latest AI news.
    - What is the current weather in Chennai?
    - Find the weather in Delhi and explain it.
    """)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# User input
user_input = st.chat_input(
    "Ask about news, weather, or anything else..."
)

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Agent is thinking..."):
                agent_executor = create_agent()

                result = agent_executor.invoke({
                    "input": user_input
                })

                answer = result["output"]

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")
