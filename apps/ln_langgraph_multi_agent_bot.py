import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
from langchain_core.messages import HumanMessage, AIMessage


# coding , google_search, weather

# Stat
class FlowState(BaseModel):
    question: str = Field(description="The question being asked")
    category: Literal["coding", "google_search", "weather"] = Field(default="google_search", 
                                                                    description="The category of the question")
    answer: str = Field(default="", description="The answer to the question")
    
class QuestionCategory(BaseModel):
    category: Literal["coding", "google_search","weather"] = Field(default="google_search", description="The category of the question")


llmGroq = ChatGroq(model="openai/gpt-oss-20b")



# google search agent
googleSearch = GoogleSerperAPIWrapper()
tools = [googleSearch.run]

google_agent = create_agent(
    model = llmGroq,
    tools = tools,
    system_prompt = "You are a helpful assistant that uses Google Search to answer questions."
)


# weather agent
weather_api_key = os.getenv("OPENWEATHER_API_KEY")
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Insert your weather API call here (e.g., OpenWeatherMap)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric"
    response = requests.get(url).json()
    temp = response['main']['temp']
    desc = response['weather'][0]['description']
    return f"The temperature in {city} is {temp}°C with {desc}."

weather_agent = create_agent(
    model = llmGroq,
    tools = [get_weather],
    system_prompt = "You are a helpful assistant that provides current weather information for a given city."
)


def check_question_category(state: FlowState) -> FlowState:
    category_llm= llmGroq.with_structured_output(QuestionCategory, method="json_mode")
    result = category_llm.invoke(
        f"Classify this question and return the result as a JSON object with a category field. "
        f"Choose one category: coding, google_search, or weather. Question: {state.question}"
    )
    state.category = result.category
    return state


flow = FlowState(question="What is the temperature of Pune?")
check_question_category(flow)
print(flow)


def route(state: FlowState) -> Literal[
    "google_search", 
    "weather", 
    "coding"]:
    return state.category


def coding_node(state: FlowState) -> FlowState:
    res = llmGroq.invoke(f"You are a coding expert: {state.question}")
    state.answer = res
    return state


def weather_node(state: FlowState) -> FlowState:
    # print("Question received by weather node:", repr(state.question))

    result = weather_agent.invoke({
        "messages": [HumanMessage(content=state.question)] 
    })

    last_message = result["messages"][-1]
    state.answer = last_message.content
    return state


def google_search_node(state: FlowState) -> FlowState:
    res = google_agent.invoke({"messages": [{"role": "user", "content": state.question}]})
    last = res["messages"][-1]          # AIMessage
    content = last.content
    # Some models (e.g. Gemini) return content as a list of parts
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    state.answer = content
    return state

memory = InMemorySaver()


graph = StateGraph(FlowState)

graph.add_node("check_question_category", check_question_category)
graph.add_node("google_search", google_search_node)
graph.add_node("weather", weather_node)
graph.add_node("coding", coding_node)


# Only use string names for nodes. LangGraph expects strings for node references.
graph.add_edge(START, "check_question_category") # not! (START, check_question_category)
graph.add_conditional_edges("check_question_category", route)
graph.add_edge("google_search", END)
graph.add_edge("weather", END)
graph.add_edge("coding", END)

graph = graph.compile(memory_saver=memory)

res = graph.invoke({"question": "what is weather in new york"})
print(res,"\n\n")
print(res["answer"])














