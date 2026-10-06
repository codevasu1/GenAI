import os
from dotenv import load_dotenv
load_dotenv()

import requests
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage


# The graph state carries the current turn and the checkpointed conversation history.
class FlowState(BaseModel):
    question: str = Field(description="The question being asked")
    category: Literal["coding", "google_search", "weather"] = Field(default="google_search", 
                                                                    description="The category of the question")
    answer: str = Field(default="", description="The answer to the question")
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    

class QuestionCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["coding", "google_search", "weather"] = Field(description="The category of the question")


llmGroq = ChatGroq(model="openai/gpt-oss-20b")


# Specialist agents and their tools
# Google Search and weather use tool-enabled agents; coding uses the base LLM.
googleSearch = GoogleSerperAPIWrapper()
tools = [googleSearch.run]

google_agent = create_agent(
    model = llmGroq,
    tools = tools,
    system_prompt = "You are a helpful assistant that uses Google Search to answer questions."
)


# The weather agent uses this API tool and also receives the conversation history.
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

# Tool called by the weather agent to fetch current conditions.
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
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


# LangGraph node functions
# The classifier selects a specialist; each specialist returns state updates.
def check_question_category(state: FlowState):
    # Classify the latest turn with its history so short follow-ups can be understood.
    category_llm = llmGroq.with_structured_output(
        QuestionCategory,
        method="json_schema",
        strict=True,
    )
    result = category_llm.invoke([
        SystemMessage(content=(
            "Classify the latest user message as coding, google_search, or weather. "
            "Return valid json with a category field. Use earlier messages to resolve follow-ups."
        )),
        *state.messages,
    ])
    return {"category": result.category}


def route(state: FlowState) -> Literal[
    "google_search", 
    "weather", 
    "coding"]:
    return state.category


def coding_node(state: FlowState):
    print("[coding_node] message received: ", state.messages[-1].content)
    # Put the role instruction first, followed by each prior turn as a separate message.
    result = llmGroq.invoke([
        SystemMessage(content="You are a coding expert."),
        *state.messages,
    ])
    answer = result.content
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def weather_node(state: FlowState):
    print("[weather_node] message received: ", state.messages[-1].content)

    # Forward the full history so the agent can resolve follow-up questions.
    result = weather_agent.invoke({"messages": state.messages})

    answer = result["messages"][-1].content
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def google_search_node(state: FlowState):
    print("[google_search_node] message received: ", state.messages[-1].content)
    # Forward the full history so the agent can resolve follow-up questions.
    result = google_agent.invoke({"messages": state.messages})
    answer = result["messages"][-1].content

    if isinstance(answer, list):
        answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in answer
        )

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# Graph construction and checkpointing
# Checkpoints stay in this process; reusing a thread ID resumes that conversation.
memory = InMemorySaver()

graph = StateGraph(FlowState)

graph.add_node("check_question_category", check_question_category)
graph.add_node("google_search", google_search_node)
graph.add_node("weather", weather_node)
graph.add_node("coding", coding_node)


# Classify each turn, route it to one specialist node, then finish the graph.
graph.add_edge(START, "check_question_category")
graph.add_conditional_edges("check_question_category", route)
graph.add_edge("google_search", END)
graph.add_edge("weather", END)
graph.add_edge("coding", END)

# Attach the checkpointer so graph state, including messages, is saved per thread.
graph = graph.compile(checkpointer=memory)


#  chat loop
while True:
    # input() displays the prompt and typed text; avoid printing query a second time.
    query = input("user: ")
    if query.lower() in ["bye", "exit", "quit"]:
        print("Good bye")
        break

    # Keep this thread ID for turns in the same conversation.
    result = graph.invoke(
        {
            "question": query,
            "messages": [HumanMessage(content=query)],
        },
        {"configurable": {"thread_id": "5drteuykloip684099hm5op6bh5rtg54"}},
    )

    answer = result["answer"]
    print("ai: ", answer , "\n")


# sample output conversation in terminal
"""
user: hi 
[google_search_node] message received:  hi
ai:  Hello! 👋 How can I help you today? 

user: who is president of india
[google_search_node] message received:  who is president of india
ai:  The current President of India is **Droupadi Murmu**. She has been serving as the 15th President since her inauguration on July 25, 2022. 

user: and prime minister
[google_search_node] message received:  and prime minister
ai:  The current Prime Minister of India is **Narendra Modi**. He has been in office since May 26, 2014, leading the Bharatiya Janata Party (BJP) and serving as the head of government for the country. 

user: hows the weather in delhi
[weather_node] message received:  hows the weather in delhi
ai:  **Delhi - Current Weather**

- **Temperature:** 27.5 °C  
- **Condition:** Clear sky

Let me know if you'd like more details (humidity, wind, forecast, etc.)! 

user: what is the use of == operator in python
[coding_node] message received:  what is the use of == operator in python
ai:  ### The `==` operator in Python

`==` is the **equality comparison operator**.  
It checks whether the *values* of two objects are equal and returns a Boolean (`True` or `False`).

```python
a = 5
b = 5
print(a == b)          # True
```

---

## How it works

1. **Value comparison** - `==` compares the *contents* of the operands, not their identities (memory addresses).  
   ```python
   [1, 2] == [1, 2]      # True  (same contents)
   ```

2. **Method dispatch** - For user-defined classes, Python calls the object's `__eq__` method (if defined).  
   ```python
   class Point:
       def __init__(self, x, y):
           self.x, self.y = x, y
       def __eq__(self, other):
           return self.x == other.x and self.y == other.y

   p1 = Point(1, 2)
   p2 = Point(1, 2)
   print(p1 == p2)      # True
   ```

3. **Short-circuit** - In expressions like `if a == b:`, the comparison is evaluated first; the result decides the branch.

---

## Common use cases

| Context | Example |
|---------|---------|
| **Conditionals** | `if username == "admin":` |
| **Loops** | `while count == 0:` |
| **Assertions** | `assert a == b, "Values must match"` |
| **Unit tests** | `self.assertEqual(result, expected)` |
| **Dictionary key lookup** - *not* used directly, but the lookup uses `__eq__` under the hood. |

---

## Things to keep in mind

| Pitfall | What to watch out for |
|---------|-----------------------|
| **Floating-point precision** | `0.1 + 0.2 == 0.3` → `False` (use `math.isclose` instead). |
| **Identity vs. equality** | `a is b` checks if both refer to the same object; `a == b` checks value equality. |
| **Small integer caching** | `x = 256; y = 256; x == y` → `True`, but `x is y` may be `False` for larger ints. |
| **Mutable containers** | Two lists with same elements compare equal: `[1, 2] == [1, 2]`. |
| **Custom objects** | If `__eq__` isn't defined, Python falls back to identity (`is`). |

---

## Quick cheat sheet

```python
# Equality
1 == 1          # True
"foo" == "foo"  # True
[1, 2] == [1, 2]  # True

# Identity
1 is 1          # True (small ints cached)
[1, 2] is [1, 2]  # False (different objects)

# Floating-point
0.1 + 0.2 == 0.3          # False
import math
math.isclose(0.1 + 0.2, 0.3)  # True
```

---

### Bottom line

Use `==` whenever you need to **compare the values** of two objects to decide something (branch, assertion, test, etc.). If you need to know whether two variables point to the **exact same object**, use `is`. 

user: bye
Good bye
"""
