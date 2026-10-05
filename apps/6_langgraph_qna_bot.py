from dotenv import load_dotenv
load_dotenv()

from typing import Annotated

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# 1) State definition
class ChatState(BaseModel):
    """Tracks the conversation history for a single chat thread."""

    messages: Annotated[list, add_messages]


# 2) LLM setup
llm_groq = ChatGroq(model="openai/gpt-oss-120b")


# 3) Graph node
def chat_bot_node(state: ChatState) -> ChatState:
    """Send the current conversation to the model and store the reply."""

    response = llm_groq.invoke(state.messages)

    # Keep the model response as the latest message in the state.
    state.messages = [response]
    return state


# 4) Build LangGraph workflow
memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatbot", chat_bot_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile the graph with memory so the conversation can be resumed later.
graph = graph.compile(checkpointer=memory)


# 5) Chat loop
while True:
    query = input("user: ")

    if query.lower() in ["bye", "exit", "quit"]:
        print("Good bye")
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "5r34tejg5gy4jg4099hm5op6bh5rtg54"}},
    )

    answer = result["messages"][-1].content
    print("ai:", answer)


# Example output:
# user: hi
# ai: Hello! How can I assist you today?
# user: where is effile tower located
# ai: The Eiffel Tower is located in Paris, France.
# user: who is their president
# ai: As of 2026, the President of France is Emmanuel Macron.
# user: exit
# Good bye
