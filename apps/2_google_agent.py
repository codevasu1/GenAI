from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

#added memory increases tokens usage as added history messages is passed to llm
llmGroq = ChatGroq(model="openai/gpt-oss-120b")
GoogleSearch = GoogleSerperAPIWrapper()

memory = MemorySaver()

agent = create_agent(
    model = llmGroq,
    tools = [GoogleSearch.run],# notice: no colons after .run, we dont want to run it now
    checkpointer= memory,
    system_prompt="You are a tool for searching the internet. you can search any question on google search."
)


while True:
    query = input("User: ")
    if query.lower() == 'bye':
        print("Good bye")
        break

    response = agent.invoke({'messages':[{'role':'me: ', 'content':query}]},
                            {"configurable":{"thread_id":"asd45"}})
    
    print("AI: ",response["messages"][-1].content)