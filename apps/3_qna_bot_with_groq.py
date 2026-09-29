from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st


llmGroq = ChatGroq(model="openai/gpt-oss-120b", streaming=True)
GoogleSearch = GoogleSerperAPIWrapper()

tools = [GoogleSearch.run]

# so same memory object will be present in memory insteadof re-creating memmory  obj again and loosing memory
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

 
agent = create_agent(
    model = llmGroq,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="you are an ai agent, you can search the internet via google search."
)


#### Building web interface
st.subheader("🪿Quick-Quack - Get fast answer with our AI Agent😊")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content )

user_query = st.chat_input("Ask Anything..")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.history.append({"role":"user","content":user_query})

    ai_response = agent.stream( 
        {"messages":[{"role":"user","content":user_query}]},
        {"configurable":{"thread_id":"qnabot1231423reERfrgDEF"}},
        stream_mode="messages"
    )

    ai_ctnr = st.chat_message("ai")
    with ai_ctnr:
        space = st.empty()

        message = "" #extracting complete msg from stream

        for chunk in ai_response:
            message = message + chunk[0].content
            space.markdown(message)

        st.session_state.history.append({"role":"ai","content":message})








