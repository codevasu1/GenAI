"""RAG Document Q&A Assistant.

Task: Upload PDF files, convert them into searchable chunks, retrieve the most
relevant content for a user question, and generate an answer using that context.
This app demonstrates a simple retrieval-augmented generation workflow with
Streamlit and LangChain.
"""

from dotenv import load_dotenv

# Load environment variables from .env so API keys are available during runtime.
load_dotenv()

# Document loader for PDFs and vector database support.
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# ------------------------------------------------------------
# Session state setup
# ------------------------------------------------------------
# These variables keep important app data across Streamlit reruns.
# They store the upload status, the working agent, vector store, and chat history.
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------
# Document processing function
# ------------------------------------------------------------
# This function handles the main RAG setup: load PDFs, split them, build
# embeddings, create a retriever, and configure the AI agent.
def processDocumnetFn(path):
    """Process uploaded PDF files and create a retrieval-based document Q&A agent."""

    # Step 1: Load all PDF files from the selected folder.
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Step 2: Break large documents into smaller text chunks so they are easier
    # to search and retrieve efficiently.
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Step 3: Create text embeddings for each document chunk using Google AI.
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # Step 4: Store the chunked documents in an in-memory vector database for
    # similarity-based retrieval when a question is asked.
    vector_store = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="ai_docs",
    )

    # Step 5: Choose the LLM that will answer questions using the retrieved context.
    # Note: ChatGoogleGenerativeAI can also be used here instead of Groq if preferred.
    llm = ChatGroq(model="openai/gpt-oss-120b")

    # Step 6: Define a custom tool that performs similarity search in the vector store.
    @tool
    def retrieve_context(query: str):
        """Retrieve the most relevant document chunks for a user query."""
        context = ""

        docs = vector_store.similarity_search(query=query, k=5)
        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    # Step 7: Set agent instructions so it answers only from document context.
    system_prompt = """You are a helpful assistant `Mr. Doc` who is a Document reader expert, that answers questions using retrieved context.
    My knowledge base consists of the details from the uploaded document. you extract information for the uploaded document.
    Always use the `retrieve_context` tool to answer the user's question when external knowledge is needed.
    If the user asks something that is not present in the uploaded document, say so clearly.
    Do not make up answers.
    """

    # Step 8: Create memory for the conversation so the agent can maintain context.
    memory = InMemorySaver()

    # Step 9: Create the LangChain agent with the retrieval tool and LLM.
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    # Save the created agent and upload status for later use in the chat UI.
    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ------------------------------------------------------------
# Upload interface
# ------------------------------------------------------------
# This section lets the user upload one or more PDF files to build the knowledge base.
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload Document", type="pdf", accept_multiple_files=True)
    st.subheader("Upload PDF files so Mr. Doc can extract information from them.")

    if uploaded:
        with st.spinner("Processing..."):
            path = "./uploaded_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            # After saving files, run the document processing workflow.
            processDocumnetFn(path)
            st.rerun()


# ------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------
# Once the document is processed, the user can ask questions about it.
if st.session_state.document_uploaded and st.session_state.agent:
    st.subheader("Mr. Doc | Document Expert")

    # Show previous chat messages saved in the session state.
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    # This is the input box where the user asks questions.
    query = st.chat_input("Ask anything related to uploaded documents")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        # Invoke the agent with the current question to get the final answer.
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "efreGrtyhrtery54y6754yh54yhe5tr54ywe542yw35e5546"}},
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})
