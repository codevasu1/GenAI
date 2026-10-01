from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent


# RAG pipeline
# 1. Load the source PDF into LangChain Document objects.
loader = PyPDFLoader("../data/Artificial Intelligence Overview.pdf")
docs = loader.load()


# 2. Split documents into overlapping chunks for retrieval.
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)


# 3. Set up the embedding model used to represent each chunk as vectors.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


# 4. Embed the chunks and store them in an in-memory vector store.
vector_store = InMemoryVectorStore.from_documents(
    documents=split_docs, 
    embedding=embeddings, 
    collection_name="ai_docs"
    )


# 5. Optionally check that similarity search can find relevant chunks.
same_record = vector_store.similarity_search("Gen AI")

# Agentic RAG pipeline
# 6. Expose vector search as a tool the agent can call for user questions.

@tool
def retiever_tool(query:str):
    """
    This tool can help you retrieve the relevant data of the 
    PDF document.
    """
    docs = vector_store.similarity_search(query=query, k=5)
    print(len(docs), docs[0].page_content)

    context = ""
    for doc in docs:
        context = doc.page_content + '\n\n'

    return context 


# 7. Configure the language model that will reason over the retrieved content.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# 8. Instruct the agent to use retrieval and avoid unsupported answers.
prompt = """
    You are a helpful AI assistant.
    Use the 'retiever_tool' tool to answer the user questions. 
    If you don't know the answer,
    just say that you don't know, 
    don't try to make up an answer.
    """


# 9. Assemble the agent with the model, retrieval tool, and system instructions.
agent = create_agent(
    model = llm,
    tools=[retiever_tool],
    system_prompt=prompt
)

# 10. Supply the user's question to the agent.
query = "Artificial Intelligence content"


# 11. Run the agent; it can call the retrieval tool before producing an answer.
resposne = agent.invoke(
    {"messages":[{"role":"user", "content": query}]}
)


# 12. Read and print the text from the agent's final AI message.
print(resposne["messages"][-1].content[0]["text"])


"""
Based on the provided document, here is a concise briefing on **Artificial Intelligence (AI)**:

### **Executive Summary**
Artificial Intelligence represents the simulation of human intelligence by computer systems. Modern developments are driven by deep learning networks, large language models (LLMs), and scalable compute infrastructure, enabling a rapid transition from research frameworks to critical enterprise operations.

### **Market & Growth Snapshot**
* **37.3% CAGR (2023–2030):** Global market expansion.
* **72% Enterprise Adoption:** Integrating AI in $\ge 1$ function.
* **10x Compute Scaling:** Model training power increase.

---

### **Core Paradigms**
1. **Machine Learning (Predictive):** Statistical models that improve performance through experience without explicit programming. Essential for forecasting and structured data classification.
2. **Deep Learning (Perceptual):** Multi-layered neural networks inspired by human brain structure. Powers computer vision, speech synthesis, and advanced pattern recognition.
3. **Generative AI (Synthesis):** Models capable of creating text, images, code, and audio from complex underlying distributions, led by transformer architectures.
4. **Reinforcement Learning (Autonomous):** Agents learning optimal action policies through trial, error, and reward mechanisms in complex environments like robotics or games.

---

### **Enterprise Deployment Matrix**

| Domain | Primary Applications | Impact Metric |
| :--- | :--- | :--- |
| **Healthcare** | Medical imaging analysis, drug discovery, personalized medicine | 50% faster diagnostic triaging |
| **Finance** | Algorithmic trading, fraud detection, automated compliance risk | 90%+ accuracy in fraud prevention |
| **Software Dev** | Code completion, automated unit testing, documentation generation | 35–55% developer speedup |
| **Supply Chain** | Demand forecasting, route optimization, inventory balancing | 15–20% inventory reduction |
"""