from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma


# 1.load Data
loader = PyPDFLoader("../data/Artificial Intelligence Overview.pdf")
docs = loader.load()

# 2.split Data
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# 3. Generate Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 4. Store Embeddings in Vector DB
vector_store = Chroma.from_documents(
    documents=split_docs, 
    embedding=embeddings, 
    collection_name="ai_docs",
    persist_directory="../data/chroma_db_ai"
    )

query  = "Artificial Intelligence and generative AI content."

# if our doc is large we only get similar data for our llm : reducing tokens usage
data = vector_store.similarity_search(query=query , k = 10)

# 5. Context to be provide to llm based on query
context = ""
for doc in data:
    context += doc.page_content + "\n"

# 6. Deciding Model for our project
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# in Langchain: get/generate context -> prompt -> llm -> output

def get_context_from_query(query, vector_store, k=5):#max output set to 5 docs
    data = vector_store.similarity_search(query=query , k=k)
    context = ""
    for doc in data:
        context += doc.page_content + "\n"
    return {
        "context": context,
        "question": query
    } 

prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant.
    Use the following context to answer the question at the end. 
    If you don't know the answer,
    just say that you don't know, 
    don't try to make up an answer.
    Context: {context}
    Question: {question}
    """)

# 7. RAG chain - pipeline
rag_chain = get_context_from_query | prompt | llm

# 8. get result/output by invoking our rag chain
res = rag_chain.invoke(
    "what is the difference between AI and generative AI?",
      vector_store=vector_store)

# 9. output
print(res.content[-1]['text'])

"""
Based on the provided context:

* **Artificial Intelligence (AI)** is broadly defined as the simulation of human intelligence by computer systems, driven by deep learning networks, large language models (LLMs), and scalable compute infrastructure. 
* **Generative AI** is a specific core paradigm within AI focused on synthesis—specifically, models capable of creating text, images, code, and audio from complex underlying distributions, led by transformer architectures.
"""