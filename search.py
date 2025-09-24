import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "legal_docs"

# Embeddings
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_retrieval_qa(model_name="llama-3.1-8b-instant"):
    """Return RetrievalQA chain using Groq LLM"""
    
    # FIX: Create client and vectordb inside the function
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        # Using the get_or_create_collection pattern here to be robust
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        # Fallback to create if get_or_create fails for any reason
        collection = client.create_collection(name=COLLECTION_NAME)

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=hf,
        collection_name=COLLECTION_NAME,
        client=client,
    )
    # END OF FIX

    llm = ChatGroq(
          model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1024,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    # Define a custom prompt template for summarization
    CUSTOM_PROMPT_TEMPLATE = """
    You are a helpful assistant. Your task is to provide a concise summary of the provided text.
    
    Context:
    {context}
    
    Summary:
    """
    CUSTOM_PROMPT = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return qa

def answer_query(query: str):
    qa = get_retrieval_qa()
    result = qa.invoke({"query": query})
    return result