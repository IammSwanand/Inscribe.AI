import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from langchain.retrievers import MultiQueryRetriever # For Agentic RAG Implementation
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "legal_docs"

# Embeddings
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_retrieval_qa(model_name="llama-3.1-8b-instant"):
    """Return RetrievalQA chain using Groq LLM"""

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=hf,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    llm = ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1024,
    )

    # Base retriever
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    
    # Multi-Query Retriever
    mq_retriever = MultiQueryRetriever.from_retriever(
        llm=llm,
        retriever=base_retriever,
    )
    
    # Contextual Compression Retriever
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Final Retruever with compression
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever,
    )
    

    # 🔑 STRICT + STRUCTURED PROMPT
    CUSTOM_PROMPT_TEMPLATE = """
    You are highly efficient legal assistant. Use ONLY the context below to answer the question.
    If the context does not contain the answer, reply: "Not found in the documents."

    Cite the source inline by referencing the source_file and page number in square brackets, 
    right after the relevant sentence (example: [contract.docx, page 2]).

    Format the response in a clear and structured way with headings and bullet points.

    ---------------------
    Context:
    {context}
    ---------------------

    Question:
    {question}

    Answer (use only the context above):
    """

    CUSTOM_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

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

    # 🔑 Neatly format results
    structured_answer = "### 📄 Answer\n" + result["result"] + "\n\n"

    # structured_answer += "### 📚 Sources\n"
    # for doc in result.get("source_documents", []):
    #     meta = doc.metadata
    #     structured_answer += f"- {meta.get('source_file', 'unknown')} (chunk {meta.get('chunk', '?')})\n"

    return {"result": structured_answer}
