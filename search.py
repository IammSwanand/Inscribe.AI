# search.py
import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
# 👈 NEW IMPORTS for Compression
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

    # 1. Base Retriever: Defines how to search the vector store
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 10}) # 👈 INCREASE K to retrieve more context for the filter

    # 2. MultiQueryRetriever (Query Division): Generates sub-queries
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # --- NEW: Contextual Compression Layer ---
    # 3. Compressor: Uses the LLM to extract only the highly relevant parts from the retrieved chunks
    compressor = LLMChainExtractor.from_llm(llm) 

    # 4. Final Retriever: Combines MultiQuery and Compression
    # This retriever executes MultiQuery first, then passes all chunks to the compressor.
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=mq_retriever
    )
    # --- END OF COMPRESSION LAYER ---

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
        retriever=final_retriever, # 👈 Pass the final, filtered retriever
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return qa


def answer_query(query: str):
    qa = get_retrieval_qa()
    result = qa.invoke({"query": query})

    # 🔑 Neatly format results
    structured_answer = "### 📄 Answer\n" + result["result"] + "\n\n"
    
    # NOTE: Since you commented out the source code formatting in your last version, 
    # I've kept it commented out here.

    return {"result": structured_answer}