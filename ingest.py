# ingest.py
import os
from utils import encrypt_bytes, parse_document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import datetime
import schedule

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "legal_docs"

# Create a single Chroma client instance at the top level
client = chromadb.PersistentClient(path=CHROMA_DIR)

# The collection variable is now removed from the top level

def ingest_file(filename: str, file_bytes: bytes, uploader: str = "unknown"):
    # RE-CREATE/GET THE COLLECTION HERE, EVERY TIME
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Using collection: {collection.name}")
    
    # encrypt and store raw file (quick)
    os.makedirs("encrypted_files", exist_ok=True)
    enc = encrypt_bytes(file_bytes)
    filepath = os.path.join("encrypted_files", filename + ".enc")
    with open(filepath, "wb") as f:
        f.write(enc)

    # parse
    text = parse_document(filename, file_bytes)
    if not text or len(text.strip()) == 0:
        text = "[NO TEXT EXTRACTED]"

    # create sentence-transformers model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # splitter config
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # chunk
    docs = splitter.split_text(text)
    # create ids & metadata
    doc_ids = []
    metadatas = []
    texts = []
    for i, chunk in enumerate(docs):
        id_ = f"{filename}__chunk_{i}"
        doc_ids.append(id_)
        texts.append(chunk)
        #metadatas.append({"source_file": filename, "chunk": i, "uploader": uploader})
        metadatas.append({"source_file": filename, "chunk": i, "uploader": uploader, "created_at": int(datetime.datetime.now().timestamp())})
    # embeddings (batch)
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # add to chroma
    collection.add(
        ids=doc_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    return {"added": len(doc_ids), "file": filename}