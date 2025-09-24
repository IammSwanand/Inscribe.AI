# app.py
import streamlit as st
from ingest import ingest_file
from search import answer_query
import os
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "legal_docs"

# Configure Streamlit page
st.set_page_config(page_title="Inscribe.AI", layout="wide")
st.title("Inscribe.AI")

# Simple password gate (for dev only)
if "authorized" not in st.session_state:
    st.session_state.authorized = False

if not st.session_state.authorized:
    pwd = st.text_input("Enter password to access (for dev only)", type="password")
    if st.button("Enter"):
        if pwd == "devpass":  # change to something secure
            st.session_state.authorized = True
        else:
            st.error("Wrong password")
    st.stop()

# ---------------- File Upload ----------------
st.header("Upload files")
uploaded_files = st.file_uploader("PDF / DOCX / TXT", accept_multiple_files=True)
uploader_name = st.text_input("Uploader name (optional)")

agree = st.checkbox("I agree to giving consent to Inscribe.AI to process my documents")

if agree:
    st.info("please check the box to agree before uploading")
else:
    st.warning("You must agree before uploading")

if st.button("Ingest files"):
    if not uploaded_files:
        st.warning("Please choose files")
    else:
        status_area = st.empty()
        for f in uploaded_files:
            if agree:
                b = f.read()
                status_area.text(f"Ingesting {f.name}...")
                res = ingest_file(f.name, b, uploader=uploader_name or "unknown")
                status_area.text(f"Ingested {res['file']}: {res['added']} chunks")
            else:
                st.error("You must agree to the consent checkbox to upload")
        st.success("Done ingesting")

# ---------------- Clear Database ----------------
st.header("Manage Database")
if st.button("Clear all documents from the database"):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        st.text("Clearing collection...")
        client.delete_collection(COLLECTION_NAME)
    st.success("Database cleared! You can now upload new documents.")

# ---------------- Query Section ----------------
st.header("Query")
q = st.text_area("Ask a question about your ingested documents")

if st.button("Search"):
    if not q:
        st.warning("Type a question")
    else:
        if not GROQ_API_KEY:
            st.error("Set GROQ_API_KEY in .env to run retrieval + LLM")
        else:
            with st.spinner("Searching..."):
                res = answer_query(q)

            if isinstance(res, dict) and "result" in res:
                st.markdown("### Answer")
                st.write(res["result"])

            #     st.markdown("### Sources")
            #     for doc in res.get("source_documents", []):
            #         meta = doc.metadata
            #         st.write(f"- {meta.get('source_file', 'unknown')} (chunk {meta.get('chunk', '?')})")
            else:
                st.write(res)

