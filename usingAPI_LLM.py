# app_streamlit_rag.py
import os
import streamlit as st
from uuid import uuid4
from io import BytesIO
import tempfile

# PDF reading
from PyPDF2 import PdfReader

# Embedding + Pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone.exceptions import PineconeApiException

import numpy as np
from typing import List

# Optional: talk to your Gradio/Colab public URL (if you want)
try:
    from gradio_client import Client as GradioClient
    HAVE_GRADIO = True
except Exception:
    HAVE_GRADIO = False

# ---------- CONFIG (set via env vars or Streamlit secrets) ----------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")   # set in env / secrets
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rag-files")
GRADIO_PUBLIC_URL = os.environ.get("GRADIO_PUBLIC_URL", "https://f4b1bb5c13d8313f42.gradio.live/")  # optional

# Developer-provided local path fallback (per dev instruction)
DEV_UPLOADED_PATH = "/mnt/data/1814dcfa-776c-40c6-aedb-fea130de1f2a.png"

# ---------- Helpers ----------
@st.cache_resource
def load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    # Extract text with PyPDF2
    r = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in r.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()

def chunk_text(text: str, max_chars:int = 800) -> List[str]:
    # simple chunker that prefers newline/space breaks
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + max_chars, N)
        if end < N:
            # try to break at newline or space
            br = text.rfind("\n", start, end)
            if br <= start:
                br = text.rfind(" ", start, end)
            if br > start:
                end = br
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

@st.cache_resource
def init_pinecone(api_key: str, env: str, index_name: str, dim: int):
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment.")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    EMBED_DIM = embedder.get_sentence_embedding_dimension()
    try:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )
        print(f"Created Pinecone index: {PINECONE_INDEX}")
    except PineconeApiException as e:
    # Check if the error is due to the index already existing
        if e.status == 409 and "ALREADY_EXISTS" in str(e.body):
            print(f"Pinecone index '{PINECONE_INDEX}' already exists. Connecting to existing index.")
        else:
        # Re-raise other Pinecone API exceptions
            raise e
    return pc.Index(index_name)

def index_chunks_to_pinecone(idx, embedder, title, chunks, source_path, batch_size=64):
    n = len(chunks)
    st.info(f"Indexing {n} chunks into Pinecone (index: {idx.index_name})...")
    for i in range(0, n, batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = embedder.encode(batch, normalize_embeddings=True)
        upserts = []
        for j, (txt, emb) in enumerate(zip(batch, embeddings)):
            vec_id = f"{title.replace(' ','_')}_{i+j}_{uuid4().hex[:8]}"
            metadata = {"title": title, "text": txt[:1000], "source": source_path}
            upserts.append((vec_id, emb.tolist(), metadata))
        idx.upsert(vectors=upserts)
    st.success("Upserted to Pinecone.")

def optionally_send_to_gradio(gradio_url: str, local_file_path: str):
    if not HAVE_GRADIO:
        st.warning("gradio_client not installed; skipping Gradio push.")
        return None
    client = GradioClient(gradio_url)
    # best-effort call: if your Gradio app has an upload API, adjust api_name and payload accordingly
    try:
        # some Gradio apps expect file path or URL as a string input; adapt to your app
        res = client.predict(local_file_path, api_name="/upload", timeout=60)
        st.info("Sent file to Gradio endpoint; response:")
        st.write(res)
        return res
    except Exception as e:
        st.error(f"Failed to call Gradio endpoint: {e}")
        return None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG uploader (Streamlit)", layout="centered")

st.title("RAG PDF uploader → Pinecone")
st.write("Upload a PDF, preview extracted text, then embed & push to Pinecone. Keys must be set via env vars or Streamlit secrets.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# Allow user to optionally use the developer-provided local path as source
use_dev_path = st.checkbox("(Dev) Use pre-uploaded file path as source metadata", value=False)
if use_dev_path:
    st.write("Using developer path as source:", DEV_UPLOADED_PATH)

if uploaded_file is None and not use_dev_path:
    st.info("Upload a PDF or enable dev fallback to proceed.")
else:
    # choose title
    title = st.text_input("Document title (for metadata / ids)", value=(uploaded_file.name if uploaded_file else "dev_file"))

    if uploaded_file:
        # show size
        st.write("Uploaded file:", uploaded_file.name, "| size:", uploaded_file.size, "bytes")
        raw_bytes = uploaded_file.read()
        # extract text
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf_bytes(raw_bytes)
        if not text:
            st.error("No extractable text found. If it's a scanned image PDF you need OCR (tesseract).")
        else:
            st.subheader("Preview of extracted text")
            st.text_area("extracted_text_preview", text[:5000], height=300)

    else:
        # dev path fallback: we do not have the PDF bytes — metadata will point to the dev path
        text = f"File referenced by dev path: {DEV_UPLOADED_PATH}"
        st.markdown(f"**Note:** using dev fallback source: `{DEV_UPLOADED_PATH}`")

    # show action button
    if st.button("Embed & Push to Pinecone"):
        # load embedder
        with st.spinner("Loading embedder..."):
            embedder = load_embedder()
            dim = embedder.get_sentence_embedding_dimension()

        # init pinecone
        try:
            idx = init_pinecone(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, dim)
        except Exception as e:
            st.error(f"Failed to init pinecone: {e}")
            st.stop()

        # chunk text (if upload had text); if using dev fallback, index a single doc referencing dev path
        if uploaded_file and text:
            chunks = chunk_text(text, max_chars=800)
        else:
            chunks = [f"Referenced file: {DEV_UPLOADED_PATH}"]

        source_path = DEV_UPLOADED_PATH if use_dev_path else (uploaded_file.name if uploaded_file else DEV_UPLOADED_PATH)

        # index
        try:
            index_chunks_to_pinecone(idx, embedder, title, chunks, source_path)
        except Exception as e:
            st.error(f"Failed to upsert vectors: {e}")
            st.stop()

        st.success("Done indexing into Pinecone.")

        # optionally notify your Gradio/Colab app (best-effort)
        if st.checkbox("Also send file path to Gradio/Colab endpoint (optional)"):
            with st.spinner("Calling Gradio endpoint..."):
                # choose path to send: we send the local path if using dev fallback, else you can write uploaded file to tmp and send that path
                if use_dev_path:
                    path_to_send = DEV_UPLOADED_PATH
                else:
                    # write uploaded file to a temp path so the Gradio side (if accessible) can fetch it
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(raw_bytes)
                    tmp.flush()
                    tmp.close()
                    path_to_send = tmp.name
                

st.markdown("---")
st.caption("Security: never commit your Pinecone API key. Revoke any keys you posted in public and create a new one.")
