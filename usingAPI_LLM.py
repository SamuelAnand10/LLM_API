# app_streamlit_rag.py
import os
import streamlit as st
from uuid import uuid4
from io import BytesIO
import tempfile
import requests
import json

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
GRADIO_PUBLIC_URL = os.environ.get("GRADIO_PUBLIC_URL", "https://871f07cf0ef9785f47.gradio.live/")  # optional
GRADIO_PREDICT = os.path.join(GRADIO_PUBLIC_URL.rstrip("/"), "api/predict/")

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
    """
    Initialize Pinecone: create index if not exists, return Index object.
    Uses passed api_key/env/index_name/dim so it's not dependent on globals.
    Attaches index_name attribute to the returned Index for backwards compatibility.
    """
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment.")
    # Create client and index (ServerlessSpec kept from your original code)
    pc = Pinecone(api_key=api_key, environment=env)
    try:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=env
            )
        )
        print(f"Created Pinecone index: {index_name}")
    except PineconeApiException as e:
        # If index already exists, continue; else re-raise
        try:
            if getattr(e, "status", None) == 409 or "ALREADY_EXISTS" in str(getattr(e, "body", "")):
                print(f"Pinecone index '{index_name}' already exists. Connecting to existing index.")
            else:
                raise e
        except Exception:
            # If SDK shapes differ, try string match
            if "ALREADY_EXISTS" in str(e):
                print(f"Pinecone index '{index_name}' already exists. Connecting to existing index.")
            else:
                raise e
    idx = pc.Index(index_name)
    # attach index_name for compatibility with other parts of the code
    try:
        setattr(idx, "index_name", index_name)
    except Exception:
        pass
    return idx

def index_chunks_to_pinecone(idx, embedder, title, chunks, source_path, batch_size=64):
    n = len(chunks)
    display_index = getattr(idx, "index_name", "<unknown_index>")
    st.info(f"Indexing {n} chunks into Pinecone (index: {display_index})...")
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

def send_query_to_gradio_api(gradio_url: str, question: str, max_new_tokens:int=128,
                             temperature:float=0.7, top_p:float=0.95,
                             use_rag:bool=False, top_k:int=4, show_docs:bool=False,
                             timeout: int = 30):
    """
    Robust POST to a Gradio share /api/predict/ endpoint.
    Tries payloads that Gradio commonly expects, including fn_index.
    Returns (status_code, response_json, debug_info)
    - status_code may be None if request failed entirely.
    - response_json is parsed JSON or a dict with 'raw_text' or 'error'.
    - debug_info tells which URL/payloads were attempted.
    """

    base = gradio_url.rstrip("/") + "/"
    predict_path = "api/predict/"
    predict_url = urllib.parse.urljoin(base, predict_path)

    # Build data (we assume your remote function signature ordering)
    data_list = [
        question,
        int(max_new_tokens),
        float(temperature),
        float(top_p),
        bool(use_rag),
        int(top_k),
        bool(show_docs)
    ]

    headers = {"Content-Type": "application/json"}
    attempts = []

    # 1) Try gradio_client if available (preferred)
    if HAVE_GRADIO:
        try:
            client = GradioClient(gradio_url)
            # gradio_client.predict uses the python-callable signature, no fn_index needed
            res = client.predict(question, int(max_new_tokens), float(temperature),
                                 float(top_p), bool(use_rag), int(top_k), bool(show_docs),
                                 api_name="/predict", timeout=timeout)
            return 200, res, {"method": "gradio_client.predict", "url": gradio_url}
        except Exception as e:
            attempts.append({"method": "gradio_client.predict", "error": str(e)})

    # 2) Try POST with fn_index = 0 (most common)
    payloads = [
        {"fn_index": 0, "data": data_list},
        # 3) fallback: no fn_index (some older/simple endpoints accept this)
        {"data": data_list},
        # 4) sometimes servers expect fn_index as string (rare) or query param; include query param attempt below
    ]

    for payload in payloads:
        try:
            resp = requests.post(predict_url, json=payload, headers=headers, timeout=timeout)
            try:
                j = resp.json()
            except Exception:
                j = {"raw_text": resp.text}
            attempts.append({"url": predict_url, "payload": payload, "status_code": resp.status_code, "response_preview": str(j)[:500]})
            # 200-299 -> success
            if 200 <= resp.status_code < 300:
                return resp.status_code, j, {"attempts": attempts}
        except Exception as e:
            attempts.append({"url": predict_url, "payload": payload, "error": str(e)})

    # 5) Try adding fn_index as query parameter (some edge cases)
    try:
        qurl = predict_url + "?fn_index=0"
        resp = requests.post(qurl, json={"data": data_list}, headers=headers, timeout=timeout)
        try:
            j = resp.json()
        except Exception:
            j = {"raw_text": resp.text}
        attempts.append({"url": qurl, "payload": {"data": data_list}, "status_code": resp.status_code, "response_preview": str(j)[:500]})
        if 200 <= resp.status_code < 300:
            return resp.status_code, j, {"attempts": attempts}
    except Exception as e:
        attempts.append({"url": predict_url + "?fn_index=0", "error": str(e)})

    # Nothing worked
    return None, {"error": "All attempts failed", "attempts": attempts}, {"attempts": attempts}

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
                optionally_send_to_gradio(GRADIO_PUBLIC_URL, path_to_send)

# ----------------- Query UI (send question to your Colab/Gradio model) -----------------
st.markdown("---")
st.header("Ask your LLM (Colab / Gradio) — with RAG")
st.write("This will POST to your Gradio app's `/api/predict/` endpoint. Make sure your Colab Gradio app is running and `share=True` if it's remote.")

q_col1, q_col2 = st.columns([3,1])
with q_col1:
    question = st.text_area("Question", value="Summarize the uploaded document in 2 sentences.", height=120)
with q_col2:
    q_max_t = st.number_input("max_new_tokens", min_value=16, max_value=1024, value=128)
    q_temp = st.number_input("temperature", min_value=0.01, max_value=2.0, value=0.7, format="%.2f")
    q_top_p = st.number_input("top_p", min_value=0.01, max_value=1.0, value=0.95, format="%.2f")
    q_use_rag = st.checkbox("Use RAG (let remote model query Pinecone)", value=True)
    q_top_k = st.slider("RAG: top_k", 1, 10, value=4)
    q_show_docs = st.checkbox("Show retrieved docs (if remote returns them)", value=True)

if st.button("Send question to LLM"):
    if not GRADIO_PUBLIC_URL:
        st.error("GRADIO_PUBLIC_URL not set. Set the env var or Streamlit secrets to your Gradio share URL.")
    else:
        with st.spinner("Sending query to Gradio predict API..."):
            status, resp = send_query_to_gradio_api(GRADIO_PUBLIC_URL, question, max_new_tokens=q_max_t, temperature=q_temp, top_p=q_top_p, use_rag=q_use_rag, top_k=q_top_k, show_docs=q_show_docs)
            if status is None:
                st.error(f"Request failed: {resp.get('error')}")
                st.info("Common causes: Gradio share session expired, remote server blocks programmatic access, or wrong endpoint.")
            else:
                st.write("Status:", status)
                st.subheader("Model response (raw):")
                # Gradio /api/predict/ often returns {"data": [...outputs...], "duration":..., ...}
                if isinstance(resp, dict) and 'data' in resp and isinstance(resp['data'], list):
                    # Usually the first element is model text; but remote apps vary — show everything
                    try:
                        # Prefer first item
                        primary = resp['data'][0]
                        # If primary is a string, show it raw; if list/dict, pretty-print
                        if isinstance(primary, str):
                            st.text_area("LLM output", primary, height=300)
                        else:
                            st.json(primary)
                    except Exception:
                        st.json(resp['data'])
                    # If the remote app returned structured docs, try to display them too
                    # Some remote implementations return the retrieved docs as a 2nd item or embedded in a dict
                    st.markdown("----")
                    st.subheader("Full response JSON")
                    st.json(resp)
                else:
                    # fallback: just print the whole response object
                    st.write(resp)

st.markdown("---")
st.caption("Security: never commit your Pinecone API key. Revoke any keys you posted in public and create a new one.")

