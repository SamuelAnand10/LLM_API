# app_streamlit_rag.py
import os
import streamlit as st
from uuid import uuid4
from io import BytesIO
import tempfile
from gtts import gTTS
import base64
import requests
import json
import urllib.parse
from typing import Tuple, Any, Dict
import traceback


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
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_7PKoKb_AgCoHzfAv8u4j5NemXo3t4uELqLjj8f8tCsYf4oPTVXj9s2MAmZQgePgCVnUDvC")   # set in env / secrets
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rag-files")
GRADIO_PUBLIC_URL = os.environ.get("GRADIO_PUBLIC_URL", "https://f883b90946b8a85e72.gradio.live/")  # optional
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
                             timeout: int = 30) -> Tuple[Any, Dict, Dict]:
    """
    Call the Gradio app using the discovered api_name '/_generate' and named args.
    Returns (status_code_or_None, parsed_response_or_error_dict, debug_info)
    """
    debug = {"attempts": []}
    try:
        # Normalize base and build generate URL
        base = gradio_url.rstrip("/") + "/"
        generate_api_path = "_generate"          # note leading underscore per your discovery
        generate_url = urllib.parse.urljoin(base, generate_api_path)

        # 1) Preferred: use gradio_client (keeps exactly the same signature as your tested call)
        if HAVE_GRADIO:
            try:
                client = GradioClient(gradio_url)
                res = client.predict(
                    q=question,
                    mt=int(max_new_tokens),
                    t=float(temperature),
                    p=float(top_p),
                    rag=bool(use_rag),
                    k=int(top_k),
                    show_docs_flag=bool(show_docs),
                    api_name="/_generate",
                )
                debug["attempts"].append({"method": "gradio_client.predict", "api_name": "/_generate", "result_preview": str(res)[:1000]})
                return 200, res, debug
            except Exception as e:
                debug["attempts"].append({"method": "gradio_client.predict", "error": repr(e), "trace": traceback.format_exc()})

        # 2) HTTP POST fallback to the discovered path '/_generate'
        try:
            payload = {
                "q": question,
                "mt": int(max_new_tokens),
                "t": float(temperature),
                "p": float(top_p),
                "rag": bool(use_rag),
                "k": int(top_k),
                "show_docs_flag": bool(show_docs)
            }
            headers = {"Content-Type": "application/json"}
            r = requests.post(generate_url, json=payload, headers=headers, timeout=timeout)
            try:
                parsed = r.json()
            except Exception:
                parsed = {"raw_text": r.text}
            debug["attempts"].append({
                "method": "http_post_direct",
                "url": generate_url,
                "payload_preview": str(payload)[:1000],
                "status_code": r.status_code,
                "response_preview": str(parsed)[:1000]
            })
            if 200 <= r.status_code < 300:
                return r.status_code, parsed, debug
        except Exception as e:
            debug["attempts"].append({"method": "http_post_direct", "url": generate_url, "error": repr(e), "trace": traceback.format_exc()})

        # nothing worked
        return None, {"error": "All attempts failed (tried gradio_client and POST to /_generate)."}, debug

    except Exception as e:
        return None, {"error": "Unexpected exception in send_query_to_gradio_api", "exception": repr(e), "trace": traceback.format_exc()}, debug

def autoplay_audio(mp3_path):
    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

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
            try:
                # <-- NOTE: unpack three values (status, resp, debug)
                status, resp, debug = send_query_to_gradio_api(
                    GRADIO_PUBLIC_URL,
                    question,
                    max_new_tokens=q_max_t,
                    temperature=q_temp,
                    top_p=q_top_p,
                    use_rag=q_use_rag,
                    top_k=q_top_k,
                    show_docs=q_show_docs,
                )

                # Show high-level result
                st.write("Status:", status)
                st.subheader("Model response (raw):")
                st.write(resp)
                tts = gTTS(text=resp, lang="en-uk")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp.name)
                autoplay_audio(tmp.name)
                st.success("Done! Your audio is playing automatically.")

                # Always show debug details to diagnose payload/endpoint issues
                st.markdown("---")
                st.subheader("Debug attempts (inspect to find correct fn_index / payload shape):")
                st.json(debug)

                # Helpful hint when common errors appear
                if status is None:
                    st.error("Request failed. See debug for attempted payloads and errors.")
                    st.info("Common causes: expired Gradio share, wrong fn_index, or remote blocks programmatic access.")
                else:
                    st.success("Request completed (see above).")

            except Exception as e:
                # Shouldn't normally happen because send_query_to_gradio_api catches exceptions,
                # but keep a fallback to show full trace if something unexpected occurs.
                import traceback
                st.error("Unhandled exception when calling send_query_to_gradio_api:")
                st.error(repr(e))
                st.text(traceback.format_exc())


st.markdown("---")
st.caption("Security: never commit your Pinecone API key. Revoke any keys you posted in public and create a new one.")

