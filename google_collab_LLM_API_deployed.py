# Falcon 7B LoRA + Pinecone v3/v8 RAG — complete single-file app
# IMPORTANT: Do NOT hardcode API keys. Set these env vars before running:
#   PINECONE_API_KEY, PINECONE_ENV (optional), PINECONE_INDEX
#
# Recommended Colab install:
# !pip install -q transformers accelerate peft bitsandbytes gradio sentence-transformers "pinecone-client>=8.0.0"

import os
import json
import traceback
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Embeddings
from sentence_transformers import SentenceTransformer

# Pinecone v3/v8 client
from pinecone import Pinecone, ServerlessSpec

# ---------------------- CONFIG ----------------------
LORA_PATH = os.environ.get("LORA_PATH", "/content/drive/MyDrive/LLM_checkpoints/lora_step_2000")
BASE_MODEL = os.environ.get("BASE_MODEL", "tiiuae/falcon-7b-instruct")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_7PKoKb_AgCoHzfAv8u4j5NemXo3t4uELqLjj8f8tCsYf4oPTVXj9s2MAmZQgePgCVnUDvC")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")  # optional for v3/v8
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "rag-files")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

# -------------------- MODEL LOAD --------------------
print("CUDA available:", torch.cuda.is_available())
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)

print("Loading base model (quantized 4-bit). This may take a few minutes...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Attaching LoRA adapter from:", LORA_PATH)
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="auto" if torch.cuda.is_available() else None)
model.eval()

# avoid past_key_values None bug
model.config.use_cache = False
if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
    model.base_model.config.use_cache = False

# ------------------- EMBEDDER -------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedder:", EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

# ------------------ PINECONE (v3/v8) ------------------
_pc = None
_index = None

def get_pc():
    """Return a Pinecone client instance (Pinecone class)."""
    global _pc
    if _pc is not None:
        return _pc
    if not PINECONE_API_KEY:
        print("PINECONE_API_KEY not set. RAG disabled.")
        return None
    try:
        _pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        print("Created Pinecone client.")
        return _pc
    except Exception as e:
        print("Failed to create Pinecone client:", repr(e))
        traceback.print_exc()
        _pc = None
        return None

def get_index():
    """Return a Pinecone index object (pc.Index(name))."""
    global _index
    if _index is not None:
        return _index
    pc = get_pc()
    if pc is None:
        return None
    try:
        _index = pc.Index(PINECONE_INDEX)
        print("Connected to Pinecone index:", PINECONE_INDEX)
        return _index
    except Exception as e:
        print("Failed to get Pinecone index:", repr(e))
        traceback.print_exc()
        _index = None
        return None

# ----------------- Debug helpers -----------------
def pinecone_list_indexes():
    pc = get_pc()
    if pc is None:
        return []
    try:
        names = pc.list_indexes()
        print("Pinecone indexes:", names)
        return names
    except Exception as e:
        print("list_indexes failed:", e)
        traceback.print_exc()
        return []

def pinecone_debug_query(query, top_k=6):
    """Encode query, query Pinecone, print compact match info and return list."""
    idx = get_index()
    if idx is None:
        print("Pinecone index not available.")
        return []
    emb = get_embedder().encode([query], normalize_embeddings=True)[0].tolist()
    try:
        res = idx.query(vector=emb, top_k=top_k, include_metadata=True)
        # normalize matches
        raw_matches = res.matches if hasattr(res, "matches") else (res.get("matches", []) if isinstance(res, dict) else [])
        out = []
        print(f"Matches returned: {len(raw_matches)}")
        for m in raw_matches:
            # m may be object or dict
            if isinstance(m, dict):
                mid = m.get("id")
                score = m.get("score")
                meta = m.get("metadata", {}) or {}
            else:
                mid = getattr(m, "id", None)
                score = getattr(m, "score", None)
                meta = getattr(m, "metadata", {}) or {}
            # find text
            text = None
            if isinstance(meta, dict):
                for k in ("text","content","chunk","body","source","raw_text"):
                    if meta.get(k):
                        text = meta.get(k)
                        break
            if text is None and isinstance(meta, str):
                text = meta
            snippet = (text[:400] + "...") if isinstance(text, str) and len(text) > 400 else text
            keys = list(meta.keys()) if isinstance(meta, dict) else None
            print(f"ID={mid} SCORE={score} KEYS={keys}\nSNIPPET: {snippet}\n----")
            out.append({"id": mid, "score": score, "keys": keys, "snippet": snippet})
        if not raw_matches:
            print("No matches returned.")
        return out
    except Exception as e:
        print("Pinecone query failed:", e)
        traceback.print_exc()
        return []

# ----------------- RAG retrieval -----------------
def rag_retrieve(query, top_k=DEFAULT_TOP_K):
    idx = get_index()
    if idx is None:
        return []
    emb = get_embedder().encode([query], normalize_embeddings=True)[0].tolist()
    try:
        res = idx.query(vector=emb, top_k=top_k, include_metadata=True)
        raw_matches = res.matches if hasattr(res, "matches") else (res.get("matches", []) if isinstance(res, dict) else [])
        docs = []
        for m in raw_matches:
            meta = m.metadata if hasattr(m, "metadata") else (m.get("metadata", {}) if isinstance(m, dict) else {})
            text = None
            if isinstance(meta, dict):
                for k in ("text","content","chunk","body","source","raw_text"):
                    if meta.get(k):
                        text = meta.get(k)
                        break
            if text is None and isinstance(meta, str):
                text = meta
            if text:
                docs.append(text[:1500])
        return docs
    except Exception as e:
        print("RAG retrieval failed:", e)
        traceback.print_exc()
        return []

# ----------------- Prompting -----------------
def build_rag_prompt(query, docs):
    if not docs:
        return f"### Instruction:\n{query}\n\n### Response:\n"
    ctx = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
    return (
        "You are an assistant with access to external documentation.\n"
        "Use ONLY the provided context to answer the question. If the context doesn't contain the answer, reply 'I don't know.'\n\n"
        f"### Context:\n{ctx}\n\n"
        f"### Instruction:\n{query}\n\n"
        "### Response:\n"
    )

# ----------------- Generation -----------------
def generate_reply(user_text, max_new_tokens=128, temperature=0.7, top_p=0.95, use_rag=False, top_k=DEFAULT_TOP_K, show_docs=False):
    context = []
    if use_rag:
        context = rag_retrieve(user_text, top_k=int(top_k))
        # also print debug to logs
        pinecone_debug_query(user_text, top_k=int(top_k))
    prompt = build_rag_prompt(user_text, context if context else None)
    # debug prompt (truncated)
    print("PROMPT (truncated 2000 chars):")
    print(prompt[:2000])

    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    reply = txt.split("### Response:")[-1].strip() if "### Response:" in txt else (txt[len(prompt):].strip() if txt.startswith(prompt) else txt.strip())

    if show_docs and context:
        docs_text = "\n\n--- Retrieved docs (top_k) ---\n\n"
        for i, d in enumerate(context):
            docs_text += f"[{i+1}] {d}\n\n"
        return reply + docs_text
    return reply

# ----------------- GRADIO UI -----------------
with gr.Blocks() as demo:
    gr.Markdown("## Local LoRA Falcon Chat (with Pinecone v3/v8 RAG)")
    with gr.Row():
        inp = gr.Textbox(label="You", placeholder="Ask something...", lines=2)
        out = gr.Textbox(label="AI response", lines=8)
    with gr.Row():
        max_t = gr.Slider(16, 512, value=128, step=16, label="max_new_tokens")
        temp = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
    with gr.Row():
        use_rag = gr.Checkbox(label="Use Pinecone RAG (requires env vars and index)", value=False)
        top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="RAG: top_k")
        show_docs = gr.Checkbox(label="Show retrieved docs (inline)", value=False)

    def _generate(q, mt, t, p, rag, k, show_docs_flag):
        try:
            return generate_reply(q, max_new_tokens=mt, temperature=t, top_p=p, use_rag=rag, top_k=k, show_docs=show_docs_flag)
        except Exception as e:
            traceback.print_exc()
            return f"Error during generation: {repr(e)} (see logs)"

    btn = gr.Button("Generate")
    btn.click(fn=_generate, inputs=[inp, max_t, temp, top_p, use_rag, top_k, show_docs], outputs=out)

# Print Gradio config and launch
try:
    print("Gradio config:", json.dumps(demo.get_config(), indent=2))
except Exception:
    pass

print("Launching Gradio (share=True)...")
demo.launch(share=True, server_name="0.0.0.0")
