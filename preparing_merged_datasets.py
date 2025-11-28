"""
prepare_merged_dataset.py

Downloads ShareGPT, Dolly-15k, and Alpaca from Hugging Face,
cleans and merges them into a single JSONL suitable for
instruction-tuning your assistant.

Output file: assistant_data_cleaned_merged.jsonl
Default target size: 50_000 examples

Usage:
    python prepare_merged_dataset.py

Requirements (install in venv):
    pip install datasets transformers sentencepiece tqdm regex

Notes:
- Tone: Energetic + enthusiastic is enforced by prepending a short style instruction
  to each prompt: "You are an energetic and enthusiastic assistant. "
- Safety: basic blacklist filtering for illegal/harmful topics and profanity.
- Length filter uses the tokenizer to estimate token length (max_tokens threshold).
"""

import json, re, random, os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------- CONFIG ----------
OUTPUT_FILE = "assistant_data_cleaned_merged.jsonl"
TARGET_SIZE = 50000
RANDOM_SEED = 42

# Tone / style prefix added at the start of every prompt so the model learns style.
STYLE_PREFIX = "You are an energetic and enthusiastic assistant. "

# Basic blacklist of dangerous / illegal keywords (will drop examples containing these)
BLACKLIST = [
    # weapons / explosives / illegal drugs
    "bomb", "explode", "detonate", "detonation", "how to make a bomb",
    "ricin", "explosive", "weaponize", "manufacture explosives",
    "meth", "how to make meth", "grow shrooms", "manufacture fentanyl",
    # hacking / breaking into systems
    "exploit", "sql injection", "bypass authentication", "brute force",
    "hack into", "how to hack", "ddos", "denial of service",
    # violent wrongdoing instructions
    "kill", "assassinate", "how to kill", "murder",
    # sexual exploitation / child sexual
    "child porn", "cp ", "sexual exploitation",
]

# profanity pattern (simple, not exhaustive)
PROFANITY_RE = re.compile(r"\b(shit|fuck|bitch|cunt|motherfucker|asshole)\b", re.IGNORECASE)

# max tokens (approx). We'll use a tokenizer to estimate tokens,
# but also cap on raw characters to avoid huge examples.
MAX_TOKENS = 1024
MAX_CHARS = 20000

# tokenizer for length checks (use a small tokenizer that is available)
TOKENIZER_MODEL = "huggingface/CodeBERT-small-v1"  # small tokenizer; only used for token-length heuristics
# ----------------------------

random.seed(RANDOM_SEED)

def basic_clean_text(s: str) -> str:
    # normalize whitespace, strip weird unicode control chars
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\u200b|\u200e|\u200f", "", s)  # remove zero-width chars
    s = s.strip()
    # collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def contains_blacklist(s: str) -> bool:
    txt = s.lower()
    for kw in BLACKLIST:
        if kw in txt:
            return True
    if PROFANITY_RE.search(txt):
        return True
    return False

def extract_pairs_from_sharegpt(ds):
    """
    Each item in ShareGPT has 'conversations' list with dicts {from: human/gpt, value: str}
    We'll extract (human -> next gpt) pairs across turns.
    """
    pairs = []
    for item in tqdm(ds, desc="Processing ShareGPT"):
        conv = item.get("conversations") or []
        # flatten tiny cleaning
        for i in range(len(conv)-1):
            a = conv[i]
            b = conv[i+1]
            if not a or not b:
                continue
            if a.get("from","").lower() == "human" and b.get("from","").lower() in ("gpt","assistant","gpt4","openai"):
                prompt = basic_clean_text(a.get("value",""))
                response = basic_clean_text(b.get("value",""))
                if prompt and response:
                    pairs.append((prompt, response))
    return pairs

def extract_pairs_from_dolly(ds):
    """
    Dolly format usually has fields 'instruction' and 'response' (or similar).
    We'll try common keys.
    """
    pairs = []
    for item in tqdm(ds, desc="Processing Dolly"):
        # Dolly from databricks/databricks-dolly-15k uses 'instruction' and 'response' or 'input'+'output'
        prompt = item.get("instruction") or item.get("input") or item.get("prompt") or ""
        response = item.get("response") or item.get("output") or item.get("completion") or ""
        prompt = basic_clean_text(prompt)
        response = basic_clean_text(response)
        if prompt and response:
            pairs.append((prompt, response))
    return pairs

def extract_pairs_from_alpaca(ds):
    """
    Alpaca tends to have 'instruction' and 'output'
    """
    pairs = []
    for item in tqdm(ds, desc="Processing Alpaca"):
        prompt = item.get("instruction") or item.get("input") or ""
        response = item.get("output") or item.get("response") or ""
        prompt = basic_clean_text(prompt)
        response = basic_clean_text(response)
        if prompt and response:
            pairs.append((prompt, response))
    return pairs

def passes_filters(prompt, response, tokenizer):
    # length checks
    if len(prompt) > MAX_CHARS or len(response) > MAX_CHARS:
        return False
    # token length estimate
    plen = len(tokenizer.tokenize(prompt))
    rlen = len(tokenizer.tokenize(response))
    if plen + rlen > MAX_TOKENS:
        return False
    # blacklist / profanity checks across both fields
    if contains_blacklist(prompt) or contains_blacklist(response):
        return False
    # avoid extremely short responses (like "Ok")
    if len(response) < 3:
        return False
    return True

def format_prompt(prompt):
    # Prepend style instruction so assistant learns tone.
    # We keep the user's original prompt after the prefix.
    return STYLE_PREFIX + prompt

def main():
    # load small tokenizer for token-length heuristics
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
    except Exception as e:
        print("Tokenizer load failed, using fallback whitespace tokenizer. Error:", e)
        class DummyTok:
            def tokenize(self, text): return text.split()
        tokenizer = DummyTok()

    all_pairs = []

    # ---------- ShareGPT ----------
    print("Loading ShareGPT (this can take a while)...")
    try:
        sg = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        pairs_sg = extract_pairs_from_sharegpt(sg)
        print(f"Extracted {len(pairs_sg)} pairs from ShareGPT")
        all_pairs.extend(pairs_sg)
    except Exception as e:
        print("Warning: failed to load ShareGPT:", e)

    # ---------- Dolly 15k ----------
    print("Loading Dolly 15k...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        pairs_dolly = extract_pairs_from_dolly(dolly)
        print(f"Extracted {len(pairs_dolly)} pairs from Dolly")
        all_pairs.extend(pairs_dolly)
    except Exception as e:
        print("Warning: failed to load Dolly:", e)

    # ---------- Alpaca ----------
    print("Loading Alpaca...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        pairs_alpaca = extract_pairs_from_alpaca(alpaca)
        print(f"Extracted {len(pairs_alpaca)} pairs from Alpaca")
        all_pairs.extend(pairs_alpaca)
    except Exception as e:
        print("Warning: failed to load Alpaca:", e)

    print("Total raw pairs collected:", len(all_pairs))

    # ---------- Cleaning + filtering ----------
    print("Filtering / deduping / length-checks ...")
    seen = set()
    cleaned = []
    for (p, r) in tqdm(all_pairs, desc="Cleaning pairs"):
        # basic normalization
        p2 = basic_clean_text(p)
        r2 = basic_clean_text(r)
        key = (p2.lower(), r2.lower())
        if key in seen:
            continue
        # safety + length filters
        if not passes_filters(p2, r2, tokenizer):
            continue
        seen.add(key)
        cleaned.append((p2, r2))

    print("After cleaning & dedupe:", len(cleaned))

    # ---------- Sampling & balancing ----------
    print("Shuffling and sampling to target size:", TARGET_SIZE)
    random.shuffle(cleaned)
    # If not enough examples, we will just use all. Otherwise sample.
    if len(cleaned) >= TARGET_SIZE:
        sampled = cleaned[:TARGET_SIZE]
    else:
        sampled = cleaned  # not enough, use all
        print(f"Warning: only {len(cleaned)} cleaned examples available < target {TARGET_SIZE}")

    # ---------- Format into JSONL and write file ----------
    print("Writing JSONL to", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for (p, r) in tqdm(sampled, desc="Writing"):
            prompt = format_prompt(p)
            obj = {"prompt": prompt, "response": r}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done. Wrote", len(sampled), "examples to", OUTPUT_FILE)


main()
