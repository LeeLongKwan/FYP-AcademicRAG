#!/usr/bin/env python3
"""
build_index.py - Optimized for strict "."-based sentence chunks
- Uses fast MiniLM embeddings (all-MiniLM-L6-v2)
- MPS GPU support for Mac M2
- Builds FAISS index + metadata + JSON monitoring file
"""
import os
import pickle
import json
import random
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import faiss

# ---------------- Config ----------------
CHUNKS_TXT = os.environ.get("CHUNKS_TXT", "chunks.txt")
INDEX_FILE = os.environ.get("INDEX_FILE", "faiss.index")
META_FILE = os.environ.get("META_FILE", "chunks.pkl")
JSON_METADATA_FILE = "index_metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EMBED_MAX_LEN = int(os.environ.get("EMBED_MAX_LEN", "512"))
SEED = 42

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using MPS (M2 GPU) for embedding")
        return torch.device("mps")
    else:
        print("⚠️ MPS not available → falling back to CPU")
        return torch.device("cpu")


def load_fast_backbone(model_name_or_path: str, device: torch.device):
    print(f"[embed] Loading fast model: {model_name_or_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts_mean_pool(model, tokenizer, texts, device, max_len=512, batch_size=32):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc, return_dict=True).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        vecs.append(vec.cpu().numpy())
    return np.vstack(vecs)


def main():
    set_seeds(SEED)
    device = get_device()

    print(f"[io] Loading chunks from: {CHUNKS_TXT}")
    records: List[Dict] = []
    with open(CHUNKS_TXT, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            txt = line.strip()
            if txt:
                records.append({
                    "id": f"chunk-{i}",
                    "text": txt,
                    "title": None,
                    "url": None,
                    "page": None
                })

    print(f"[io] Loaded {len(records)} '.'-based chunks (new strict sentence chunking)")

    texts = [r["text"] for r in records]

    tokenizer, model = load_fast_backbone(EMBED_MODEL, device)

    print(f"[embed] Computing embeddings (batch_size={BATCH_SIZE}, max_len={EMBED_MAX_LEN})...")
    X = encode_texts_mean_pool(model, tokenizer, texts, device, max_len=EMBED_MAX_LEN, batch_size=BATCH_SIZE)

    print(f"[faiss] Building IndexFlatIP (cosine) with dimension {X.shape[1]}...")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # Save index + metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(records, f)

    # Create JSON metadata for monitoring
    metadata_for_json = []
    for r in records:
        preview = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
        metadata_for_json.append({
            "id": r["id"],
            "preview": preview,
            "full_length": len(r["text"]),
            "title": r.get("title"),
            "page": r.get("page")
        })

    with open(JSON_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(records),
            "index_file": INDEX_FILE,
            "meta_file": META_FILE,
            "embedding_model": EMBED_MODEL,
            "chunking_method": "strict_period_based_sentence_chunking",
            "chunks": metadata_for_json
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Index saved → {INDEX_FILE}")
    print(f"[ok] Metadata saved → {META_FILE}")
    print(f"[ok] JSON monitoring file created → {JSON_METADATA_FILE}")
    print(f"✅ Build complete! You can now run rag_chat.py")


if __name__ == "__main__":
    main()