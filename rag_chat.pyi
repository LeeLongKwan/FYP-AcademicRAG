#!/usr/bin/env python3
import os
import re
import json
import time
import pickle
import warnings
import threading
from queue import Queue
from typing import Dict, Optional, Generator

import torch
import faiss
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# configurations
MODEL_NAME = "huggyllama/llama-7b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss.index"
META_FILE = "chunks.pkl"
RESPONSES_JSON = "dictionary.json"

EMBED_MAX_LEN = 512
MAX_NEW_TOKENS = 128
MAX_SOURCES = 3
RAG_TIMEOUT_SECONDS = 120

# streaming rate cap: 1 word per 0.25 s => 4 words/sec
RATE_WORDS_PER_SEC = 4.0
RATE_WORD_INTERVAL = 1.0 / RATE_WORDS_PER_SEC  # = 0.25 seconds per word

APOLOGY_MSG = (
    "Sorry, this is taking longer than expected (over 120 seconds) and I don't have the answer now. "
    "I’ve stopped this run. You can try narrowing your question or reducing the scope."
)

# dictionary 
with open(RESPONSES_JSON, "r", encoding="utf-8") as f:
    KEYWORD_RESPONSES: Dict[str, str] = json.load(f)

PATTERN_RESPONSES: Dict[re.Pattern, str] = {
    re.compile(r"(?i)what.*(is|the).*acousto.*electric"): KEYWORD_RESPONSES.get("acousto-electric", ""),
    re.compile(r"(?i)how.*(does|work|principle|mechanism)"): KEYWORD_RESPONSES.get("principle", ""),
    re.compile(r"(?i)(debye|uvp|ultrasonic.*vibration)"): KEYWORD_RESPONSES.get("debye", ""),
    re.compile(r"(?i)(uai|ultrafast.*acoustoelectric)"): KEYWORD_RESPONSES.get("uai", ""),
    re.compile(r"(?i)(heart|cardiac|rat|swine)"): KEYWORD_RESPONSES.get("rat heart", ""),
    re.compile(r"(?i)(lobster|nerve)"): KEYWORD_RESPONSES.get("lobster", ""),
    re.compile(r"(?i)(factor|constant|k_i|interaction.*factor)"): KEYWORD_RESPONSES.get("interaction factor", ""),
    re.compile(r"(?i)(application|applications|use case|practical use)"): KEYWORD_RESPONSES.get("application", ""),
    re.compile(r"(?i)(general use|general application|everyday use|typical use)"): KEYWORD_RESPONSES.get("general use", ""),
    re.compile(r"(?i)(significance|importance|why important|value|benefit)"): KEYWORD_RESPONSES.get("significance", ""),
    re.compile(r"(?i)(measure|measurement|how to measure)"): KEYWORD_RESPONSES.get("measurement", ""),
    re.compile(r"(?i)(equipment|setup|hardware)"): KEYWORD_RESPONSES.get("equipment", ""),
    re.compile(r"(?i)(resolution|spatial|temporal)"): KEYWORD_RESPONSES.get("resolution", ""),
    re.compile(r"(?i)(tissue|biological|in vivo|in vitro)"): KEYWORD_RESPONSES.get("tissue vs saline", ""),
    re.compile(r"(?i)(safety|safe|biological effect)"): KEYWORD_RESPONSES.get("safety", ""),
    re.compile(r"(?i)(brain|neural|nerve)"): KEYWORD_RESPONSES.get("brain", ""),
    re.compile(r"(?i)(compare|comparison|vs|versus)"): KEYWORD_RESPONSES.get("comparison", ""),
    re.compile(r"(?i)(future|next|advance|development)"): KEYWORD_RESPONSES.get("future", ""),
}

def get_fixed_response(user_query: str) -> Optional[str]:
    query_lower = user_query.lower().strip()
    for keyword, response in KEYWORD_RESPONSES.items():
        if keyword and keyword in query_lower:
            return response
    for pattern, response in PATTERN_RESPONSES.items():
        if response and pattern.search(user_query):
            return response
    return None

# LOAD MODELS 
print("[ACTION] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    meta = pickle.load(f)

print("[ACTION] Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)
embed_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
embed_model.to(embed_device).eval()

print("[ACTION] Loading generation model (Llama-7B)... This may take 1-2 minutes on M2")
gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
if gen_tokenizer.pad_token_id is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

gen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="mps" if torch.backends.mps.is_available() else "cpu",
    low_cpu_mem_usage=True,
)

# cooperative cancel
class CancelToken:
    def __init__(self):
        self.stop_event = threading.Event()
        self.start_time = time.time()

    def reset(self):
        self.stop_event.clear()
        self.start_time = time.time()

    def stop(self):
        self.stop_event.set()

    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def elapsed(self) -> float:
        return time.time() - self.start_time

def check_abort_or_timeout(token: "CancelToken") -> Optional[str]:
    if token.stopped():
        return "Response stopped by user."
    if token.elapsed() > RAG_TIMEOUT_SECONDS:
        return APOLOGY_MSG
    return None

# rate limiter
class RateLimiter:
    """
    Deterministic rate limiter for streaming words.
    Ensures at most `rate_per_sec` words/sec are emitted.
    First acquire() returns immediately; subsequent acquires wait to meet the interval.
    """
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / max(1e-6, rate_per_sec)
        self.next_time = time.perf_counter()

    def acquire(self, units: int = 1):
        target = self.next_time
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
            now = time.perf_counter()
        self.next_time = now + self.interval * units

# snippet formatting
def format_snippets(sources, max_chars=300):
    if not sources:
        return "No candidate snippets found."
    lines = []
    for i, s in enumerate(sources, 1):
        txt = (s.get("snippet", "") or "").strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "…"
        lines.append(f"{i}. {txt}")
    return "Retrieved candidate snippets:\n" + "\n".join(lines)

# search 
def search(query: str, token: "CancelToken", k: int = MAX_SOURCES):
    print("[ACTION] Embedding query for retrieval...")
    enc = embed_tokenizer([query], padding=True, truncation=True,
                        max_length=EMBED_MAX_LEN, return_tensors="pt").to(embed_device)

    msg = check_abort_or_timeout(token)
    if msg:
        print(f"[ACTION] Aborting before retrieval: {msg}")
        return []

    with torch.no_grad():
        out = embed_model(**enc, return_dict=True).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        q_emb = vec.cpu().numpy()

    msg = check_abort_or_timeout(token)
    if msg:
        print(f"[ACTION] Aborting after embedding: {msg}")
        return []

    print("[ACTION] Searching FAISS for top documents...")
    sims, idxs = index.search(q_emb, k)
    results = []
    for i, s in zip(idxs[0], sims[0]):
        if 0 <= i < len(meta):
            results.append({"snippet": meta[i].get("text", "")})
    print(f"[ACTION] Retrieved {len(results)} candidate snippets.")
    return results

# generation with timeouts (60s)
def _generate_blocking(prompt: str) -> str:
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.inference_mode():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.7,
        )
    full_text = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text

def generate_response(message: str, token: "CancelToken") -> Generator[str, None, None]:
    fixed = get_fixed_response(message)
    if fixed is not None:
        print(f"[ACTION] FAST PATH matched. Returning canned response for: {message[:60]}...")
        yield fixed
        return

    print(f"[ACTION] RAG PATH start for: {message[:60]}...")
    yield "Retrieving relevant documents..."

    # retrieval
    sources = search(message, token)
    abort_msg = check_abort_or_timeout(token)
    if abort_msg and not sources:
        # Timed out or stopped before any result—just apologize/stop
        print(f"[ACTION] Retrieval aborted before results: {abort_msg}")
        yield abort_msg
        return

    # Always show retrieved snippets, regardless of what happens next
    snippet_block = format_snippets(sources)
    yield snippet_block

    # If we already timed out or user stopped, append apology/stop and exit
    abort_msg = check_abort_or_timeout(token)
    if abort_msg:
        print(f"[ACTION] Aborting after retrieval, before generation: {abort_msg}")
        yield abort_msg
        return

    if not sources or all(len(s.get("snippet", "")) < 50 for s in sources):
        print("[ACTION] No sufficiently informative sources found.")
        yield "Sorry, I do not have an answer at the moment."
        return

    # build prompt
    context = "\n\n".join([s["snippet"][:400] for s in sources[:MAX_SOURCES]])
    prompt = f"""You are a concise technical assistant specializing in the acoustoelectric effect.
Use only the sources below to answer.

Sources:
{context}

Question: {message}

Answer (be concise and technical):"""

    # Non-blocking generation start
    print("[ACTION] Starting generation...")
    q: Queue = Queue(maxsize=1)

    def run_gen():
        try:
            q.put((_generate_blocking(prompt), None))
        except Exception as e:
            q.put(("", e))

    t = threading.Thread(target=run_gen, daemon=True)
    t.start()

    # Poll-join: detach immediately on Stop; enforce hard timeout without freezing UI
    deadline = token.start_time + RAG_TIMEOUT_SECONDS
    while t.is_alive():
        if token.stopped():
            print("[ACTION] Stop pressed; detaching from generation thread and returning control to UI.")
            # Snippets are already shown; finish this turn with a clear stop message.
            yield "Response stopped by user."
            return
        if time.time() > deadline:
            print("[ACTION] Hard timeout reached during generation. Detaching.")
            yield APOLOGY_MSG
            return
        t.join(timeout=0.05)  # keep UI responsive

    try:
        full_text, err = q.get_nowait()
    except Exception as e:
        print(f"[ACTION] Queue error after generation: {e}")
        yield "Sorry, an internal error occurred during generation."
        return

    if err is not None:
        print(f"[ACTION] Generation error: {err}")
        yield "Sorry, an internal error occurred during generation."
        return

    print("[ACTION] Generation completed. Preparing to stream tokens to UI...")
    answer = full_text.split("Answer (be concise and technical):", 1)[-1].strip()

    # rate-limited streaming by words (first word shows immediately)
    limiter = RateLimiter(RATE_WORDS_PER_SEC)
    words = answer.split()
    current = snippet_block + "\n\n"  # keep snippets at the top of this assistant message
    first_word = True
    for word in words:
        abort_msg = check_abort_or_timeout(token)
        if abort_msg:
            print(f"[ACTION] Streaming aborted: {abort_msg}")
            # provide the apology/stop message after the partial answer
            yield current + f"\n\n{abort_msg}"
            return

        if first_word:
            first_word = False
        else:
            limiter.acquire(1)  # ~0.25 s between words

        current += word + " "
        yield current

    print("[ACTION] Streaming finished successfully.")
from gradio.events import Dependency

# GRADIO UI HANDLERS 
class CancelTokenState(gr.State):
    pass
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer  # for clarity only

def respond(message, history, token_state):
    if history is None:
        history = []

    if token_state is None:
        token_state = CancelToken()
    else:
        token_state.reset()

    if not message:
        return history, token_state

    history = history + [[message, None]]
    yield history, token_state

    for chunk in generate_response(message, token_state):
        history[-1][1] = chunk
        yield history, token_state

def retry_last(history, token_state):
    if history is None:
        history = []

    if token_state is None:
        token_state = CancelToken()
    else:
        token_state.reset()

    if not history:
        return history, token_state
    last_user_msg = history[-1][0]
    new_history = history[:-1] + [[last_user_msg, None]]
    yield new_history, token_state
    for chunk in generate_response(last_user_msg, token_state):
        new_history[-1][1] = chunk
        yield new_history, token_state

def stop_rag(history, token_state):
    # non-blocking Stop: return control to UI immediately, don't mutate the current turn
    if token_state is None:
        token_state = CancelToken()
    token_state.stop()
    print("[ACTION] Stop requested by user (non-blocking).")
    return history, token_state

def clear_chat(token_state):
    if token_state is None:
        token_state = CancelToken()
    else:
        token_state.reset()
    return [], token_state

with gr.Blocks(title="Acousto-Electric Effect RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Acousto-Electric Effect RAG")
    chatbot = gr.Chatbot(height=580, value=[])  # ensure history starts as []
    token_state = gr.State(value=None)

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask anything about the acoustoelectric effect...", scale=8)
        submit_btn = gr.Button("Submit", variant="primary", scale=2)

    with gr.Row():
        retry_btn = gr.Button("Retry Last", variant="secondary")
        stop_btn = gr.Button("Stop", variant="stop")
        clear_btn = gr.Button("Clear Chat", variant="secondary")

    gr.Examples([
        ["What is the acousto-electric effect?"],
        ["What is UAI?"],
        ["Tell me about applications in the heart"],
        ["What is the significance of the AE effect?"],
    ], inputs=msg)

    submit_btn.click(respond, [msg, chatbot, token_state], [chatbot, token_state])
    msg.submit(respond, [msg, chatbot, token_state], [chatbot, token_state])
    retry_btn.click(retry_last, [chatbot, token_state], [chatbot, token_state])
    stop_btn.click(stop_rag, [chatbot, token_state], [chatbot, token_state])
    clear_btn.click(clear_chat, [token_state], [chatbot, token_state])

print("✅ App started successfully! (Non-blocking Stop; RAG with pre-gen snippets; 1 word/0.25 s streaming)")
demo.launch(share=False)