import os
import json
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# — Environment Config —
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "email-leads")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Supported Cohere model names
DEFAULT_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBED_INPUT_TYPE = os.getenv("COHERE_EMBED_INPUT_TYPE", "search_document")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not all([COHERE_API_KEY, PINECONE_API_KEY]):
    raise RuntimeError("Missing COHERE_API_KEY or PINECONE_API_KEY")

# Cohere client
co = cohere.Client(COHERE_API_KEY)

# Pinecone client + index setup
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
index = pc.Index(PINECONE_INDEX)

app = FastAPI(title="RAG Email Classifier API")

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    score: float = None
    neighbors: List[Dict[str, Any]] = []

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = co.embed(
        texts=texts,
        model=DEFAULT_EMBED_MODEL,
        input_type=EMBED_INPUT_TYPE
    )
    return resp.embeddings

@app.on_event("startup")
def startup_upsert_training():
    TRAIN_FILE = os.getenv("TRAIN_FILE", "training_data.jsonl")
    if not os.path.exists(TRAIN_FILE):
        logging.warning(f"No training data file at {TRAIN_FILE}, skipping upload.")
        return

    examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            text = obj.get("text")
            label = obj.get("label")
            if text and label:
                examples.append({"id": f"train-{i}", "text": text, "label": label})

    if not examples:
        logging.warning("No valid training examples found.")
        return

    vectors = []
    for ex in examples:
        embed = embed_texts([ex["text"]])[0]
        vectors.append((ex["id"], embed, {"label": ex["label"], "text": ex["text"]}))

    index.upsert(vectors=vectors)
    logging.info(f"Upserted {len(vectors)} training examples.")

def query_neighbors(emb: List[float], top_k: int = TOP_K):
    resp = index.query(vector=emb, top_k=top_k, include_metadata=True, include_values=False)
    return [
        {"id": m["id"], "score": m["score"], "label": m["metadata"].get("label"), "text": m["metadata"].get("text")}
        for m in resp.get("matches", [])
    ]

def build_prompt(neighbors: List[Dict[str, Any]], text: str) -> str:
    lines = [
        "You are an assistant that classifies incoming emails as warm, cold, or spam.",
        "Respond with only the lowercase label (warm, cold, spam).",
        "\nExamples:"
    ]
    for n in neighbors:
        example = n["text"].replace("\n", " ").strip()
        lines.append(f"Label: {n['label']}\nEmail: {example}\n")
    lines.extend(["\nClassify this:", f"Email: {text.strip().replace(chr(10), ' ')}\n\nLabel:"])
    return "\n".join(lines)

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    try:
        emb = embed_texts([text])[0]
    except Exception:
        logging.exception("Embedding error")
        raise HTTPException(status_code=500, detail="Embedding failed")

    neighbors = query_neighbors(emb)
    label_counts = {}
    for n in neighbors:
        lbl = n.get("label")
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    well_label = None
    if label_counts:
        sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) == 1 or sorted_items[0][1] > sorted_items[1][1]:
            well_label = sorted_items[0][0]

    if well_label:
        avg_score = sum(n["score"] for n in neighbors if n.get("label") == well_label) / max(1, sum(1 for n in neighbors if n.get("label") == well_label))
        return ClassifyResponse(label=well_label, score=avg_score, neighbors=neighbors)

    prompt = build_prompt(neighbors, text)
    try:
        gen = co.generate(model=COHERE_EMBED_MODEL, prompt=prompt, max_tokens=4, temperature=0.0, stop_sequences=["\n"])
        raw = gen.generations[0].text.strip().lower()
        chosen = next((c for c in
