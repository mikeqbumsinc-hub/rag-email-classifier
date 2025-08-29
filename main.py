import os
import json
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
from dotenv import load_dotenv
from pinecone import Pinecone  # <- Updated import

load_dotenv()

# Config from env
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "email-leads")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
TOP_K = int(os.getenv("TOP_K", "5"))
COHERE_MODEL = os.getenv("COHERE_MODEL", "xlarge")

if not all([COHERE_API_KEY, PINECONE_API_KEY]):
    raise RuntimeError("Missing one of COHERE_API_KEY or PINECONE_API_KEY")

# Initialize Cohere
co = cohere.Client(COHERE_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM)

# Reference the index
index = pc.Index(PINECONE_INDEX)

app = FastAPI(title="RAG Classifier API")

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    score: float = None
    neighbors: List[Dict[str, Any]] = []

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = co.embed(texts=texts, model=f"{COHERE_MODEL}-embed")
    return resp.embeddings

@app.on_event("startup")
def startup_upsert_training():
    TRAIN_FILE = os.getenv("TRAIN_FILE", "training_data.jsonl")
    if not os.path.exists(TRAIN_FILE):
        logging.warning(f"No training file found at {TRAIN_FILE}, skipping index setup.")
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

    BATCH = 32
    vectors = []
    for i in range(0, len(examples), BATCH):
        batch = examples[i : i + BATCH]
        texts = [e["text"] for e in batch]
        embeddings = embed_texts(texts)
        for e, v in zip(batch, embeddings):
            vectors.append((e["id"], v, {"label": e["label"], "text": e["text"]}))

    index.upsert(vectors=vectors)
    logging.info(f"Upserted {len(vectors)} training examples.")

def query_neighbors(embedding: List[float], top_k: int = TOP_K):
    resp = index.query(vector=embedding, top_k=top_k, include_metadata=True, include_values=False)
    matches = []
    for m in resp.get("matches", []):
        matches.append({
            "id": m["id"],
            "score": m["score"],
            "label": m["metadata"].get("label"),
            "text": m["metadata"].get("text")
        })
    return matches

def build_prompt(neighbors: List[Dict[str, Any]], incoming_text: str) -> str:
    lines = [
        "You are an assistant that classifies incoming email messages into one of three labels: warm, cold, or spam.",
        "Return only the single-word label in lowercase (warm, cold, or spam). Do not add extra text.",
        "\nExamples:",
    ]
    for n in neighbors:
        example = n["text"].replace("\n", " ").strip()
        lines.append(f"Label: {n['label']}\nEmail: {example}\n")
    lines += ["\nNow classify this email:", f"Email: {incoming_text.replace(chr(10), ' ')}\n\nLabel:"]
    return "\n".join(lines)

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text field is required")

    try:
        emb = embed_texts([text])[0]
    except Exception as e:
        logging.exception("Embedding failed")
        raise HTTPException(status_code=500, detail="Embedding failed")

    neighbors = query_neighbors(emb)
    label_counts = {}
    for n in neighbors:
        lbl = n.get("label")
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    majority_label = None
    if label_counts:
        sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_counts) == 1 or sorted_counts[0][1] > sorted_counts[1][1]:
            majority_label = sorted_counts[0][0]

    if majority_label:
        avg_score = sum(n["score"] for n in neighbors if n.get("label") == majority_label) / max(1, sum(1 for n in neighbors if n.get("label") == majority_label))
        return ClassifyResponse(label=majority_label, score=avg_score, neighbors=neighbors)

    prompt = build_prompt(neighbors, text)
    try:
        gen = co.generate(
            model=COHERE_MODEL,
            prompt=prompt,
            max_tokens=4,
            temperature=0.0,
            stop_sequences=["\n"]
        )
        raw_label = gen.generations[0].text.strip().lower()
        final = next((c for c in ["warm", "cold", "spam"] if c in raw_label), raw_label.split()[0] if raw_label else "cold")
    except:
        final = "cold"

    avg_score_all = sum(n["score"] for n in neighbors) / max(1, len(neighbors))
    return ClassifyResponse(label=final, score=avg_score_all, neighbors=neighbors)
