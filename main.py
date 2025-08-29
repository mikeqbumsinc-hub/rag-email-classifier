import os
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
import pinecone
from dotenv import load_dotenv

load_dotenv()

# Config from env
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west1-gcp"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "email-leads")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # depends on Cohere model
TOP_K = int(os.getenv("TOP_K", "5"))
COHERE_MODEL = os.getenv("COHERE_MODEL", "xlarge")  # adjust if needed

if not all([COHERE_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise RuntimeError("Missing one of COHERE_API_KEY, PINECONE_API_KEY, or PINECONE_ENV")

# Init clients
co = cohere.Client(COHERE_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

app = FastAPI(title="RAG Classifier API")

# Ensure index exists
if PINECONE_INDEX not in pinecone.list_indexes():
    # Create with default dims; make sure EMBED_DIM matches Cohere output
    pinecone.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM)
index = pinecone.Index(PINECONE_INDEX)


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    score: float = None
    neighbors: List[Dict[str, Any]] = []


def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = co.embed(texts=texts, model=f"{COHERE_MODEL}-embed")  # model name convention
    return resp.embeddings


@app.on_event("startup")
def startup_upsert_training():
    """
    On startup, load training_data.jsonl, embed in batches and upsert to Pinecone.
    Ids are stable (e.g., 'train-0', 'train-1' ...). Metadata stores label and text.
    """
    TRAIN_FILE = os.getenv("TRAIN_FILE", "training_data.jsonl")
    if not os.path.exists(TRAIN_FILE):
        logging.warning(f"No training file found at {TRAIN_FILE} — starting with empty index.")
        return

    # Load lines
    examples = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                text = obj.get("text")
                label = obj.get("label")
                if not text or not label:
                    continue
                examples.append({"id": f"train-{i}", "text": text, "label": label})
            except json.JSONDecodeError:
                continue

    if not examples:
        logging.warning("No valid training examples found.")
        return

    # Batch embed and upsert
    BATCH = 32
    vectors = []
    for i in range(0, len(examples), BATCH):
        batch = examples[i : i + BATCH]
        texts = [e["text"] for e in batch]
        emb = embed_texts(texts)
        for e, v in zip(batch, emb):
            vectors.append((e["id"], v, {"label": e["label"], "text": e["text"]}))

    # Upsert (Pinecone expects list of tuples)
    index.upsert(vectors=vectors)
    logging.info(f"Upserted {len(vectors)} training examples to Pinecone index '{PINECONE_INDEX}'.")


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
    """
    Create a small prompt that gives examples and asks the model to return only one of:
    warm, cold, spam
    """
    prompt_lines = [
        "You are an assistant that classifies incoming email messages into one of three labels: warm, cold, or spam.",
        "Return only the single-word label in lowercase (warm, cold, or spam). Do not add extra text."
    ]
    prompt_lines.append("\nExamples:")
    for n in neighbors:
        # keep example short
        example = n["text"].replace("\n", " ").strip()
        prompt_lines.append(f"Label: {n['label']}\nEmail: {example}\n")

    prompt_lines.append("\nNow classify this email:")
    prompt_lines.append(f"Email: {incoming_text.replace(chr(10), ' ')}\n\nLabel:")
    return "\n".join(prompt_lines)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text field is required")

    try:
        emb = embed_texts([text])[0]
    except Exception as e:
        logging.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    neighbors = query_neighbors(emb, top_k=TOP_K)

    # If neighbors have a clear majority, quick-route without calling generation
    label_counts = {}
    for n in neighbors:
        lbl = n.get("label")
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # choose majority if exists (strict majority)
    majority_label = None
    if label_counts:
        # sort by count then choose top
        sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        top_label, top_count = sorted_items[0]
        # if top_count > others (strict)
        if len(sorted_items) == 1 or top_count > sorted_items[1][1]:
            majority_label = top_label

    if majority_label:
        # compute a simple score average from neighbors for transparency
        avg_score = sum(n["score"] for n in neighbors if n.get("label") == majority_label) / max(1, sum(1 for n in neighbors if n.get("label") == majority_label))
        return ClassifyResponse(label=majority_label, score=avg_score, neighbors=neighbors)

    # Else build a small prompt and call Cohere generate to break tie / confirm
    prompt = build_prompt(neighbors, text)
    try:
        gen = co.generate(
            model=COHERE_MODEL,
            prompt=prompt,
            max_tokens=4,
            temperature=0.0,
            stop_sequences=["\n"]
        )
        # The model text should be the label
        raw_label = gen.generations[0].text.strip().lower()
        # sanitize label
        for candidate in ["warm", "cold", "spam"]:
            if candidate in raw_label:
                final_label = candidate
                break
        else:
            final_label = raw_label.split()[0] if raw_label else "cold"
    except Exception as e:
        logging.exception("Cohere generation failed")
        final_label = "cold"  # fallback

    # Provide a score estimate (average neighbor score)
    avg_score_all = sum(n["score"] for n in neighbors) / max(1, len(neighbors))
    return ClassifyResponse(label=final_label, score=avg_score_all, neighbors=neighbors)
