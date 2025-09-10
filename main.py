import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import cohere
from pinecone import Pinecone

# --- Load environment variables ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "email-classifier-index")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "cohere_training_data.jsonl")
TOP_K = int(os.getenv("TOP_K", 5))

# --- Init clients ---
co = cohere.Client(api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

app = FastAPI()


# --- Request Model ---
class Req(BaseModel):
    text: str


# --- Load & Index Training Data ---
def load_and_index_examples():
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Training data file not found, skipping indexing.")
        return

    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))

    texts = [d["text"] for d in docs]

    # ✅ Cohere v4 embeddings return object → use .float
    embeds = co.embed(
        model="embed-english-v3.0",  # or embed-multilingual-v3.0 if needed
        texts=texts,
        input_type="search_document"
    ).embeddings.float

    to_upsert = []
    for i, d in enumerate(docs):
        meta = {"text": d["text"], "label": d["label"]}
        to_upsert.append((str(i), embeds[i], meta))

    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"✅ Upserted {len(to_upsert)} examples to Pinecone index.")


# Run indexing on startup
@app.on_event("startup")
def startup_event():
    load_and_index_examples()


# --- Classify Route ---
@app.post("/classify")
def classify(req: Req):
    try:
        # ✅ Embed query properly with .float
        q_emb = co.embed(
            model="embed-english-v3.0",
            texts=[req.text],
            input_type="search_query"
        ).embeddings.float[0]

        res = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
        docs = [
            {"text": match.metadata["text"], "label": match.metadata["label"], "score": match.score}
            for match in res["matches"]
        ]

        # Use Cohere chat to refine classification
        response = co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": f"Classify this email into a single label only: {req.text}"}],
            documents=[{"data": {"text": doc["text"], "label": doc["label"]}} for doc in docs]
        )

        label = response.message.content[0].text.strip()
        return {"label": label, "examples": docs}

    except Exception as e:
        print("[ERROR during classify] ▶", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Health Check Route ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Email Classifier API is running!"}
