import os
import json
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import cohere
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "email-classifier-index")
TRAIN_DATA_PATH = "cohere_training_data.jsonl"
TOP_K = 3

# --- INITIALIZE CLIENTS ---
app = FastAPI(debug=True)
co = cohere.ClientV2(api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Middleware to log server-side errors
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        print("[ERROR] Internal exception:\n", traceback.format_exc())
        raise  # Let FastAPI handle the error response

# --- Ensure Pinecone index exists ---
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # ✅ v3 embeddings are 1024 dims
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # adjust if needed
        )
    )

index = pc.Index(INDEX_NAME)

# --- LOAD & INDEX TRAINING DATA ---
def load_and_index_examples():
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Training data file not found, skipping indexing.")
        return
    
    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))
    
    texts = [d["text"] for d in docs]

    # ✅ Use Cohere v3 embeddings
    embeds = co.embed(
        model="embed-english-v3.0",      # or "embed-multilingual-v3.0"
        texts=texts,
        input_type="search_document"
    ).embeddings

    to_upsert = []
    for i, d in enumerate(docs):
        meta = {"text": d["text"], "label": d["label"]}
        to_upsert.append((str(i), embeds[i], meta))
    
    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"✅ Upserted {len(to_upsert)} examples to Pinecone index.")

load_and_index_examples()

# --- REQUEST MODEL ---
class Req(BaseModel):
    text: str

# --- CLASSIFY ROUTE ---
@app.post("/classify")
def classify(req: Req):
    try:
        # ✅ Embed query
        q_emb = co.embed(
            model="embed-english-v3.0",   # or "embed-multilingual-v3.0"
            texts=[req.text],
            input_type="search_query"
        ).embeddings

        res = index.query(vector=q_emb[0], top_k=TOP_K, include_metadata=True)
        docs = [
            {"text": match.metadata["text"], "label": match.metadata["label"], "score": match.score}
            for match in res["matches"]
        ]

        # Use Cohere chat to refine classification
        response = co.chat(
            model="command-r-plus",   # ✅ stable v3 chat model
            messages=[{"role": "user", "content": f"Classify this email into a single label only: {req.text}"}],
            documents=[{"data": {"text": doc["text"], "label": doc["label"]}} for doc in docs]
        )

        # Extract label cleanly
        label = response.message.content[0].text.strip()
        return {"label": label, "examples": docs}
    
    except Exception as e:
        print("[ERROR during classify] ▶", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
