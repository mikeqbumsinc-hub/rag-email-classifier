import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
# ... your other imports (cohere, Pinecone, etc.)

app = FastAPI(debug=True)

@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        print("[ERROR] Internal exception:\n", traceback.format_exc())
        raise  # to allow FastAPI / default error handler to respond

class Req(BaseModel):
    text: str

@app.post("/classify")
async def classify(req: Req):
    try:
        # existing classify logic
    except Exception as e:
        print("[ERROR during classify] ▶", repr(e))
        raise HTTPException(status_code=500, detail="Internal error occurred")
        
import os
import json
from fastapi import FastAPI, HTTPException
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
app = FastAPI()
co = cohere.ClientV2(api_key=COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn’t exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Cohere/OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # can be changed if needed
        )
    )

index = pc.Index(INDEX_NAME)

# --- LOAD & INDEX EXAMPLES ---
def load_and_index_examples():
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Training data file not found, skipping indexing.")
        return
    
    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))
    
    texts = [d["text"] for d in docs]
    embeds = co.embed(
        model="embed-vanilla-002",
        texts=texts,
        input_type="search_document",
        embedding_types=["float"]
    ).embeddings

    to_upsert = []
    for i, d in enumerate(docs):
        meta = {"text": d["text"], "label": d["label"]}
        to_upsert.append((str(i), embeds[i], meta))
    
    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"✅ Upserted {len(to_upsert)} examples to Pinecone index.")

load_and_index_examples()

# --- Pydantic model ---
class Req(BaseModel):
    text: str

# --- CLASSIFY ROUTE ---
@app.post("/classify")
def classify(req: Req):
    try:
        # Embed query
        q_emb = co.embed(
            model="embed-vanilla-002",
            texts=[req.text],
            input_type="query",
            embedding_types=["float"]
        ).embeddings

        res = index.query(vector=q_emb[0], top_k=TOP_K, include_metadata=True)
        docs = [
            {"text": match.metadata["text"], "label": match.metadata["label"], "score": match.score}
            for match in res["matches"]
        ]

        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": f"Classify this email: {req.text}"}],
            documents=[{"data": {"text": doc["text"], "label": doc["label"]}} for doc in docs]
        )

        label = response.message.content[0].text.strip()
        return {"label": label, "examples": docs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
