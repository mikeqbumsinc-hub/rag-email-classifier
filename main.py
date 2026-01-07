import os
import json
import traceback
import time  # NEW: Required for rate-limit protection
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import cohere
from pinecone import Pinecone, ServerlessSpec 

# --- CONFIGURATION ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "email-classifier-index") 
TRAIN_DATA_PATH = "cohere_training_data.jsonl"
TOP_K = 3

# --- INITIALIZE CLIENTS ---
app = FastAPI()
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_pinecone_index():
    try:
        if INDEX_NAME not in pc.list_indexes().names(): 
            pc.create_index(
                name=INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pc.Index(INDEX_NAME)
    except Exception as e: 
        print(f"[Pinecone Init Error] {e}")
        raise HTTPException(status_code=500, detail="Vector DB connection failed.")

@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        print("[ERROR] Internal exception:\n", traceback.format_exc())
        raise

# --- LOAD & INDEX TRAINING DATA (Throttled for Trial Keys) ---
def load_and_index_examples():
    """Load and index with rate-limit protection to avoid 429 errors."""
    try:
        index = get_pinecone_index()
    except Exception:
        print("⚠️ Skipping indexing: Pinecone error.")
        return
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Training data file not found.")
        return

    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))

    if not docs:
        return

    # RATE LIMIT PROTECTION: Process in small batches
    # Trial keys prefer smaller batches to avoid "burst" limits
    BATCH_SIZE = 10 
    all_embeddings = []
    print(f"🔄 Processing {len(docs)} examples in batches...")

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_texts = [d["text"] for d in batch]
        
        try:
            res = co.embed(
                model="embed-english-v3.0",
                texts=batch_texts,
                input_type="search_document"
            )
            all_embeddings.extend(res.embeddings)
            print(f"✅ Embedded batch {i//BATCH_SIZE + 1}")
            
            # Wait 12 seconds between batches to stay safely under trial limits
            time.sleep(12) 
        except Exception as e:
            print(f"❌ Batch embedding failed at index {i}: {e}")
            continue

    if len(all_embeddings) == len(docs):
        to_upsert = []
        for i, d in enumerate(docs):
            meta = {"text": d["text"], "label": d["label"]}
            to_upsert.append((str(i), all_embeddings[i], meta))
        
        index.upsert(vectors=to_upsert)
        print(f"🚀 Successfully indexed {len(to_upsert)} examples!")

@app.on_event("startup")
def startup_event():
    load_and_index_examples()

class Req(BaseModel):
    text: str

@app.post("/classify")
def classify(req: Req, index = Depends(get_pinecone_index)):
    try:
        # TRUNCATION FIX: Limit input text to 4000 chars to avoid Error 400
        safe_text = req.text[:4000] 
        
        # 1. Embed query
        q_emb = co.embed(
            model="embed-english-v3.0",
            texts=[safe_text],
            input_type="search_query"
        ).embeddings[0]

        # 2. Query Pinecone
        res = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
        docs = [
            {"text": match.metadata["text"], "label": match.metadata["label"], "score": match.score}
            for match in res.matches
        ]

        # 3. Chat classification
        response = co.chat(
            model="command-r-plus-08-2024", 
            message=(
                f"Classify this email. Respond with ONLY 'warm', 'cold', or 'spam'."
                f"\n\n--- EMAIL ---\n{safe_text}\n"
                f"\n--- EXAMPLES ---\n"
                + "\n".join([f"- {d['text']} ({d['label']})" for d in docs])
            )
        )

        clean_label = response.text.lower().strip()
        
        # Consistent label logic for Make.com Router
        if 'warm' in clean_label:
            final_label = 'warm'
        elif 'spam' in clean_label:
            final_label = 'spam'
        else:
            final_label = 'cold'

        return {"label": final_label, "examples": docs}

    except Exception as e:
        print(f"[ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
