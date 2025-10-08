import os
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import cohere
# REMOVED: from cohere.core.errors import CohereAPIError # FIXES DEPLOYMENT CRASH
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

# --- DEPENDENCY INJECTION / LAZY INDEX INITIALIZATION ---
def get_pinecone_index():
    """Returns the Pinecone Index object, ensuring it exists."""
    try:
        # Parentheses fix is critical here
        if INDEX_NAME not in pc.list_indexes().names(): 
            pc.create_index(
                name=INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pc.Index(INDEX_NAME)
    
    except Exception as e: 
        print(f"[Pinecone Init Error] Initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Vector DB connection failed on startup/init.")


# --- MIDDLEWARE for logging ---
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        print("[ERROR] Internal exception:\n", traceback.format_exc())
        raise

# --- LOAD & INDEX TRAINING DATA (Runs once on service start) ---
def load_and_index_examples():
    """Load data and upsert to Pinecone."""
    try:
        index = get_pinecone_index()
    except HTTPException:
        print("⚠️ Skipping indexing due to Pinecone connection error during startup.")
        return
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Training data file not found, skipping indexing.")
        return

    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))

    texts = [d["text"] for d in docs]
    
    if not texts:
        print("No training texts to embed. Skipping upsert.")
        return
        
    embeds = co.embed(
        model="embed-english-v3.0",
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

# --- STARTUP EVENT ---
@app.on_event("startup")
def startup_event():
    load_and_index_examples()

# --- Pydantic model ---
class Req(BaseModel):
    text: str

# --- CLASSIFY ROUTE ---
@app.post("/classify")
def classify(req: Req, index = Depends(get_pinecone_index)):
    try:
        # 1. Embed query
        q_emb = co.embed(
            model="embed-english-v3.0",
            texts=[req.text],
            input_type="search_query"
        ).embeddings[0]

        # 2. Query Pinecone
        res = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
        docs = [
            {
                "text": match.metadata["text"],
                "label": match.metadata["label"],
                "score": match.score,
            }
            for match in res.matches
        ]

        # 3. Pass retrieved docs into Cohere Chat
        response = co.chat(
            model="command-r-plus",
            message=f"Classify this email:\n{req.text}\n\n"
                    f"Here are some examples:\n" +
                    "\n".join([f"- {d['text']} (label: {d['label']})" for d in docs])
        )

        label = response.text
        clean_label = label.lower().strip().split()[0].replace('.', '').replace(':', '')
        
        return {"label": clean_label, "examples": docs}

    # FINAL COHERE EXCEPTION CATCH
    except cohere.CohereError as e:  # Using the base exception from the cohere module
        print(f"[ERROR during classify - Cohere API] ▶ {repr(e)}")
        # If the key is invalid or a rate limit is hit, this will catch it
        raise HTTPException(status_code=500, detail="AI service (Cohere) failed. Check API key/limits.")
        
    # Catch any remaining Pinecone/Network errors
    except Exception as e:
        print(f"[ERROR during classify - Unhandled] ▶ {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during classification.")
