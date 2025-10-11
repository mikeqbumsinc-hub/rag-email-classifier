import os
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import cohere
from pinecone import Pinecone, ServerlessSpec 

# --- CONFIGURATION ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# CORRECTED: Using PINECONE_INDEX to match your .env file
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
        # FIX: Using .names() method and relying on Exception catch for robustness
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
        model="embed-english-v3.0", # FIX: Using correct hyphenated model name
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
        print("INFO: Starting Cohere embedding call...")
        
        # 1. Embed query
        q_emb = co.embed(
            model="embed-english-v3.0", # FIX: Using correct hyphenated model name
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

        # 3. Pass retrieved docs into Cohere Chat - IMPROVED PROMPT
        response = co.chat(
            # FIX: Using supported versioned chat model
            model="command-r-plus-08-2024", 
            message=(
                f"You are an expert email classifier. Your task is to classify the following email."
                f"Based on the provided examples (RAG results), respond with ONLY ONE WORD: 'warm', 'cold', or 'spam'."
                f"Do NOT include any other text, punctuation, or explanation."
                f"\n\n--- EMAIL TO CLASSIFY ---\n{req.text}\n"
                f"\n--- CLASSIFICATION EXAMPLES ---\n"
                + "\n".join([f"- {d['text']} (Label: {d['label']})" for d in docs])
            )
        )

        label = response.text
        
        # POST-PROCESSING: Ensures the label is a clean, predictable string for Make.com Router
        clean_label = label.lower().strip()
        
        if 'warm' in clean_label:
            final_label = 'warm'
        elif 'cold' in clean_label:
            final_label = 'cold'
        elif 'spam' in clean_label:
            final_label = 'spam'
        else:
            final_label = 'cold' # Default fallback if the model fails to follow instructions

        # SUCCESS: Returns classification label and supporting examples
        return {"label": final_label, "examples": docs}

    # ROBUST EXCEPTION CATCH
    except Exception as e:
        error_detail = str(e)
        
        if "api_key" in error_detail.lower() or "unauthorized" in error_detail.lower() or "limit" in error_detail.lower():
            print(f"[CRITICAL ERROR] Cohere API Key/Limit Failure: {error_detail}")
            raise HTTPException(status_code=500, detail="AI Service Failed: Check API Key and Usage Limits.")
        
        # All other unhandled exceptions
        print(f"[ERROR during classify - Unhandled] ▶ {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during classification.")
