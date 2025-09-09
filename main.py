import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
import pinecone

# --- CONFIGURATION ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
TRAIN_DATA_PATH = "cohere_training_data.jsonl"
TOP_K = 3

# --- INITIALIZE CLIENTS ---
app = FastAPI()
co = cohere.ClientV2(api_key=COHERE_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

# --- LOAD & INDEX EXAMPLES ---
def load_and_index_examples():
    docs = []
    with open(TRAIN_DATA_PATH, "r") as f:
        for ln in f:
            docs.append(json.loads(ln))
    texts = [d["text"] for d in docs]
    embeds = co.embed(model="embed-vanilla-002", texts=texts,
                      input_type="search_document", embedding_types=["float"]).embeddings
    to_upsert = []
    for i, d in enumerate(docs):
        meta = {"text": d["text"], "label": d["label"]}
        to_upsert.append((str(i), embeds[i], meta))
    index.upsert(vectors=to_upsert)
    print(f"Upserted {len(to_upsert)} examples to Pinecone index.")

load_and_index_examples()

# --- Pydantic model ---
class Req(BaseModel):
    text: str

# --- CLASSIFY ROUTE ---
@app.post("/classify")
def classify(req: Req):
    try:
        # Embed query
        q_emb = co.embed(model="embed-vanilla-002", texts=[req.text],
                         input_type="query", embedding_types=["float"]).embeddings
        res = index.query(vector=q_emb[0], top_k=TOP_K, include_metadata=True)
        docs = [{"text":match.metadata["text"], "label":match.metadata["label"], "score":match.score}
                for match in res["matches"]]
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": f"Classify this email: {req.text}"}],
            documents=[{"data": {"text": doc["text"], "label": doc["label"]}} for doc in docs]
        )
        label = response.message.content[0].text.strip()
        return {"label": label, "examples": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
