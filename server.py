from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import os
import uvicorn
from litellm import completion
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from pydantic import BaseModel
from typing import List, Dict
import fitz
import uuid

# Load environment variables
load_dotenv()
for key in ("DEEPSEEK_API_KEY", "HF_TOKEN"):
    if value := os.getenv(key):
        os.environ[key] = value

# Initialize FastAPI App
app = FastAPI()

# Load Models
print("Loading Embedding Model...")
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="./fastembed_cache",
    threads=1,
)
client = QdrantClient(path="./qdrant_db")
if not client.collection_exists(collection_name="indian_constitution"):
    client.create_collection(
        collection_name="indian_constitution",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
print("Ready!")

# ----------------- HELPER FUNCTIONS -----------------
def chunk_text(text, chunk_size=150, overlap=20):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start+chunk_size]
        if not chunk_words: break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words): break
    return chunks

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

# ----------------- ENDPOINTS -----------------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Read the uploaded PDF file directly from memory
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        
    # 2. Chunk it using the exact same logic we used before
    chunks = chunk_text(text)
    
    # 3. Embed and save to Qdrant using random UUIDs so we don't overwrite the constitution!
    points = []
    for chunk, embedding in zip(chunks, model.embed(chunks)):
        points.append(
            PointStruct(
                id=uuid.uuid4().hex,  # Random ID 
                vector=embedding.tolist(),
                payload={"text": chunk, "source": file.filename} # Save filename!
            )
        )
        
    client.upsert(collection_name='indian_constitution', points=points)
    return {"message": f"Successfully processed {file.filename}!"}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    top_k = 10
    user_query = request.query
    
    # 1. Search Database
    query_embedding = next(model.embed([user_query])).tolist()
    scores = client.query_points(collection_name='indian_constitution', query=query_embedding, limit=top_k)
    
    rag_context = ""
    for idx in scores.points:
        rag_context += f"Source: {idx.payload.get('source', 'Unknown')}\n{idx.payload['text']}\n\n"
        
    # 2. Generalized System Prompt (Not just for Indian Constitution anymore!)
    messages = [
        {"role": "system", "content": "You are a helpful AI Document Assistant. You will be given some data extracted from uploaded documents related to the user's query. Answer the query based ONLY on the provided context. If the context does not contain the answer, say so. Do not use markdown format. Add a simple one line summary and example where needed."},
        {"role": "system", "content": f"Given additional info:\n{rag_context}"}
    ]
    
    for msg in request.history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": user_query})
    
    try:
        response = completion(
            model="deepseek/deepseek-v4-flash",
            messages=messages
        )
        model_response = response.choices[0].message.content
        return {"response": model_response}
    except Exception as e:
        return {"response": f"Network Error: {str(e)}"}


@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
