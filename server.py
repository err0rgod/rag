from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import uvicorn
from litellm import completion
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import List, Dict

# Load environment variables
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize FastAPI App
app = FastAPI()

# Load Models
print("Loading Embedding Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(path="./qdrant_db")
print("Ready!")

# Data structure to accept from frontend
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    top_k = 10
    user_query = request.query
    
    # 1. Embed and Search Qdrant
    query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
    scores = client.query_points(collection_name='indian_constitution', query=query_embedding, limit=top_k)
    
    rag_context = ""
    for idx in scores.points:
        rag_context += idx.payload["text"] + "\n\n"
        
    # 2. Build the messages list
    messages = [
        {"role": "system", "content": "You are a Legal assistant for indian system you will be given some data from the indian constitution related with the user's query. you have to give response in simple text no markdown format. add a simple one line summary and example where needed"},
        {"role": "system", "content": f"Given additional info: {rag_context}"}
    ]
    
    # 3. Add the chat history that the frontend sends us!
    for msg in request.history:
        messages.append(msg)
        
    # 4. Add the current question
    messages.append({"role": "user", "content": user_query})
    
    # 5. Call DeepSeek (No stream this time so it's easier for the frontend to read)
    try:
        response = completion(
            model="deepseek/deepseek-v4-flash",
            messages=messages
        )
        model_response = response.choices[0].message.content
        return {"response": model_response}
    except Exception as e:
        return {"response": f"Network Error: {str(e)}"}

# This function serves the HTML file when you go to localhost:8000
@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
