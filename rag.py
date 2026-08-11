from huggingface_hub import snapshot_download
import os
import glob
import numpy as np
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

docs_path = snapshot_download(
    repo_id="err0rgod/indian_const",
    repo_type="dataset"
)

print("Documents downloaded to:", docs_path)

def chunk_text(text, chunk_size=150,overlap=20):
  words = text.split() #break the string into words
  chunks = []
  step = chunk_size-overlap #how far we sslide the window each time

  for start in range(0, len(words), step):
    chunk_words = words[start:start+chunk_size]
    if not chunk_words:
      break
    chunks.append(" ".join(chunk_words))
    if start + chunk_size >= len(words):
      break
  return chunks



# create embedding from chunks
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
DATA_PATH = docs_path
all_chunks = []
metadata = []
for filepath in glob.glob(os.path.join(DATA_PATH, "*.md")):
  with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()
  chunks = chunk_text(text)
  for i,chunk in enumerate(chunks):
    all_chunks.append(chunk)
    metadata.append({"source": filepath, "chunk_index": i})

chunk_embeddings = model.encode(all_chunks, normalize_embeddings=True)
print(chunk_embeddings.shape)


# take user input and convert into embeddings
query = "What does article 21 gurantees?"
query_embedding = model.encode(query, normalize_embeddings=True)

# calculate cosine similarity
scores =  chunk_embeddings @ query_embedding
# print(scores)

# return top k chunks
top_k = 10
top_indices = np.argsort(-scores)[:top_k]   # sort descending, take top 3
rag_context = ""
for idx in top_indices:
    rag_context += (all_chunks[idx])
    rag_context += "\n\n"


# api deepseek call

for chunk in completion(
    model="deepseek/deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a Legal assistant for indian system you will be given some data from the indian constitution related with the user's query. you have to give response in simple text no markdown format."},
        {"role": "system", "content": f"Given additional info: {rag_context}"},
        {"role": "user", "content": query}
    ],
    stream=True,
    max_tokens=500
):
    print(chunk.choices[0].delta.content or "API not working properly.")
