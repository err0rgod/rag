from huggingface_hub import snapshot_download
import os
import glob
import numpy as np
from litellm import completion
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



# load environment variables
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Download data
# def data_load() -> str:
#     docs_path = snapshot_download(
#         repo_id="err0rgod/indian_const",
#         repo_type="dataset"
#     )
#     return docs_path


# def chunk_text(text, chunk_size=150,overlap=20):
#   words = text.split() #break the string into words
#   chunks = []
#   step = chunk_size-overlap #how far we sslide the window each time

#   for start in range(0, len(words), step):
#     chunk_words = words[start:start+chunk_size]
#     if not chunk_words:
#       break
#     chunks.append(" ".join(chunk_words))
#     if start + chunk_size >= len(words):
#       break
#   return chunks


def chat():
    top_k = 10
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = QdrantClient(path="./qdrant_db")

    # user cat loop 
    while(True):
        user_query = input("Enter your query: ")
        query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
        # calculate cosine sim
        scores = client.query_points(collection_name='indian_constitution', query=query_embedding,limit=top_k)

        rag_context = ""
        for idx in scores.points:
           rag_context += idx.payload["text"]
           rag_context += "\n\n"

        # deepseek api call
        for chunk in completion(
           model="deepseek/deepseek-v4-flash",
           messages=[
                {"role": "system", "content": "You are a Legal assistant for indian system you will be given some data from the indian constitution related with the user's query. you have to give response in simple text no markdown format. add a simple one line summary and example where needed"},
                {"role": "system", "content": f"Given additional info: {rag_context}"},
                {"role": "user", "content": user_query}
           ],
           stream=True
        ):
           print(chunk.choices[0].delta.content or "", end="")
        print("\n")
           
if __name__ == "__main__":
   chat()