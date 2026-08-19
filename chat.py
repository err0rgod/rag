
import os

import numpy as np
from litellm import completion
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



# load environment variables
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def chat():
    top_k = 10
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = QdrantClient(path="./qdrant_db")

    # user cat loop 
    try:
        while(True):
            chat_history =""
            user_query = input("Enter your query: ")
            chat_history += "User: "
            chat_history += user_query

            query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
            # calculate cosine sim
            scores = client.query_points(collection_name='indian_constitution', query=query_embedding,limit=top_k)

            rag_context = ""
            for idx in scores.points:
               rag_context += idx.payload["text"]
               rag_context += "\n\n"
            Model_response = ""
            # deepseek api call
            for chunk in completion(
               model="deepseek/deepseek-v4-flash",
               messages=[
                    {"role": "system", "content": "You are a Legal assistant for indian system you will be given some data from the indian constitution related with the user's query. you have to give response in simple text no markdown format. add a simple one line summary and example where needed"},
                    {"role": "system", "content": f"Given additional info: {rag_context}"},
                    {"role": "user", "content": chat_history}
               ],
               stream=True
            ):
               print(chunk.choices[0].delta.content or "", end="")
               Model_response += chunk.choices[0].delta.content or ""
            print("\n")
            chat_history += "Model : "
            chat_history += Model_response
    except KeyboardInterrupt:
        print("\n\n Force Quiting... Goodbye")


        
if __name__ == "__main__":
   chat()