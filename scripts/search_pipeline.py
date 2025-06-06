from scripts.pinecone_utils import index
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv("config/.env")

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def search_query(query_text, top_k=1):
    start_time = time.time()

    # Step 1: Embed the query
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Step 2: Query Pinecone
    pinecone_result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    # print(pinecone_result)

    # Step 3: Re-rank using cosine similarity with summary embeddings
    summaries = [match.metadata.get("summary", "") for match in pinecone_result.matches]
    # ids = [match.id for match in pinecone_result.matches]
    # summary_dict = {match.id: match.metadata.get("summary", "") for match in pinecone_result.matches}
    # # Access individual summaries by ID:
    # for doc_id, summary in summary_dict.items():

    #     print(f"ID: {doc_id}, Summary: {summary}")



    summary_embeddings = model.encode(summaries, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, summary_embeddings)[0]

    # Step 4: Find the best match
    top_idx = cosine_scores.argmax().item()
    top_summary = summaries[top_idx]

    # Step 5: Display only the final answer and time
    elapsed = time.time() - start_time
    print(f"\n✅ Top Answer: {top_summary}")
    print(f"⏱️  Search Time: {elapsed:.2f} seconds")
