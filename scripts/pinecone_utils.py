from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv("config/.env")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment='us-east-1-aws'
)

index_name = os.getenv("PINECONE_INDEX_NAME")

# Delete if already exists (optional)
if index_name in pc.list_indexes().names():
    # pc.delete_index(index_name)
    pass

# Create the index (UNCOMMENTED)
else:
     pc.create_index(
        name=index_name,
        dimension=1536,  # your dense vector dimension
        metric="dotproduct",  # required for hybrid
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            # pod_type="s1.x1"  # or another hybrid-capable pod type
        )
    )

index = pc.Index(index_name)
