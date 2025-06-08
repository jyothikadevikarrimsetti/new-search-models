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
<<<<<<< HEAD
index = pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
# stats = index.describe_index_stats()
print(index)

=======
else:
    #   pc.create_index(
    #     name=index_name,
    #     dimension=768,
    #     metric="cosine",
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     )
    # )
     pc.create_index(
        name=index_name,
        dimension=768,  # your dense vector dimension
        metric="dotproduct",  # required for hybrid
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            # pod_type="s1.x1"  # or another hybrid-capable pod type
        )
    )
>>>>>>> 9914bd6788a2ae7dc73974a15c37bc071852eb84
# index = Pinecone.Index(index_name)

# # Check available namespaces
# index_stats = index.describe_index_stats()
# print(index_stats["namespaces"]) 
# Now we can safely create the index connection
index = pc.Index(index_name)
