import json
from scripts.search_pipeline import search_query, hybrid_search

# Example: Test metadata filtering for intent = 'claim_process'

def test_metadata_filtering():
    query = "test query"
    filter_dict = {"intent": {"$eq": "claim_process"}}
    print("\n--- Testing Dense Search with Metadata Filter ---")
    dense_result = search_query(query, top_k=3, filter=filter_dict)
    print(f"Dense search returned {len(dense_result['results'])} results.")
    for doc in dense_result['results']:
        print(f"Dense: {doc['document_name']} | intent: {filter_dict['intent']['$eq']} | summary: {doc['summary'][:80]}")

    print("\n--- Testing Hybrid Search with Metadata Filter ---")
    hybrid_result = hybrid_search(query, top_k=3, filter=filter_dict)
    print(f"Hybrid search returned {len(hybrid_result['results'])} results.")
    for doc in hybrid_result['results']:
        print(f"Hybrid: {doc['document_name']} | intent: {filter_dict['intent']['$eq']} | summary: {doc['summary'][:80]}")

if __name__ == "__main__":
    test_metadata_filtering()
