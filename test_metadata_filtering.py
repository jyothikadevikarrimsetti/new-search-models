# test_metadata_filtering.py
from scripts.search_pipeline import search_query, hybrid_search
from scripts.pinecone_utils import index
result = index.fetch(ids=["Jimson_Ratnam_JavaFullStackDeveloper_2+years_chunk1"], namespace="__default__")
print(result.vectors["Jimson_Ratnam_JavaFullStackDeveloper_2+years_chunk1"].metadata)

# def print_metadata_results(results, field="intent"):
#     print(f"\nResults (showing '{field}' metadata):")
#     for i, doc in enumerate(results.get('results', []), 1):
#         print(f"{i}. Document: {doc.get('document_name')}")
#         print(f"   Summary: {doc.get('summary', '')[:100]}")
#         print(f"   Metadata: {field} = {doc.get(field, '[not present]')}")

# if __name__ == "__main__":
#     # Try a value you know exists
#     filter_val = "policy_info"  # Change to a value you know is present in your data
#     print(f"\n--- DENSE SEARCH with metadata_filter: intent == '{filter_val}' ---")
#     dense_result = search_query("test query", top_k=5, metadata_filter={"intent": {"$eq": filter_val}})
#     print_metadata_results(dense_result, field="intent")
#     print(f"Dense LLM Answer: {dense_result.get('answer')}")

#     print(f"\n--- HYBRID SEARCH with metadata_filter: intent == '{filter_val}' ---")
#     hybrid_result = hybrid_search("test query", top_k=5, metadata_filter={"intent": {"$eq": filter_val}})
#     print_metadata_results(hybrid_result, field="intent")
#     print(f"Hybrid LLM Answer: {hybrid_result.get('answer')}")

#     # Try a value you know does NOT exist
#     filter_val = "nonexistent_intent"
#     print(f"\n--- DENSE SEARCH with metadata_filter: intent == '{filter_val}' ---")
#     dense_result = search_query("test query", top_k=5, metadata_filter={"intent": {"$eq": filter_val}})
#     print_metadata_results(dense_result, field="intent")
#     print(f"Dense LLM Answer: {dense_result.get('answer')}")

#     print(f"\n--- HYBRID SEARCH with metadata_filter: intent == '{filter_val}' ---")
#     hybrid_result = hybrid_search("test query", top_k=5, metadata_filter={"intent": {"$eq": filter_val}})
#     print_metadata_results(hybrid_result, field="intent")
#     print(f"Hybrid LLM Answer: {hybrid_result.get('answer')}")

# import json
# from scripts.search_pipeline import search_query, hybrid_search

# # Example: Test metadata filtering for intent = 'claim_process'

# def test_metadata_filtering():
#     query = "test query"
#     filter_dict = {"intent": {"$eq": "claim_process"}}
#     print("\n--- Testing Dense Search with Metadata Filter ---")
#     dense_result = search_query(query, top_k=3, filter=filter_dict)
#     print(f"Dense search returned {len(dense_result['results'])} results.")
#     for doc in dense_result['results']:
#         print(f"Dense: {doc['document_name']} | intent: {filter_dict['intent']['$eq']} | summary: {doc['summary'][:80]}")

#     print("\n--- Testing Hybrid Search with Metadata Filter ---")
#     hybrid_result = hybrid_search(query, top_k=3, filter=filter_dict)
#     print(f"Hybrid search returned {len(hybrid_result['results'])} results.")
#     for doc in hybrid_result['results']:
#         print(f"Hybrid: {doc['document_name']} | intent: {filter_dict['intent']['$eq']} | summary: {doc['summary'][:80]}")

# if __name__ == "__main__":
#     test_metadata_filtering()
