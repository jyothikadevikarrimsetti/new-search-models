import streamlit as st
import json
import warnings
from scripts.search_pipeline import search_query, hybrid_search
from scripts.filter_utils import generate_filter

# Suppress FutureWarning messages for a cleaner user experience
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="PDF Vector Search", layout="wide")
st.title("PDF Vector Search & RAG Demo")

st.markdown("""
This app lets you search your PDF knowledge base using dense, hybrid, and LLM-powered retrieval.
""")

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Enter your question:")
    search_type = st.radio("Search type", ["Dense", "Hybrid"], horizontal=True)
    top_k = st.slider("Top K Results", 1, 10, 3)

with col2:
    st.markdown("### Filter Options")
    filter_method = st.radio("Filter Method", ["Automatic", "Manual"], horizontal=True)
    
    if filter_method == "Manual":
        # Show manual filter input as JSON
        manual_filter = st.text_area(
            "Enter filter as JSON:", 
            value='{"intent": {"$eq": "claim_process"}}',
            help="Example: {'intent': {'$eq': 'claim_process'}}"
        )
        try:
            filter_dict = json.loads(manual_filter) if manual_filter.strip() else None
        except json.JSONDecodeError:
            st.error("Invalid JSON filter format")
            filter_dict = None
    else:        # Show quick filter options
        intent_options = [
            "None",
            "claim_process",
            "case_status",
            "document_request",
            "technical_support",
            "general_info"
        ]
        selected_intent = st.selectbox(
            "Filter by Intent", 
            intent_options
        )
        if selected_intent != "None":
            filter_dict = {"intent": {"$eq": selected_intent}}
        else:
            filter_dict = None
            
    # Show the current filter
    if filter_dict:
        st.markdown("**Active Filter:**")
        st.json(filter_dict)

MIN_RERANK_SCORE = 0.5  # Minimum reranking score for a result to be considered relevant

if st.button("Search") and query:
    with st.spinner("Searching..."):
        # If no manual filter, generate automatically
        auto_filter = filter_dict if filter_method == "Manual" else generate_filter(query)
        if search_type == "Dense":
            st.write("### Dense Search Results")
            result = search_query(query, top_k=top_k, filter=auto_filter)
            # Check for no results or low reranking score
            no_results = not result or not result.get('results')
            low_score = (
                result and result.get('reranking_score') is not None and
                result.get('reranking_score') < MIN_RERANK_SCORE
            )
            if no_results or low_score:
                st.info("No results found.")
            else:
                st.markdown(f"**LLM Answer:**\n{result['answer']}")
                st.markdown(f"_Search Time: {result.get('search_time', 0.0):.2f} seconds_")
                st.write("---")
                for doc in result['results']:
                    st.markdown(f"**Document:** {doc.get('document_name','')}")
                    if 'pinecone_score' in doc:
                        st.markdown(f"Pinecone Score: {doc['pinecone_score']:.6f}")
                    if 'dense_score' in doc:
                        st.markdown(f"Dense Score: {doc['dense_score']:.6f}")
                    if 'sparse_score' in doc:
                        st.markdown(f"Sparse Score: {doc['sparse_score']:.6f}")
                    if 'hybrid_score' in doc:
                        st.markdown(f"Hybrid Score: {doc['hybrid_score']:.6f}")
                    if 'bm25_score' in doc:
                        st.markdown(f"BM25 Score: {doc['bm25_score']:.6f}")
                    if 'rerank_score' in doc:
                        st.markdown(f"Rerank Score: {doc['rerank_score']:.6f}")
                    if 'cosine_score' in doc and doc['cosine_score'] is not None:
                        st.markdown(f"Cosine Score: {doc['cosine_score']:.6f}")
                    st.markdown(f"Summary: {doc.get('summary','')[:500]}")
                    st.write("---")
        else:
            st.write("### Hybrid Search Results")
            result = hybrid_search(query, top_k=top_k, filter=auto_filter)
            no_results = not result or not result.get('results')
            low_score = (
                result and result.get('reranking_score') is not None and
                result.get('reranking_score') < MIN_RERANK_SCORE
            )
            if no_results or low_score:
                st.info("No results found.")
            else:
                st.markdown(f"**LLM Answer (Top Result):**\n{result.get('answer', '')}")
                st.markdown(f"**Time Taken:** {result.get('time_complexity', '')}")
                st.write("---")
                # Show top-k hybrid results (document_name, reranking_score, dense_score, sparse_score, summary)
                if 'results' in result and result['results']:
                    st.markdown(f"**Top {len(result['results'])} Hybrid Results:**")
                    for i, doc in enumerate(result['results'], 1):
                        st.markdown(f"**{i}. Document:** {doc.get('document_name', '')}")
                        st.markdown(f"Reranking Score: {doc.get('reranking_score', ''):.6f}" if doc.get('reranking_score') is not None else "")
                        st.markdown(f"Dense Score: {doc.get('dense_score', ''):.6f}" if doc.get('dense_score') is not None else "")
                        st.markdown(f"Sparse Score: {doc.get('sparse_score', ''):.6f}" if doc.get('sparse_score') is not None else "")
                        st.markdown(f"Summary: {doc.get('summary', '')[:500]}")
                        st.write("---")

st.markdown("---")
st.caption("Built with Streamlit, Pinecone, and Azure OpenAI.")
