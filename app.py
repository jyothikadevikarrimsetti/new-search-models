import streamlit as st
from scripts.search_pipeline import search_query, hybrid_search
import warnings

# Suppress FutureWarning messages for a cleaner user experience
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="PDF Vector Search", layout="wide")
st.title("PDF Vector Search & RAG Demo")

st.markdown("""
This app lets you search your PDF knowledge base using dense, hybrid, and LLM-powered retrieval.
""")

query = st.text_input("Enter your question:")
search_type = st.radio("Search type", ["Dense", "Hybrid"], horizontal=True)
top_k = st.slider("Top K Results", 1, 10, 3)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        if search_type == "Dense":
            st.write("### Dense Search Results")
            result = search_query(query, top_k=top_k)
        else:
            st.write("### Hybrid Search Results")
            result = hybrid_search(query, top_k=top_k)
        if result:
            st.markdown(f"**LLM Answer:**\n{result['answer']}")
            st.markdown(f"_Search Time: {result['search_time']:.2f} seconds_")
            st.write("---")
            for doc in result['results']:
                st.markdown(f"**Document:** {doc.get('document_name','')}")
                if 'pinecone_score' in doc:
                    st.markdown(f"Pinecone Score: {doc['pinecone_score']:.6f}")
                if 'bm25_score' in doc:
                    st.markdown(f"BM25 Score: {doc['bm25_score']:.6f}")
                if 'rerank_score' in doc:
                    st.markdown(f"Rerank Score: {doc['rerank_score']:.6f}")
                if 'cosine_score' in doc and doc['cosine_score'] is not None:
                    st.markdown(f"Cosine Score: {doc['cosine_score']:.6f}")
                st.markdown(f"Summary: {doc.get('summary','')[:500]}")
                st.write("---")
        else:
            st.info("No results found.")

st.markdown("---")
st.caption("Built with Streamlit, Pinecone, and Azure OpenAI.")
