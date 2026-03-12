import sys
import os
# Force the Linux server to recognize the current folder as a module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils.retrieval_utils import trim_truncated_boundaries, rerank_chunks, reorder_for_attention_curve, generate_grounded_prompt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Gutenberg Knowledge Engine", page_icon="📚", layout="wide")

# --- 2. RESOURCE CACHING ---
@st.cache_resource
def load_models_and_data():
    # These paths must match where you put the files in your GitHub repo
    index = faiss.read_index("data/showcase_index.faiss")
    with open("data/showcase_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return {"index": index, "metadata": metadata, "encoder": encoder}

# Initialize resources
res = load_models_and_data()
index = res["index"]
metadata = res["metadata"]
encoder = res["encoder"]

# --- 3. UI: SIDEBAR ---
with st.sidebar:
    st.header("Graph Filters")
    selected_theme = st.selectbox("Require Theme:", ["None", "justice", "power", "liberty", "morality"])
    st.divider()
    st.markdown("### System Stats")
    st.markdown(f"- **Total Chunks:** {index.ntotal:,}")
    st.markdown("- **Hardware:** CPU (Streamlit Cloud)")

# --- 4. UI: MAIN SEARCH ---
st.title("📚 Gutenberg Knowledge Engine")
query = st.text_input("Enter research query:", placeholder="e.g., Arguments for liberty...")

if query:
    with st.spinner("Searching 153k chunks..."):
        # 1. Vectorize the User Query
        query_vec = encoder.encode([query], normalize_embeddings=True)
        
        # 2. Search FAISS index
        # We retrieve top 15 to account for potential filtering/reranking
        D, I = index.search(query_vec, k=15)
        
        # 3. Map indices back to metadata
        raw_chunks = [metadata[idx]['x'] for idx in I[0]]
        
        # 4. Run Pipeline Utilities
        cleaned = [trim_truncated_boundaries(c) for c in raw_chunks]
        ranked = rerank_chunks(query, cleaned)
        optimized = reorder_for_attention_curve(ranked[:10]) # Use top 10 for U-Curve
        prompt = generate_grounded_prompt(query, optimized)

    # --- 5. RESULTS ---
    st.success("Analysis Complete")
    tab1, tab2 = st.tabs(["Verified Evidence", "LLM Prompt"])
    with tab1:
        for i, text in enumerate(ranked[:5]):
            st.info(f"**Top Result {i+1}:** {text}")
    with tab2:
        st.code(prompt, language="markdown")
