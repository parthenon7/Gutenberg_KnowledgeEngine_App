import streamlit as st
import os
import re
import numpy as np

# 1. STRICT MEMORY CAPPING: Prevent PyTorch from spawning RAM-heavy background threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import faiss
import pickle
from sentence_transformers import SentenceTransformer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gutenberg Knowledge Engine", page_icon="📚", layout="wide")

# --- RETRIEVAL UTILITIES ---
def trim_truncated_boundaries(chunk_text: str) -> str:
    terminators = [m.start() for m in re.finditer(r'[.!?]', chunk_text)]
    if len(terminators) < 2: return chunk_text 
    return chunk_text[terminators[0] + 1 : terminators[-1] + 1].strip()

def reorder_for_attention_curve(ranked_chunks: list[str]) -> list[str]:
    if not ranked_chunks: return []
    reordered = [None] * len(ranked_chunks)
    left_index, right_index = 0, len(ranked_chunks) - 1
    for i, chunk in enumerate(ranked_chunks):
        if i % 2 == 0:
            reordered[left_index] = chunk
            left_index += 1
        else:
            reordered[right_index] = chunk
            right_index -= 1
    return reordered

def generate_grounded_prompt(query: str, optimized_context_array: list[str]) -> str:
    context_string = "\n\n---\n\n".join(optimized_context_array)
    return f"""You are an objective research analyst. 
Answer the user's query utilizing ONLY the provided text blocks.

<CONTEXT>
{context_string}
</CONTEXT>

User Query: {query}"""

# --- LAZY RESOURCE CACHING ---
@st.cache_resource
def load_engine():
    # Removed MMAP as it frequently causes silent segfaults on Flat Indexes in Linux
    index = faiss.read_index("data/showcase_index.faiss")
    with open("data/showcase_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    return index, metadata, encoder

# --- UI: SIDEBAR ---
with st.sidebar:
    st.header("Graph Filters")
    st.selectbox("Require Theme:", ["None", "justice", "power", "liberty", "morality"])
    st.divider()
    st.markdown("### System Stats")
    st.markdown("- **Engine Size:** 153k Vector Chunks")
    st.markdown("- **Hardware:** Streamlit Free Tier (1GB RAM Cap)")

# --- UI: MAIN SEARCH ---
st.title("📚 Gutenberg Knowledge Engine")
query = st.text_input("Enter research query:", placeholder="e.g., Arguments for liberty against state power...")

if query:
    # Because loading is behind this IF statement, the UI will always load instantly.
    with st.spinner("Booting ML Models (Warning: Approaching 1GB Cloud RAM Limit)..."):
        index, metadata, encoder = load_engine()
        
        query_vec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        D, I = index.search(query_vec, k=10)
        
        raw_chunks = [metadata[idx]['x'] for idx in I[0]]
        cleaned = [trim_truncated_boundaries(c) for c in raw_chunks]
        optimized = reorder_for_attention_curve(cleaned) 
        prompt = generate_grounded_prompt(query, optimized)

    st.success("Analysis Complete")
    tab1, tab2 = st.tabs(["Verified Evidence", "LLM Prompt"])
    with tab1:
        for i, text in enumerate(cleaned[:5]):
            st.info(f"**Top Result {i+1}:** {text}")
    with tab2:
        st.code(prompt, language="markdown")