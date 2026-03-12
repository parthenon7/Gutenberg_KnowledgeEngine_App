import streamlit as st
import time
import faiss
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Gutenberg Knowledge Engine", page_icon="📚", layout="wide")

# --- 2. RETRIEVAL UTILITIES ---
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
    prompt = f"""You are an objective research analyst. 
Your singular directive is to answer the user's query utilizing ONLY the provided text blocks.

<CONTEXT>
{context_string}
</CONTEXT>

<INSTRUCTIONS>
1. Evaluate the data strictly within the <CONTEXT> boundaries.
2. Formulate your response directly from this data. Do not incorporate external knowledge or pre-trained assumptions.
3. If the provided context lacks the specific facts required to answer the query, state: "The retrieved documents do not contain the necessary information."
</INSTRUCTIONS>

User Query: {query}
"""
    return prompt

# --- 3. RESOURCE CACHING (Memory Optimized) ---
@st.cache_resource
def load_models_and_data():
    # IO_FLAG_MMAP forces FAISS to read from disk, saving ~800MB of RAM
    index = faiss.read_index("data/showcase_index.faiss", faiss.IO_FLAG_MMAP)
    with open("data/showcase_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return {"index": index, "metadata": metadata, "encoder": encoder}

res = load_models_and_data()
index = res["index"]
metadata = res["metadata"]
encoder = res["encoder"]

# --- 4. UI: SIDEBAR ---
with st.sidebar:
    st.header("Graph Filters")
    selected_theme = st.selectbox("Require Theme:", ["None", "justice", "power", "liberty", "morality"])
    st.divider()
    st.markdown("### System Stats")
    st.markdown(f"- **Total Vectors:** {index.ntotal:,}")
    st.markdown("- **Hardware:** CPU (Streamlit Cloud)")

# --- 5. UI: MAIN SEARCH ---
st.title("📚 Gutenberg Knowledge Engine")
query = st.text_input("Enter research query:", placeholder="e.g., Arguments for liberty against state power...")

if query:
    with st.spinner("Searching 153k chunks..."):
        # 1. Vectorize Query and strictly cast to float32 to prevent C++ segfaults
        query_vec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        
        # 2. Search FAISS
        D, I = index.search(query_vec, k=10)
        
        # 3. Map indices back to text
        raw_chunks = [metadata[idx]['x'] for idx in I[0]]
        
        # 4. Pipeline Execution (Reranker bypassed for memory stability)
        cleaned = [trim_truncated_boundaries(c) for c in raw_chunks]
        optimized = reorder_for_attention_curve(cleaned) 
        prompt = generate_grounded_prompt(query, optimized)

    # --- 6. RESULTS ---
    st.success("Analysis Complete")
    tab1, tab2 = st.tabs(["Verified Evidence", "LLM Prompt"])
    with tab1:
        for i, text in enumerate(cleaned[:5]):
            st.info(f"**Top Result {i+1}:** {text}")
    with tab2:
        st.code(prompt, language="markdown")