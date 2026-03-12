import re
from sentence_transformers import CrossEncoder

# Initialize reranker globally upon import to optimize Streamlit caching
reranker = CrossEncoder('BAAI/bge-reranker-base')

def trim_truncated_boundaries(chunk_text: str) -> str:
    terminators = [m.start() for m in re.finditer(r'[.!?]', chunk_text)]
    if len(terminators) < 2: return chunk_text 
    return chunk_text[terminators[0] + 1 : terminators[-1] + 1].strip()

def rerank_chunks(query: str, retrieved_chunks: list[str]) -> list[str]:
    if not retrieved_chunks: return []
    model_inputs = [[query, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(model_inputs)
    ranked_pairs = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in ranked_pairs]

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
3. If the provided context lacks the specific facts required to answer the query, you must state: "The retrieved documents do not contain the necessary information to form a factual conclusion." 
4. Cite the explicit mechanics, techniques, or quotes present in the text to validate your claims.
</INSTRUCTIONS>

User Query: {query}
"""
    return prompt