import re
from sentence_transformers import CrossEncoder

# Initialize reranker globally upon import
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
