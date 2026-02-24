import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "../data/index_metadata_cosine_idea"

index = faiss.read_index(f"{INDEX_DIR}/ict_index.faiss")

with open(f"{INDEX_DIR}/texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open(f"{INDEX_DIR}/chunk_ids.pkl", "rb") as f:
    chunk_ids = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_scores(query, top_k=6):
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return [float(s) for s in scores[0] if s is not None and s > -1e9]

def best_score(query):
    scores = search_scores(query, top_k=6)
    return max(scores) if scores else -1.0

def summarize(label, queries):
    scores = [best_score(q) for q in queries]
    scores_sorted = sorted(scores, reverse=True)

    print(f"\n=== {label} ===")
    for q, s in sorted(zip(queries, scores), key=lambda x: x[1], reverse=True):
        print(f"{s:.3f} | {q}")

    print("\nSummary:")
    print(f"  n={len(scores)}")
    print(f"  mean={np.mean(scores):.3f}  median={np.median(scores):.3f}")
    print(f"  min={np.min(scores):.3f}   max={np.max(scores):.3f}")
    return scores

# EDIT THESE LISTS to match your course coverage:
IN_SCOPE = [
    "What is a computer?",
    "Write down the characteristics of computer",
    "What is an input device?",
    "Explain mouse",
    "Explain trackball",
    "What is machine language?",
    "Explain operating system",
]

OUT_OF_SCOPE = [
    "What is artificial intelligence?",
    "How do I grow tomatoes?",
    "Explain Fourier transforms",
    "Write a poem about winter",
    "What is the capital of Brazil?",
    "How do I invest in index funds?",
]

in_scores = summarize("IN-SCOPE", IN_SCOPE)
out_scores = summarize("OUT-OF-SCOPE", OUT_OF_SCOPE)

# A simple “suggested threshold” heuristic:
# Pick a threshold midway between the in-scope 25th percentile and out-of-scope 75th percentile.
in_p25 = float(np.percentile(in_scores, 25))
out_p75 = float(np.percentile(out_scores, 75))
suggested = (in_p25 + out_p75) / 2.0

print("\n=== Separation ===")
print(f"in-scope p25 = {in_p25:.3f}")
print(f"out-scope p75 = {out_p75:.3f}")
print(f"suggested threshold ≈ {suggested:.3f}")

if in_p25 <= out_p75:
    print("WARNING: score distributions overlap a lot. Consider improving chunking/cleaning or adding a reranker.")
else:
    print("Good: in-scope scores are generally above out-of-scope scores.")
