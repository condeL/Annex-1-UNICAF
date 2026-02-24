import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_DIR = "../data/chunks_by_metadata_idea"
INDEX_DIR = "../data/index_metadata_cosine_idea"
os.makedirs(INDEX_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
chunk_ids = []

for fname in sorted(os.listdir(CHUNK_DIR)):
    if fname.endswith(".txt"):
        with open(os.path.join(CHUNK_DIR, fname), "r", encoding="utf-8") as f:
            texts.append(f.read())
        chunk_ids.append(fname)

print(f"Loaded {len(texts)} chunks")

embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# Normalize to unit vectors: cosine similarity becomes dot product
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product

index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_DIR, "ict_index.faiss"))

with open(os.path.join(INDEX_DIR, "chunk_ids.pkl"), "wb") as f:
    pickle.dump(chunk_ids, f)

with open(os.path.join(INDEX_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("Cosine/IP FAISS index successfully created")
