import faiss
import pickle
from sentence_transformers import SentenceTransformer

from tutor_controller_chat import TutorSession

# -----------------------------
# Load FAISS index
# -----------------------------

INDEX_DIR = "../data/index_metadata_cosine_idea"

index = faiss.read_index(f"{INDEX_DIR}/ict_index.faiss")

with open(f"{INDEX_DIR}/texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open(f"{INDEX_DIR}/chunk_ids.pkl", "rb") as f:
    chunk_ids = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, top_k: int = 6):
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)

    out = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        out.append({
            "id": chunk_ids[idx],
            "score": float(score),
            "text": texts[idx],
        })
    return out


def main():
    tutor = TutorSession(
        retrieve_fn=retrieve,
        model="llama3.2:3b-instruct-q4_K_M",
        include_welcome=True,
        debug=True,
        use_assessor=True,
        advance_conf_threshold=0.66,
        assessor_debug=True,
    )

    welcome = tutor.start()
    print("\nTutor:", welcome)

    while True:
        user_text = input("\nLearner: ").strip()
        if user_text.lower() == "exit":
            break

        answer = tutor.turn(user_text)
        print("\nTutor:", answer)


if __name__ == "__main__":
    main()
