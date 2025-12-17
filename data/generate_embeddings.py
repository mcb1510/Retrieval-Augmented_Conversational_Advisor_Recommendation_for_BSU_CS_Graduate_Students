import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# -----------------------------
# Main Pipeline
# -----------------------------
# Generate embeddings for faculty profiles
def main():
    print("[INFO] Loading dataset: faculty_ready.csv")
    # Load dataset
    df = pd.read_csv("data/faculty_ready.csv", encoding="utf-8")

    # Check for profile_text column
    if "profile_text" not in df.columns:
        raise ValueError("The file must contain a 'profile_text' column.")

    # Extract texts
    texts = df["profile_text"].fillna("").tolist()
    names = df["name"].tolist()

    print(f"[INFO] Loaded {len(texts)} faculty profiles.")

    print("[INFO] Loading SentenceTransformer model: BAAI/bge-large-en-v1.5")
    print("[INFO] This model is optimized for retrieval tasks (1024 dimensions)")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # generate embeddings
    print("[INFO] Generating embeddings...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    embeddings = normalize(embeddings)

    print("[INFO] Saving embedding matrix...")
    np.save("embeddings.npy", embeddings)

    print("[INFO] Saving faculty IDs...")
    with open("faculty_ids.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=4)

    print("[INFO] Saving faculty texts (for retrieval)…")
    with open("faculty_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4)

    print("\n[DONE] Embeddings generated and saved successfully!")
    print("Files created:")
    print(" - embeddings.npy")
    print(" - faculty_ids.json")
    print(" - faculty_texts.json")

if __name__ == "__main__":
    main()
