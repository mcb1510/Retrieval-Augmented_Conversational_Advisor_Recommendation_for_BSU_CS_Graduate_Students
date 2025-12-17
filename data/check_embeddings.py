import numpy as np
import json

# Load embeddings
E = np.load("embeddings.npy")
print("Embedding matrix shape:", E.shape)

# Load IDs
with open("faculty_ids.json", "r") as f:
    ids = json.load(f)
print("IDs count:", len(ids))
print("First 3 IDs:", ids[:3])

# Load texts
with open("faculty_texts.json", "r") as f:
    texts = json.load(f)
print("Texts count:", len(texts))
print("First 100 chars of first profile:\n", texts[0][:100])
