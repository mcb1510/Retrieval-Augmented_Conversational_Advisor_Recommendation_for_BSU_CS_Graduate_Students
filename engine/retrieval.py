# This file is part of the retrieval module for faculty profiles.
# # engine/retrieval.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# The FacultyRetriever class handles loading and retrieving faculty profiles.
class FacultyRetriever:
    """Handles faculty profile retrieval using semantic embeddings."""
    
    # Initialization and resource loading
    def __init__(self, query_processor):
        self.query_processor = query_processor
        self. embeddings = None
        self.faculty_ids = None
        self. faculty_texts = None
        self.embed_model = None
        self._load_rag_resources()
    
    # Load RAG resources from disk
    def _load_rag_resources(self):
        """
        Load faculty embeddings and metadata for retrieval.
        Expects: 
            - embeddings. npy
            - faculty_ids.json
            - faculty_texts.json
        in the current working directory.
        """
        try:
            print("[RAG] Loading faculty embeddings and metadata...")
            self.embeddings = np.load("embeddings.npy")

            with open("faculty_ids.json", "r", encoding="utf-8") as f:
                self.faculty_ids = json.load(f)

            with open("faculty_texts.json", "r", encoding="utf-8") as f:
                self.faculty_texts = json.load(f)

            if len(self.embeddings) != len(self.faculty_ids):
                print(
                    f"[RAG] WARNING: embeddings count ({len(self.embeddings)}) "
                    f"! = ids count ({len(self.faculty_ids)})"
                )

            print("[RAG] Loading BGE-Large model for query encoding...")
            self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            print("[RAG] Model loaded:  1024-dimensional embeddings for superior retrieval")

            # Ensure embeddings are L2-normalized (just in case)
            self.embeddings = normalize(self.embeddings)

            # Test print
            print(f"[RAG] Loaded {len(self.faculty_ids)} faculty profiles for retrieval.")
        except Exception as e: 
            print(f"[RAG] WARNING: could not load RAG resources: {e}")
            self.embeddings = None
            self.faculty_ids = None
            self.faculty_texts = None
            self.embed_model = None
    
    # Retrieve top_k relevant faculty profiles
    def retrieve_faculty(self, query, top_k=3):
        """
        Retrieve top_k most relevant faculty profiles for a given query.
        Returns a list of dicts with {name, score, profile_text}. 
        """
        if self. embed_model is None or self.embeddings is None:
            print("[RAG] Retrieval requested but RAG resources are not loaded.")
            return []        

        # Encode and normalize query
        expanded_query = self.query_processor.expand_query(query)
        q_emb = self.embed_model. encode([expanded_query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        # Cosine similarity because embeddings are normalized
        sims = self.embeddings @ q_emb

        # Get top_k results
        top_k = min(top_k, len(sims))
        idxs = np.argsort(sims)[::-1][: top_k]

        # Build results
        results = []
        for idx in idxs:
            results.append({
                "name":  self.faculty_ids[idx],
                "score": float(sims[idx]),
                "profile_text": self.faculty_texts[idx]
            })

        return results