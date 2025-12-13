# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY = os. getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Embedding configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# File paths
EMBEDDINGS_FILE = "embeddings.npy"
FACULTY_IDS_FILE = "faculty_ids.json"
FACULTY_TEXTS_FILE = "faculty_texts. json"

# Query expansion synonyms
RESEARCH_SYNONYMS = {
    'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
    'ml': ['machine learning', 'deep learning', 'statistical learning'],
    'security': ['cybersecurity', 'privacy', 'cryptography', 'network security'],
    'hci': ['human computer interaction', 'user experience', 'interface design', 'usability'],
    'nlp': ['natural language processing', 'computational linguistics', 'text mining'],
    'cv': ['computer vision', 'image processing', 'pattern recognition'],
    'systems': ['distributed systems', 'operating systems', 'cloud computing', 'parallel computing'],
    'blockchain': ['distributed ledger', 'cryptocurrency', 'consensus protocols'],
}