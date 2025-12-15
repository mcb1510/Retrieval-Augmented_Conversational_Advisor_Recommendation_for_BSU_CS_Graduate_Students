# engine/response_engine.py
import os
import requests
import time
import re
from dotenv import load_dotenv
from engine.utils import _detect_list_query, _detect_list_with_research_query, _similarity, QueryProcessor
from engine.retrieval import FacultyRetriever
from engine.handlers import QueryHandlers
from engine.prompts import SYSTEM_PROMPT, get_rag_prompt

load_dotenv()
MODEL_NAME = "llama-3.3-70b-versatile"


class GroqClient:
    """Handles Groq API calls."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            print("WARNING: No GROQ_API_KEY found!")
            raise ValueError("GROQ_API_KEY required in . env file")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.model = MODEL_NAME
        print(f"Groq API initialized with {self.model}")
    
    def query(self, messages, max_retries=3, max_tokens=600):
        """Query Groq API with retry logic."""
        payload = {
            "model": self.model,
            "messages":  messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]. strip()
                elif response.status_code == 401:
                    return "Authentication error. Check your GROQ_API_KEY in the .env file."
                elif response.status_code == 429:
                    print(f"Rate limit, waiting...  (attempt {attempt + 1})")
                    time.sleep(2)
                    continue
                else: 
                    print(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Error:  {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        return "I'm having trouble connecting right now. Please try again in a moment."


class ResponseEngine:
    """
    Response engine using Groq API with Llama 3
    plus retrieval-augmented generation (RAG)
    over BSU CS faculty profiles.
    """

    def __init__(self):
        """Initialize Groq API connection and RAG resources."""
        self.groq_client = GroqClient()
        self.system_prompt = SYSTEM_PROMPT
        
        # RAG resources
        query_processor = QueryProcessor()
        self.retriever = FacultyRetriever(query_processor)
        self.handlers = QueryHandlers(self.retriever, self.groq_client)
        
        # Expose for backward compatibility
        self.embeddings = self.retriever.embeddings
        self.faculty_ids = self.retriever.faculty_ids
        self.faculty_texts = self.retriever.faculty_texts
        self.embed_model = self.retriever. embed_model
        self.query_processor = query_processor
        self.conversation_memory = self.handlers.conversation_memory
        self._query_groq = self.groq_client.query  # For backward compatibility

    def generate_answer(self, user_query, history=None):
        """Plain LLM answer using only the static system_prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            for msg in history[-6:]: 
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_query})
        return self. groq_client.query(messages, max_tokens=400)

    def ask(self, user_query, history=None, use_rag=False):
        """Main entry point - delegates to RAG or plain LLM."""
        if use_rag: 
            return self.generate_rag_answer(user_query, history=history)
        return self.generate_answer(user_query, history=history)

    def retrieve_faculty(self, query, top_k=3):
        """Retrieve top_k most relevant faculty profiles."""
        return self.retriever.retrieve_faculty(query, top_k)

    def generate_rag_answer(self, user_query, history=None, top_k=5):
        """RAG mode - delegates to RagAnswerGenerator."""
        from engine. rag_generator import RagAnswerGenerator
        generator = RagAnswerGenerator(self)
        return generator.generate(user_query, history, top_k)