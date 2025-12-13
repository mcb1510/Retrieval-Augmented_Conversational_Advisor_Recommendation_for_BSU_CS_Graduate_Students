import requests
import os
import time
import json
import re
import numpy as np
from difflib import SequenceMatcher  # for fuzzy name matching
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

load_dotenv()
MODEL_NAME = "llama-3.3-70b-versatile"

# ============================
# Helper functions
# ============================

def _similarity(a: str, b: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _detect_list_query(text: str) -> bool:
    """Detect if the user is asking for a list of all faculty using regex-based intent detection."""
    q = text.lower()
    patterns = [
        r"(list|show|display|give|tell me).*(all|everyone|every).*(faculty|professors? )",
        r"(list|show|display).*(faculty|professors?)",
        r"(who are|what are).*(all|everyone).*(faculty|professors?)",
        r"(all|everyone).*(faculty|professors? )",
    ]
    return any(re.search(pattern, q) for pattern in patterns)

def _detect_list_with_research_query(text: str) -> bool:
    """Detect if user wants faculty list WITH research areas."""
    q = text.lower()
    return bool(re.search(r"(list|show).*(faculty|professors? ).*(research|areas?|interests?)", q))

# HELPER CLASS FOR QUERY EXPANSION
class QueryProcessor:
    """Expands queries with domain-specific synonyms"""
    
    def __init__(self):
        self.research_synonyms = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'ml': ['machine learning', 'deep learning', 'statistical learning'],
            'security': ['cybersecurity', 'privacy', 'cryptography', 'network security'],
            'hci': ['human computer interaction', 'user experience', 'interface design', 'usability'],
            'nlp': ['natural language processing', 'computational linguistics', 'text mining'],
            'cv': ['computer vision', 'image processing', 'pattern recognition'],
            'systems': ['distributed systems', 'operating systems', 'cloud computing', 'parallel computing'],
            'blockchain': ['distributed ledger', 'cryptocurrency', 'consensus protocols'],
        }
    
    def expand_query(self, query):
        """Expand query with synonyms"""
        query_lower = query.lower()
        expanded_terms = []
        
        for keyword, synonyms in self.research_synonyms.items():
            if keyword in query_lower:
                expanded_terms.extend(synonyms)
        
        if expanded_terms:
            return f"{query} {' '.join(set(expanded_terms))}"
        return query


class ResponseEngine:
    """
    Response engine using Groq API with Llama 3
    plus retrieval-augmented generation (RAG)
    over BSU CS faculty profiles.
    """

    def __init__(self):
        """Initialize Groq API connection and RAG resources."""

        # ---------- LLM / Groq setup ----------
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            print("WARNING: No GROQ_API_KEY found!")
            raise ValueError("GROQ_API_KEY required in .env file")

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.model = MODEL_NAME

        # General persona (used for non-RAG answers)
        # Adjusted to stop asking too many clarifying questions
        self.system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science "
            "students at Boise State University.\n\n"
            "Your role:\n"
            "- Help students find suitable research advisors based on their interests, skills, and goals\n"
            "- Provide information about faculty research areas and general availability\n"
            "- Guide students through the advisor selection process\n"
            "- Answer questions about BSU CS graduate programs\n"
            "- Be direct and concise (2 to 4 sentences)\n"
            "- Only ask a clarifying question if the student's request is genuinely ambiguous\n"
            "- When possible, make the best recommendation from available information instead of asking many follow up questions\n\n"
            "When you are provided with faculty profiles in the context, you MUST rely on that "
            "data and not invent additional details."
        )

        print(f"Groq API initialized with {self.model}")

        # ---------- RAG resources (embeddings + profiles) ----------
        self._load_rag_resources()
        
        # Add query processor
        self.query_processor = QueryProcessor()
        # Conversation memory for follow-ups
        self.conversation_memory = {
            "last_query": None,
            "last_retrieved": None
        }

    # =================================================================
    # RAG INITIALIZATION
    # =================================================================

    def _load_rag_resources(self):
        """
        Load faculty embeddings and metadata for retrieval.
        Expects:
            - embeddings.npy
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
                    f"!= ids count ({len(self.faculty_ids)})"
                )

            print("[RAG] Loading BGE-Large model for query encoding...")
            self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            print("[RAG] Model loaded: 1024-dimensional embeddings for superior retrieval")


            # Ensure embeddings are L2-normalized (just in case)
            self.embeddings = normalize(self.embeddings)

            print(f"[RAG] Loaded {len(self.faculty_ids)} faculty profiles for retrieval.")
        except Exception as e:
            print(f"[RAG] WARNING: could not load RAG resources: {e}")
            self.embeddings = None
            self.faculty_ids = None
            self.faculty_texts = None
            self.embed_model = None

    # =================================================================
    # BASE LLM CALL (non-RAG)
    # =================================================================

    def generate_answer(self, user_query, history=None):
        """
        Plain LLM answer using only the static system_prompt.
        (Old behavior, still available.)
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history (keep last 6 messages for context)
        if history:
            for msg in history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        messages.append({
            "role": "user",
            "content": user_query
        })

        answer = self._query_groq(messages, max_tokens=400)
        return answer

    # Convenience alias if you ever want engine.ask(...)
    def ask(self, user_query, history=None, use_rag=False):
        """
        If use_rag is True, use the RAG pipeline with all the special
        handling for listing faculty and fuzzy name matching.
        Otherwise fall back to plain LLM.
        """
        if use_rag:
            return self.generate_rag_answer(user_query, history=history)
        return self.generate_answer(user_query, history=history)

    # =================================================================
    # RAG: RETRIEVAL + GENERATION
    # =================================================================

    def retrieve_faculty(self, query, top_k=3):
        """
        Retrieve top_k most relevant faculty profiles for a given query.
        Returns a list of dicts with {name, score, profile_text}.
        """
        if self.embed_model is None or self.embeddings is None:
            print("[RAG] Retrieval requested but RAG resources are not loaded.")
            return []        

        # Encode and normalize query
        expanded_query = self.query_processor.expand_query(query)
        q_emb = self.embed_model.encode([expanded_query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

        # Cosine similarity because embeddings are normalized
        sims = self.embeddings @ q_emb

        top_k = min(top_k, len(sims))
        idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in idxs:
            results.append({
                "name": self.faculty_ids[idx],
                "score": float(sims[idx]),
                "profile_text": self.faculty_texts[idx]
            })

        return results

    def _list_all_faculty_text(self):
        """Return a human readable list of all faculty names."""
        if not self.faculty_ids:
            return "I do not have any faculty data loaded right now."
        
        # Clear conversation memory since we're starting fresh
        self.conversation_memory = {
            "last_query": None,
            "last_retrieved":  None
        }

        lines = [f"- {name}" for name in self.faculty_ids]
        return (
            "Here is the list of CS faculty I know about:\n\n"
            + "\n".join(lines)
            + "\n\nYou can ask me about any specific person, or tell me your interests and I will recommend a few advisors."
        )

    def _list_all_faculty_with_research(self):
        """Return faculty list with research areas."""
        if not self.faculty_ids or not self.faculty_texts:
            return "I do not have any faculty data loaded right now."
        
        self.conversation_memory = {"last_query": None, "last_retrieved": None}

        lines = []
        for name, profile in zip(self.faculty_ids, self.faculty_texts):
            # Extract between "Research Areas:" and " Paper " or " Google Scholar:"
            if "Research Areas:" in profile:
                research = profile.split("Research Areas:")[1].split(" Paper ")[0].split(" Google Scholar: ")[0].strip()
            else:
                research = "Research areas not listed"
            lines.append(f"• **{name}**: {research}")
        
        return (
            "Here is the list of CS faculty with their research areas:\n\n" 
            + "\n\n".join(lines)
            + "\n\nAsk me about any specific professor for more details!"
        )
        
    def _answer_for_specific_faculty(self, faculty_name, history=None):
        """
        Build a focused prompt for one matched faculty member. 
        This is used when we fuzzy match a misspelled name.
        """
        if not self.faculty_ids or not self.faculty_texts:
            return "I could not load the faculty profiles right now."

        try:
            idx = self.faculty_ids.index(faculty_name)
        except ValueError:
            return "I could not find that faculty in my profiles."

        profile = self.faculty_texts[idx]
        
        # Store memory for follow-up questions
        self.conversation_memory["last_query"] = faculty_name
        self.conversation_memory["last_retrieved"] = [{
            "name": faculty_name,
            "profile_text": profile
        }]
        
        prompt = f"""
            You are the AI Graduate Advisor for Boise State University. 

            The user is asking about:  {faculty_name}

            FACULTY PROFILE:
            {profile}

            INSTRUCTIONS:
            1. Give a concise but rich summary of this professor's research areas.
            2. Explain what makes their research interesting or impactful (3-4 sentences)
            3. Describe what background, skills, and interests graduate students typically need to work with this professor (2-3 sentences)
            4. Include all available contact information:  email, office location, and Google Scholar link
            5. Keep the answer helpful, direct, and focused.
            6. Be specific about their research - use the actual topics from their profile
            7. Do NOT use phrases like "keywords that match your query", "specific keywords", or "as an active researcher"
            8. Avoid repeating boilerplate language.
            9. Always ask at the end if there is anything else you can help with.(1 very small question)

            TONE: 
            - Informative and professional but conversational
            - Enthusiastic about the professor's research
            - Helpful and practical for students making decisions
            """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Tell me about {faculty_name} as a potential advisor for me. "}
        ]

        # Optionally include short history
        if history:
            for msg in history[-3:]:
                messages.insert(1, {
                    "role": msg["role"],
                    "content":  msg["content"]
                })

        return self._query_groq(messages, max_tokens=600)
    
    def _answer_followup_fact(self, faculty_name, user_query, history=None):
        """
        Extracts specific factual information (like office, email, interests)
        from the stored faculty profile with a conversational tone.
        """

        # Locate the faculty in the dataset
        try:
            idx = self.faculty_ids.index(faculty_name)
        except ValueError:
            return "I couldn't find that faculty member anymore."

        profile = self.faculty_texts[idx]
        
        # Build conversation context from history if available
        conversation_context = ""
        if history and len(history) > 0:
            recent_messages = history[-6:]  # Last 3 exchanges
            conversation_context = "RECENT CONVERSATION:\n"
            for msg in recent_messages:
                role = "USER" if msg. get("role") == "user" else "ASSISTANT"
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 300:
                    content = content[:300] + "..."
                conversation_context += f"{role}: {content}\n\n"

        # Build a conversational factual prompt
        prompt = f"""
            You are the BSU Graduate Advisor AI Assistant. You are helping a student learn about Professor {faculty_name}. 

            {conversation_context}

            FACULTY PROFILE:
            {profile}

            CURRENT USER INPUT:
            {user_query}

            INSTRUCTIONS:
            - If the user said "yes", "sure", "ok", or similar, look at the RECENT CONVERSATION to see what you offered
            - Provide the specific information that was offered in your previous message
            - If you offered multiple options (e.g., "research areas or contact info"), pick the first one or most relevant
            - If the user said "no", acknowledge and offer other help
            - If you cannot determine what they want from context, ask for clarification
            - Answer directly and conversationally in complete sentences
            - After answering, offer to help further with a friendly follow-up question
            - Keep it brief but warm and helpful (2-3 sentences + follow-up offer)
            - If the information is not in the profile, say so politely and suggest alternatives (like Google Scholar)

            EXAMPLES: 

            Previous:  "Would you like to know more about her research areas or how to contact her?"
            User: "yes"
            Response: "Professor Zhou's research focuses on Trustworthy Generative AI, Human-Centered LLMs, Multimodal Machine Learning, and LLM Agents. Her work is particularly relevant for students interested in making AI systems more reliable and human-centered.  Would you like to know about her publications or how to reach out to her?"

            Previous: "Would you like to visit her Google Scholar page?"
            User: "yes"
            Response: "You can find Professor Zhou's Google Scholar page at https://scholar.google.com/citations?user=9U_Ge4MAAAAJ. This will show you her publications and current research projects.  Is there anything else you'd like to know about Professor Zhou?"

            Previous: "Would you like to know more?"
            User: "no"
            Response: "No problem! Feel free to ask me about other faculty members or research areas. How else can I help you?"
            """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]

        return self._query_groq(messages, max_tokens=200)
    


    def _is_followup(self, query: str) -> bool:
        q = query.lower()
        for name in self.faculty_ids:
            # direct substring check
            if name.lower() in q:
                return False
            sim = _similarity(q, name.lower())
            if sim > 0.65:
                return False

        mem_exists = self.conversation_memory.get("last_retrieved") is not None
        return mem_exists
    

    def classify_query_type(self, query):
        """
        Classify the user query into one of:
        - 'followup_person'
        - 'general_concept'
        - 'new_professor'
        """
        system_prompt = """
            You are a query classifier. Classify the user's question into ONE of these:

            1. followup_person:
            - The question refers to the previously discussed professor.
            - Includes pronouns like he, him, his, she, her, they, them.
            - Includes questions about their office, email, research areas, advising, etc.

            2. general_concept:
            - The question asks about a research field, definition, concept, method,
                technique, or career/job possibilities (e.g., "what is X?", "what jobs can X lead to?").

            3. new_professor:
            - The question is asking about a professor different from the last one
                (directly or indirectly), OR is requesting new advisor recommendations.

            Respond with ONLY the category name: followup_person, general_concept, or new_professor.
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        result = self._query_groq(messages, max_tokens=5).lower().strip()

        # Safety normalization
        if "follow" in result:
            return "followup_person"
        if "concept" in result:
            return "general_concept"
        if "professor" in result or "new" in result:
            return "new_professor"

        return "general_concept"  # safe fallback


    def _answer_concept_definition(self, query):
        prompt = f"""
            You are an AI assistant. Provide a clear explanation for the research concept
            or topic the user is asking about.

            Requirements:
            - Give a correct 2–4 sentence definition.
            - Use examples relevant to Computer Science.
            - If appropriate, mention what careers or research areas use this concept.
            - Always asks if there is anything else you can help with.(1 very small question)

            USER QUESTION:
            {query}
            """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        return self._query_groq(messages, max_tokens=300)

    

    # =================================================================
    # LOW LEVEL GROQ CALL
    # =================================================================

    def _query_groq(self, messages, max_retries=3,max_tokens=600):
        """Query Groq API with retry logic."""

        payload = {
            "model": self.model,
            "messages": messages,
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
                    answer = result["choices"][0]["message"]["content"].strip()
                    return answer

                elif response.status_code == 401:
                    return "Authentication error. Check your GROQ_API_KEY in the .env file."

                elif response.status_code == 429:
                    print(f"Rate limit, waiting... (attempt {attempt + 1})")
                    time.sleep(2)
                    continue

                else:
                    print(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        return "I'm having trouble connecting right now. Please try again in a moment."
    def generate_rag_answer(self, user_query, history=None, top_k=5):
        """
        RAG mode: 
        1) Special handling:   list-all queries and fuzzy name matches. 
        2) Otherwise retrieve top_k matching faculty profiles.  
        3) Inject them into a system message.
        4) Ask Llama to answer using ONLY that faculty context.
        """
        if _detect_list_with_research_query(user_query) and self.faculty_ids:
            return self._list_all_faculty_with_research()
        # ============================================================
        # PRIORITY CHECK 1: List all faculty (should happen FIRST)
        # ============================================================
        if _detect_list_query(user_query) and self.faculty_ids:
            return self._list_all_faculty_text()

        # ============================================================
        # PRIORITY CHECK 2: Direct faculty name mentions (BEFORE affirmative/followup)
        # ============================================================
        if self.faculty_ids:
            q = user_query. lower()
            query_tokens = q.split()

            for name in self.faculty_ids:
                name_tokens = name.lower().split()

                # 1. Token-level containment ("jerry fails" → "jerry alan fails")
                token_matches = sum(1 for qt in query_tokens for nt in name_tokens 
                                    if qt == nt)

                if token_matches >= 2:
                    print("TOKEN DIRECT MATCH:", name)
                    return self._answer_for_specific_faculty(name, history=history)

                # 2. Token-level fuzzy matching ("jarry" ≈ "jerry")
                fuzzy_matches = sum(1 for qt in query_tokens for nt in name_tokens 
                                    if _similarity(qt, nt) > 0.70)

                if fuzzy_matches >= 2:
                    print("TOKEN FUZZY MATCH:", name)
                    return self._answer_for_specific_faculty(name, history=history)

        # ============================================================
        # PRIORITY CHECK 3: Handle affirmative/negative responses
        # ============================================================
        affirmative_patterns = [
            r'^(yes|yeah|yep|yup|sure|ok|okay|alright|please|yes please|sure thing)\.? ! ?$',
            r'^(yes|yeah|sure),?\s+(tell me|show me|give me|send me|what about)',
            r'^tell me more$',
            r'^show me more$',
            r'^more\. ?$',
            r'^(that would be|that\'d be|sounds) (great|good|helpful|perfect|nice)',
            r'^(go ahead|please do|i\'m interested)\.?$',
        ]

        negative_patterns = [
            r'^(no|nope|nah|no thanks|no thank you)\.?!?$',
            r'^(that\'s all|that\'s it|i\'m good|i\'m all set)\.?$',
            r'^(nothing else|nothing more)\.?$',
            r'^(i\'m done|all done)\.?$',
        ]

        query_lower = user_query.lower().strip()
        is_affirmative = any(re.match(pattern, query_lower) for pattern in affirmative_patterns)
        is_negative = any(re.match(pattern, query_lower) for pattern in negative_patterns)

        # Handle affirmative responses
        if is_affirmative:
            last = self. conversation_memory.get("last_retrieved")
            if last and len(last) > 0:
                faculty_name = last[0]["name"]
                return self._answer_followup_fact(faculty_name, user_query, history=history)

        # Handle negative responses
        if is_negative:
            return "No problem!   Feel free to ask me about other faculty members, research areas, or anything else about the BSU CS graduate program.  How else can I help you?"

        # ============================================================
        # PRIORITY CHECK 4: Query classification for remaining queries
        # ============================================================
        query_type = self. classify_query_type(user_query. lower())

        # Retrieve last professor if available
        last = self.conversation_memory.get("last_retrieved")
        last_prof = last[0]["name"] if last and len(last) > 0 else None

        if query_type == "followup_person" and last_prof:
            return self._answer_followup_fact(last_prof, user_query, history=history)

        if query_type == "general_concept": 
            # Not a follow-up → do NOT use fact mode
            return self._answer_concept_definition(user_query)


        # ============================================================
        # PRIORITY CHECK 5: Normal RAG retrieval
        # ============================================================
        retrieved = self. retrieve_faculty(user_query, top_k=top_k)
        self.conversation_memory["last_query"] = user_query
        self.conversation_memory["last_retrieved"] = retrieved

        # If nothing retrieved, give a clear fallback instead of silence
        if not retrieved:
            return (
                "I could not match your question to any specific faculty profiles. "
                "Try telling me your research interests, for example: "
                "\"I am interested in AI and machine learning\" or "
                "\"I want to work on cybersecurity and privacy\"."
            )

        context_blocks = []
        for i, r in enumerate(retrieved, start=1):
            block = (
                f"FACULTY MATCH {i}:\n"
                f"Name: {r['name']}\n"
                f"Relevance score: {r['score']:.3f}\n"
                f"Profile:\n{r['profile_text']}\n"
            )
            context_blocks.append(block)
        faculty_context = "\n---\n".join(context_blocks)

        rag_system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science students "
            "at Boise State University.\n\n"
            "You are connected to a factual database of BSU CS faculty profiles.\n"
            "Below you are given the top retrieved faculty profiles that are relevant "
            "to the student's question.\n\n"
            "=== FACULTY CONTEXT START ===\n"
            f"{faculty_context}\n"
            "=== FACULTY CONTEXT END ===\n\n"
            "Instructions:\n"
            "- When recommending advisors, rely ONLY on the information in the faculty context.\n"
            "- Recommend 1 to 3 specific faculty that best match the student's interests.\n"
            "- Briefly explain why each recommended faculty member is a good match.\n"
            "- Do NOT ask unnecessary clarifying questions.  Make the best recommendation with the information you have.\n"
            "- If the context is insufficient, say you are not sure and suggest contacting the department.\n"
            "- Keep answers concise (2 to 4 sentences) and supportive."
        )

        messages = [
            {"role": "system", "content": rag_system_prompt}
        ]

        # Optional: include short history for conversational feel
        if history:
            for msg in history[-4:]: 
                if msg. get("role") in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        messages.append({
            "role": "user",
            "content": user_query
        })

        return self._query_groq(messages, max_tokens=800)