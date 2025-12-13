# engine/rag_generator.py
import re
from engine.utils import _detect_list_query, _detect_list_with_research_query, _similarity
from engine.prompts import get_rag_prompt


class RagAnswerGenerator:  
    """Handles the RAG answer generation logic with priority checks."""
    
    def __init__(self, engine):
        self.engine = engine
        self.handlers = engine.handlers
        self.retriever = engine.retriever
        self.groq_client = engine.groq_client
        self.conversation_memory = engine.conversation_memory
        self.faculty_ids = engine.faculty_ids
    
    def generate(self, user_query, history=None, top_k=5):
        """
        RAG mode: 
        1) Special handling:  list-all queries and fuzzy name matches.  
        2) Otherwise retrieve top_k matching faculty profiles.
        3) Inject them into a system message.
        4) Ask Llama to answer using ONLY that faculty context.
        """
        
        # ============================================================
        # PRIORITY CHECK 1: List with research areas (MUST come first)
        # ============================================================
        if _detect_list_with_research_query(user_query) and self.faculty_ids:
            return self. handlers._list_all_faculty_with_research()
        
        # ============================================================
        # PRIORITY CHECK 2: List all faculty
        # ============================================================
        if _detect_list_query(user_query) and self.faculty_ids:
            return self.handlers._list_all_faculty_text()

        # ============================================================
        # PRIORITY CHECK 3: Direct faculty name mentions (inline like original)
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
                    return self.handlers._answer_for_specific_faculty(name, history=history)

                # 2. Token-level fuzzy matching ("jarry" ≈ "jerry")
                fuzzy_matches = sum(1 for qt in query_tokens for nt in name_tokens 
                                    if _similarity(qt, nt) > 0.70)

                if fuzzy_matches >= 2:
                    print("TOKEN FUZZY MATCH:", name)
                    return self. handlers._answer_for_specific_faculty(name, history=history)

        # ============================================================
        # PRIORITY CHECK 4: Handle affirmative/negative responses (inline like original)
        # ============================================================
        affirmative_patterns = [
            r'^(yes|yeah|yep|yup|sure|ok|okay|alright|please|yes please|sure thing)\. ?! ? $',
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
            last = self.conversation_memory.get("last_retrieved")
            if last and len(last) > 0:
                faculty_name = last[0]["name"]
                return self.handlers._answer_followup_fact(faculty_name, user_query, history=history)

        # Handle negative responses
        if is_negative:
            return "No problem!   Feel free to ask me about other faculty members, research areas, or anything else about the BSU CS graduate program.   How else can I help you?"

        # ============================================================
        # PRIORITY CHECK 5: Query classification for remaining queries
        # ============================================================
        query_type = self. handlers. classify_query_type(user_query. lower())

        # Retrieve last professor if available
        last = self.conversation_memory.get("last_retrieved")
        last_prof = last[0]["name"] if last and len(last) > 0 else None

        if query_type == "followup_person" and last_prof:
            return self.handlers._answer_followup_fact(last_prof, user_query, history=history)

        if query_type == "general_concept":  
            # Not a follow-up → do NOT use fact mode
            return self.handlers._answer_concept_definition(user_query)

        # ============================================================
        # PRIORITY CHECK 6: Normal RAG retrieval
        # ============================================================
        retrieved = self.retriever.retrieve_faculty(user_query, top_k=top_k)
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

        rag_system_prompt = get_rag_prompt(faculty_context)

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

        return self.groq_client.query(messages, max_tokens=800)