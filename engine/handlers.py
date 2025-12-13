# engine/handlers.py
from engine.prompts import *
from engine.utils import _similarity


class QueryHandlers:
    """Handles different types of user queries."""
    
    def __init__(self, retriever, groq_client):
        self.retriever = retriever
        self.groq = groq_client
        self.conversation_memory = {"last_query": None, "last_retrieved": None}
    
    def _list_all_faculty_text(self):
        """Return a human readable list of all faculty names."""
        if not self.retriever.faculty_ids:
            return "I do not have any faculty data loaded right now."
        
        # Clear conversation memory since we're starting fresh
        self. conversation_memory = {
            "last_query": None,
            "last_retrieved":   None
        }

        lines = [f"- {name}" for name in self.retriever.faculty_ids]
        return (
            "Here is the list of CS faculty I know about:\n\n"
            + "\n". join(lines)
            + "\n\nYou can ask me about any specific person, or tell me your interests and I will recommend a few advisors."
        )

    def _list_all_faculty_with_research(self):
        """Return faculty list with research areas."""
        if not self.retriever.faculty_ids or not self.retriever.faculty_texts:
            return "I do not have any faculty data loaded right now."
        
        self.conversation_memory = {"last_query": None, "last_retrieved": None}

        lines = []
        for name, profile in zip(self.retriever.faculty_ids, self.retriever.faculty_texts):
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
        if not self.retriever. faculty_ids or not self.retriever.faculty_texts:
            return "I could not load the faculty profiles right now."

        try:
            idx = self.retriever.faculty_ids.index(faculty_name)
        except ValueError:
            return "I could not find that faculty in my profiles."

        profile = self.retriever.faculty_texts[idx]
        
        # Store memory for follow-up questions
        self.conversation_memory["last_query"] = faculty_name
        self.conversation_memory["last_retrieved"] = [{
            "name": faculty_name,
            "profile_text": profile
        }]
        
        prompt = get_faculty_prompt(faculty_name, profile)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Tell me about {faculty_name} as a potential advisor for me.  "}
        ]

        # Optionally include short history
        if history:
            for msg in history[-3:]: 
                messages.insert(1, {
                    "role": msg["role"],
                    "content":   msg["content"]
                })

        return self. groq.query(messages, max_tokens=600)
    
    def _answer_followup_fact(self, faculty_name, user_query, history=None):
        """
        Extracts specific factual information (like office, email, interests)
        from the stored faculty profile with a conversational tone.
        """

        # Locate the faculty in the dataset
        try:
            idx = self.retriever.faculty_ids.index(faculty_name)
        except ValueError:
            return "I couldn't find that faculty member anymore."

        profile = self.retriever.faculty_texts[idx]
        
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
                    content = content[: 300] + "..."
                conversation_context += f"{role}: {content}\n\n"

        # Build a conversational factual prompt
        prompt = get_followup_prompt(faculty_name, profile, conversation_context)
        # Replace placeholder in prompt
        prompt = prompt.replace("{user_query}", user_query)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]

        return self.groq.query(messages, max_tokens=200)
    
    def _is_followup(self, query:  str) -> bool:
        q = query.lower()
        for name in self.retriever.faculty_ids:
            # direct substring check
            if name.lower() in q:
                return False
            sim = _similarity(q, name. lower())
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
        messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": query},
        ]

        result = self.groq.query(messages, max_tokens=5).lower().strip()

        # Safety normalization
        if "follow" in result:
            return "followup_person"
        if "concept" in result:
            return "general_concept"
        if "professor" in result or "new" in result:
            return "new_professor"

        return "general_concept"  # safe fallback

    def _answer_concept_definition(self, query):
        prompt = CONCEPT_PROMPT.format(query=query)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        return self.groq.query(messages, max_tokens=300)