# engine/utils.py
import re
from difflib import SequenceMatcher


def _similarity(a:  str, b: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, a. lower(), b.lower()).ratio()


def _detect_list_query(text: str) -> bool:
    """Detect if the user is asking for a list of all faculty using regex-based intent detection."""
    q = text.lower()
    patterns = [
        r"(list|show|display|give|tell me).*(all|everyone|every).*(faculty|professors?  )",
        r"(list|show|display).*(faculty|professors? )",
        r"(who are|what are).*(all|everyone).*(faculty|professors? )",
        r"(all|everyone).*(faculty|professors?  )",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def _detect_list_with_research_query(text: str) -> bool:
    """Detect if user wants faculty list WITH research areas."""
    q = text.lower()
    return bool(re.search(r"(list|show).*(faculty|professors?  ).*(research|areas?|interests?)", q))


class QueryProcessor:
    """Expands queries with domain-specific synonyms"""
    
    def __init__(self):
        self.research_synonyms = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'ml': ['machine learning', 'deep learning', 'statistical learning'],
            'security':  ['cybersecurity', 'privacy', 'cryptography', 'network security'],
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