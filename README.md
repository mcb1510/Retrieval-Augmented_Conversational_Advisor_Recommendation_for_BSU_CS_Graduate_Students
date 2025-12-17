# Retrieval-Augmented Conversational Advisor Recommendation for BSU CS Graduate Students
An intelligent chatbot assistant that helps Computer Science students at Boise State University find suitable research advisors based on their interests, skills, and goals.<br>
([Live Demo](https://graduate-advisor-bsu.streamlit.app/))
([DataBase](https://docs.google.com/spreadsheets/d/1xVY4-tXLUuDhc-K8q66GNjJL-zQWyEO1Iz0_6h4XWgU/edit?gid=0#gid=0))
## Overview

This project uses **Retrieval-Augmented Generation (RAG)** with the Groq API (Llama 3.3) to provide personalized faculty recommendations and answer questions about BSU CS graduate programs. The system retrieves relevant faculty profiles using semantic search and generates contextual, conversational responses.

## Features

-  **Semantic Faculty Search** - Find professors based on research interests using BGE-Large embeddings
-  **Conversational Interface** - Natural follow-up questions and context-aware responses
-  **Faculty Listings** - List all faculty with or without research areas
-  **Fuzzy Name Matching** - Handle misspelled faculty names intelligently
-  **Query Classification** - Automatically detect follow-up questions vs. new queries
-  **Conversation Memory** - Maintains context across multiple interactions
-  **Streamlit UI** - Clean, user-friendly web interface

## Project Structure

```
graduate_AI/
├── README.md                      # This file
├── .env                          # Environment variables (API keys)
├── config.py                     # Configuration settings
├── chatbot_demo.py              # Main Streamlit application
├── response_engine_legacy.py    # Legacy monolithic engine (backup)
│
├── engine/                       # Modular engine package
│   ├── __init__.py              # Package initialization
│   ├── response_engine.py       # Main response engine class
│   ├── rag_generator.py         # RAG answer generation logic
│   ├── handlers.py              # Query type handlers
│   ├── retrieval.py             # Faculty profile retrieval
│   ├── utils.py                 # Utility functions
│   └── prompts.py               # LLM prompt templates
│
├── assets/                       # Static assets
│   └── bsu_logo.png             # BSU logo for UI
│
└── data/                         # Faculty data (generated)
    ├── embeddings.npy           # Faculty profile embeddings
    ├── faculty_ids.json         # Faculty names
    └── faculty_texts.json       # Faculty profile texts
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))

### Setup

1. **Clone the repository**
   ```bash
   cd graduate_AI
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit requests python-dotenv sentence-transformers scikit-learn numpy
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Prepare faculty data**
   
   Ensure you have the following files in your project root:
   - `embeddings.npy` - Pre-computed BGE-Large embeddings
   - `faculty_ids.json` - List of faculty names
   - `faculty_texts.json` - Faculty profile descriptions

## Usage

### Running the Application

```bash
streamlit run chatbot_demo.py
```

The app will open in your browser at `http://localhost:8501`

### Example Queries

- **"List all faculty with their research areas"**
- **"Tell me about Dr. Xinyi Zhou"**
- **"What faculty does AI research?"**
- **"I'm interested in machine learning and computer vision"**
- **"Where is her office?"** (follow-up question)
- **"What is trustworthy AI?"** (concept explanation)

## File Descriptions

### Core Application

- **`chatbot_demo.py`** - Main Streamlit application with UI components and chat interface

### Engine Package (`engine/`)

- **`__init__.py`** - Exports ResponseEngine for external use
- **`response_engine.py`** - Main engine orchestrating Groq API, retrieval, and handlers
- **`rag_generator.py`** - RAG pipeline with priority-based query handling
- **`handlers.py`** - Query handlers for different interaction types:
  - List all faculty
  - Specific faculty information
  - Follow-up questions
  - Concept definitions
- **`retrieval.py`** - Faculty profile retrieval using semantic embeddings
- **`utils.py`** - Utility functions:
  - String similarity matching
  - Query intent detection
  - Query expansion with synonyms
- **`prompts.py`** - LLM prompt templates for different scenarios

### Configuration

- **`config.py`** - Central configuration for model names, API endpoints, and synonyms
- **`.env`** - Environment variables (not committed to git)

### Legacy Code

- **`response_engine_legacy.py`** - Original monolithic implementation (kept for reference)

### Data Files

- **`embeddings.npy`** - 1024-dimensional BGE-Large embeddings for faculty profiles
- **`faculty_ids.json`** - Ordered list of faculty names
- **`faculty_texts.json`** - Full text profiles for each faculty member

## Architecture

### RAG Pipeline

1. **Query Processing**
   - Detect special queries (list all, specific faculty names)
   - Expand with domain synonyms (AI → artificial intelligence, ML, deep learning)

2. **Retrieval**
   - Encode query using BGE-Large
   - Compute cosine similarity with faculty embeddings
   - Return top-k most relevant profiles

3. **Context Injection**
   - Build system prompt with retrieved faculty profiles
   - Include conversation history for context

4. **Generation**
   - Query Groq API with Llama 3.3-70B
   - Generate conversational, context-aware response

### Query Priority System

The system handles queries in priority order:

1. **List queries** - "List all faculty with research areas"
2. **Direct name matches** - Token-level and fuzzy matching
3. **Affirmative/Negative** - "yes", "no", "sure"
4. **Query classification** - Follow-up vs. concept vs. new query
5. **Normal RAG** - Semantic retrieval and generation

## Technologies Used

- **Groq API** - Fast LLM inference (Llama 3.3-70B)
- **Sentence Transformers** - BGE-Large-EN-v1.5 for embeddings
- **Streamlit** - Web UI framework
- **NumPy** - Numerical operations for embeddings
- **Scikit-learn** - Cosine similarity and normalization

## Configuration

Edit `config.py` to customize:

- Model name
- Embedding model
- File paths
- Research area synonyms

## API Rate Limits

The Groq API has rate limits. The engine includes:
- Automatic retry logic (3 attempts)
- Exponential backoff on rate limit errors
- Graceful error messages

## Troubleshooting

### "No GROQ_API_KEY found"
- Ensure `.env` file exists with `GROQ_API_KEY=your_key`
- Check that python-dotenv is installed

### "RAG resources not loaded"
- Verify `embeddings.npy`, `faculty_ids.json`, `faculty_texts.json` exist
- Check file paths in `config.py`

### Follow-up questions not working
- Ensure conversation history is being passed correctly
- Check that `conversation_memory` is shared across components

Rigth now working properly using response_engine_legacy, module implementation got a bug after a commit that I haven't been able to fix.


