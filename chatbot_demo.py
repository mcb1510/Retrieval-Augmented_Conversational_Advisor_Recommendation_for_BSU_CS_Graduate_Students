# chatbot_demo.py
# This is the main file that creates the chatbot user interface
# It uses Streamlit to create a web-based chat interface

import streamlit as st
#from response_engine_1 import ResponseEngine
from engine import ResponseEngine
import os


# ==================== PAGE CONFIGURATION ====================
# This sets up how the webpage looks and behaves

st.set_page_config(
    page_title="BSU Advisor AI",  # Title shown in browser tab
    layout="centered",  # Content is centered on page (not full width)
    page_icon="assets/bsu_logo.png",  # Icon in browser tab
    initial_sidebar_state="collapsed"  # Sidebar is open when page loads
)

# ==================== CUSTOM STYLING ====================
# This adds custom CSS to make the chat look better

st.markdown("""
    <style>
    /* Style for chat message bubbles */
    .stChatMessage {
        padding: 1rem;  /* Space inside messages */
        border-radius: 0.5rem;  /* Rounded corners */
    }
    /* Style for main content area */
    .main {
        padding: 2rem;  /* Space around content */
    }
    </style>
""", unsafe_allow_html=True)  # Allow HTML/CSS in markdown

# ==================== HEADER ====================
# Logo and title at the top of the page

# Create columns for logo + title layout
col1, col2 = st.columns([1, 7])

with col1:
    # BSU Logo
    st.image("assets/bsu_logo.png", width=80)

with col2:
    st.title("BSU Graduate Advisor AI")
    st.caption("Your intelligent assistant for BSU CS graduate advising")

# ==================== SIDEBAR ====================
# The sidebar contains information about the project

with st.sidebar:
    # Section 1: About current demo
    st.header("Not sure where to start?")
    st.markdown("""
        Not sure how to choose a graduate advisor? You are not alone.

        This assistant helps you explore faculty, research areas, and advising options.
        There is no right way to start, just ask what you are curious about.
        """)
    st.markdown("""
    
    **Try asking:**
    - "Hello, how are you?"
    - "List all faculty with their research areas."
    - "What faculty does AI research?"
    - "Tell me about Dr. Xinyi Zhou"
    - "How do I choose an advisor?"
    """)
    
    st.divider()  # Horizontal line separator
    

# ==================== API TOKEN CHECK ====================
# Make sure the HuggingFace API token is loaded before continuing

# ==================== API TOKEN CHECK ====================
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found!")
    st.info("Please create a .env file with your Groq API key inside a .env file")
    st.stop()


# ==================== SESSION STATE INITIALIZATION ====================
# Session state keeps data persistent across page reloads
# Think of it as the chatbot's "memory"

# Initialize message history
if "messages" not in st.session_state:
    # If this is the first time loading the page, create empty message list
    st.session_state.messages = []
    
    # Add a welcome message from the assistant
    welcome_msg = {
        "role": "assistant",  # This message is from the AI
        "content": "Hi! I'm your BSU Graduate Advisor AI. I can help you learn about CS faculty, their research areas, availability, and guide you through the advisor selection process. What would you like to know?"
    }
    st.session_state.messages.append(welcome_msg)

# Initialize the response engine (our AI)
if "generator" not in st.session_state:
    # If this is first time, create the ResponseEngine
    with st.spinner("Initializing AI assistant..."):  # Show loading message
        st.session_state.generator = ResponseEngine()  # Create the AI engine

# ==================== DISPLAY CHAT HISTORY ====================
# Show all previous messages in the conversation

for message in st.session_state.messages:
    # For each message in history, display it in a chat bubble
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.write(message["content"])  # The actual message text

# ==================== CHAT INPUT ====================
# This is where the user types their question

if user_query := st.chat_input("Ask me anything about graduate advising at BSU..."):
    # The := is "walrus operator" - assigns AND checks in one line
    # This runs when user presses Enter after typing
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.write(user_query)
    
    # Add user's message to history
    # This ensures the AI remembers what the user just asked
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    
    # STEP 3: Generate AI response
    with st.chat_message("assistant"):
        # Show "thinking" spinner while waiting for response
        with st.spinner("Thinking..."):
            # Call the response engine to generate an answer
            # Pass conversation history so AI has context
            answer = st.session_state.generator.ask(
                user_query,  # The current question
                history=st.session_state.messages[:-1],  # All previous messages (not including the one we just added)
                use_rag=True  # Enable RAG for better answers
            )
        # Display the AI's response
        st.write(answer)
    
    # Add AI's response to history
    # This ensures future responses remember what the AI said
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ==================== FOOTER ====================
# Bottom of page with project info

st.markdown("---")  # Horizontal line
col1, col2, col3 = st.columns(3)  # Create 3 columns for footer info

