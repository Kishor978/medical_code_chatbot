# app.py
import streamlit as st # Import the Streamlit library for creating web applications
from backend import load_faiss, embed_query, search_faiss, get_results # Import functions from a 'backend' module
from llm import ask_openai_with_context # Import the function for interacting with the LLM from an 'llm' module

# Set basic page configuration for the Streamlit app
st.set_page_config(page_title="RCM Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS for better alignment and potential future styling ---
st.markdown("""
<style>
    /* Adjust the vertical alignment of the image with the title */
    .stImage {
        display: flex;
        align-items: center; /* Vertically center the image within its container */
        height: 100%; /* Ensure the image container takes full height of the column */
    }
    /* You might need to inspect the exact class for the title if further alignment is needed */
    /* .st-emotion-cache-10trblm { align-items: center; } /* Example for a specific title class */
</style>
""", unsafe_allow_html=True)


# --- Load FAISS Index ---
# Check if the FAISS index is already loaded in the session state
if "faiss_index" not in st.session_state:
    try:
        # Attempt to load the FAISS index and its associated metadata
        st.session_state.faiss_index, st.session_state.metadata = load_faiss()
    except Exception as e:
        # If loading fails, display an error message and stop the application
        st.error(f"Failed to load FAISS index: {e}")
        st.stop() # Halts the script execution

# --- Chat Message Management ---
# Initialize the chat messages list in the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom Header with Larger Logo and Title ---
# Use columns to place the logo and title side-by-side
# Adjust column ratios to give more space to the logo
col1, col2 = st.columns([0.15, 0.85]) # Increased ratio for col1 to make logo wider

with col1:
    # Display the logo. Increased width to match the title's visual size.
    # You might need to experiment with this 'width' value based on your logo's aspect ratio
    # and the font size of the title. A value between 80-120px is often a good starting point.
    st.image("logo.png", width=100) # Using the newly uploaded image file name

with col2:
    st.title("RCM Support Chatbot") # Display the main title of the chatbot

# --- Display Previous Messages ---
# Iterate through the stored messages and display them in chat bubbles
for msg in st.session_state.messages:
    # Determine the avatar based on the role
    avatar_icon = "👤" if msg["role"] == "user" else "🤖" # User icon for user, robot icon for assistant
    with st.chat_message(msg["role"], avatar=avatar_icon): # Pass the custom avatar
        st.markdown(msg["content"]) # Render the message content using Markdown

# --- User Input ---
# Create a chat input box at the bottom of the page
user_input = st.chat_input("Ask your Revenue Cycle Management question...")

# --- Process User Input ---
# This block executes only when the user submits a new message
if user_input:
    # 1. Save and Display User Message
    # Append the user's message to the session state messages list
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"): # Use the custom user avatar
        st.markdown(user_input) # Display the user's message immediately

    # 2. Run RAG (Retrieval-Augmented Generation) + GPT Process
    try:
        # Show a spinner to indicate that the chatbot is processing the request
        with st.spinner("Searching knowledge base and generating answer..."):
            # Embed the user's query into a vector
            query_vector = embed_query(user_input)

            # Search the FAISS index for the most similar documents
            distances, indices = search_faiss(st.session_state.faiss_index, query_vector)

            # Retrieve the actual text content of the documents based on the indices
            retrieved_texts = get_results(indices, st.session_state.metadata)

            # Combine the retrieved texts to form the context for the LLM
            context = "\n\n---\n\n".join(retrieved_texts)

            # Ask the OpenAI model a question, providing the user's query and the retrieved context
            response = ask_openai_with_context(user_input, context)
    except Exception as e:
        # If any error occurs during the RAG process, capture it and provide an error message
        response = f"⚠️ An error occurred:\n\n`{e}`"

    # 3. Save and Display Assistant Message
    # Append the assistant's response to the session state messages list
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"): # Use the custom assistant avatar
        st.markdown(response) # Display the assistant's response
