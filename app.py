import streamlit as st
from backend import load_faiss, embed_query, search_faiss, get_results
from llm import ask_openai_with_context, contextualize_question

st.set_page_config(page_title="RCM Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stImage {
    display: flex;
    align-items: center;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

if "faiss_index" not in st.session_state:
    try:
        st.session_state.faiss_index, st.session_state.metadata = load_faiss()
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([0.15, 0.85])

with col1:
    st.image("logo.png", width=100)

with col2:
    st.title("RCM Support Chatbot")

for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your Revenue Cycle Management question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    try:
        with st.spinner("Searching knowledge base and generating answer..."):
            # Contextualize the user's question for better RAG search
            # Pass a copy of the messages list to avoid modification issues
            contextualized_query = contextualize_question(user_input, st.session_state.messages.copy())
            
            query_vector = embed_query(contextualized_query)

            distances, indices = search_faiss(st.session_state.faiss_index, query_vector)

            retrieved_texts = get_results(indices, st.session_state.metadata)

            context = "\n\n---\n\n".join(retrieved_texts)

            # Pass original user_input, retrieved context, AND the full chat history to the LLM
            response = ask_openai_with_context(user_input, context, st.session_state.messages.copy())
    except Exception as e:
        response = f"⚠️ An error occurred:\n\n`{e}`"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)