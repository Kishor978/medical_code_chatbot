import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ask_openai_with_context(question: str, context: str, model: str = "gpt-4") -> str:
    # Ensure the chat history is initialized
    if "llm_chat_history" not in st.session_state:
        st.session_state.llm_chat_history = [
            {"role": "system", "content": "You are an expert in Revenue Cycle Management (RCM)."}
        ]

    # Structured prompt for current turn
    user_prompt = f"""
        You are a helpful assistant specialized in Revenue Cycle Management (RCM).

        Using the information below, answer the user's question. Make the answer concise and relevant to the question.
        You can use the context provided to formulate your response. Make the answer informative and natural. Don't include terms like "Based on the context" or "According to the context" in your response.

        Context:
        {context}

        Question:
        {question}

        If the answer is not in the context, say you don't know.
        """

    # Add user's prompt to message history
    st.session_state.llm_chat_history.append({"role": "user", "content": user_prompt})

    try:
        # Call the OpenAI chat completion API
        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.llm_chat_history,
            temperature=0.2,
            max_tokens=500
        )

        # Get and store the assistant's reply
        answer = response.choices[0].message.content.strip()
        st.session_state.llm_chat_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        return f"❌ OpenAI Error: {e}"
