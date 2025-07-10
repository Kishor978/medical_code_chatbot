import streamlit as st
from openai import OpenAI

# Load .env file variables into environment
api_key = st.secrets["OPENAI_API_KEY"]

# Create the OpenAI client
client = OpenAI(api_key=api_key)

def ask_openai_with_context(question: str, context: str, model: str = "gpt-3.5-turbo") -> str:
    prompt = f"""
        You are a helpful assistant specialized in Revenue Cycle Management (RCM).

        Using the information below, answer the user's question.

        Context:
        {context}

        Question:
        {question}

        If the answer is not in the context, say you don't know.
        """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in RCM."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI Error: {e}"
