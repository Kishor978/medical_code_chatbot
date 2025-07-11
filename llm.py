import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def contextualize_question(current_question: str, chat_history: list) -> str:
    if len(chat_history) < 2:
        return current_question

    last_assistant_response = None
    # Iterate backwards to find the last assistant response
    for msg in reversed(chat_history):
        if msg["role"] == "assistant":
            last_assistant_response = msg["content"]
            break
    
    if last_assistant_response is None:
        return current_question

    rephrase_prompt = f"""
    The user just asked a follow-up question based on your last response.
    Your last response was: "{last_assistant_response}"
    The user's follow-up question is: "{current_question}"

    If the user's follow-up question seems vague (e.g., "explain more", "tell me more", "what about that?"),
    please rephrase it into a more specific question that incorporates details from your last response.
    For example, if your last response mentioned "patient registration" and the user said "explain more",
    you might rephrase it as "Explain more about patient registration in RCM."
    If the user's question is already specific, return it as is.
    Do not answer the question, only rephrase it if necessary.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for rephrasing questions."},
                {"role": "user", "content": rephrase_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        rephrased_question = response.choices[0].message.content.strip()
        if "explain more" in rephrased_question.lower() and rephrased_question.lower() == current_question.lower():
            return current_question
        return rephrased_question
    except Exception as e:
        print(f"Error contextualizing question: {e}")
        return current_question

def ask_openai_with_context(question: str, context: str, full_chat_history: list, model: str = "gpt-4") -> str:
    # Base system message for RCM expertise and conversational awareness
    messages_for_llm = [
        {"role": "system", "content": "You are an expert in Revenue Cycle Management (RCM). You can also answer questions about the current conversation history. If asked 'what was my previous question?' or similar, identify it from the provided chat history."}
    ]

    # Add a slice of the recent chat history for conversational memory
    # Exclude the very last message as it's the current user_input being processed.
    # We take up to the last 6 messages (3 user, 3 assistant pairs) before the current turn.
    relevant_history = full_chat_history[-6:-1] 

    for msg in relevant_history:
        messages_for_llm.append({"role": msg["role"], "content": msg["content"]})


    # Now, add the current user's prompt with RAG context
    current_user_prompt_with_context = f"""
    Using the provided context (if relevant) and your RCM expertise, answer the user's question.
    If the question is about the conversation history (e.g., "what was my previous question?" or "repeat what I said?"), please answer directly from the conversation history provided above.
    Do not include phrases like "Based on the context" or "According to the context."

    Context:
    {context}

    Question:
    {question}

    If the answer is not in the context and is not about the conversation history, say you don't know or that the information isn't available in your knowledge base.
    If the questions are not relevent to questions and say you don't know the answer directly don't hallucinate the answer. And don't use your knowledge for out of context questions.
    Also, if the question is vague or too general, ask the user to clarify or provide more details.
    """
    messages_for_llm.append({"role": "user", "content": current_user_prompt_with_context})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_for_llm,
            temperature=0.2,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"❌ OpenAI Error: {e}"