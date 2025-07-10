# RCM Support Chatbot

This is a simple chatbot tool built to help users search through Revenue Cycle Management (RCM) content using natural language. It uses semantic search under the hood—so instead of doing keyword matching, it understands the meaning of your question and fetches the most relevant pieces of information.

We built it using FAISS for fast vector search, Sentence Transformers for turning text into embeddings, and Streamlit for the user interface.

---

## What it does

- Lets you ask questions in plain English about your RCM data.
- Behind the scenes, it converts your query into a vector using a transformer model and searches a pre-built FAISS index.
- Displays the most relevant results from your documents in a chat-like interface.
- Keeps track of your conversations so you can start new chats or refer to older ones.

---


