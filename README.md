# AI Doctor at Home

A RAG-based medical assistant that answers health-related questions using a knowledge base of medical documents.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:
- Loads medical documents from PDF files
- Creates embeddings using Hugging Face's sentence transformers
- Stores vectors in a FAISS database for efficient retrieval
- Uses Groq's LLaMA model to generate accurate medical responses
- Provides a user-friendly Streamlit interface

## Setup

1. Install dependencies:
```bash
pip install streamlit langchain-groq langchain-huggingface faiss-cpu python-dotenv
```

2. Set up environment variables in `.env`:
```
GROQ_API_KEY=your_groq_api_key
```

3. Add medical PDF documents to the `data/` directory

## Project Structure

- `app.py` - Main Streamlit application
- `src/`
  - `loader.py` - PDF document loading and text splitting
  - `vectorstore.py` - FAISS vector database management
  - `rag_chain.py` - RAG chain assembly and prompt engineering

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your medical question in the text input
3. Get AI-generated answers based on the medical knowledge base

## Features

- 📚 Local document storage and retrieval
- 🔍 Semantic search using FAISS
- 🤖 LLaMA 3.1 (8B) model for response generation
- 💾 Persistent vector storage
- ⚡ Fast response times with Groq

## Notes

- The system only answers based on provided medical documents
- Always consult healthcare professionals for medical advice
- First-time setup will take a few minutes to create embeddings