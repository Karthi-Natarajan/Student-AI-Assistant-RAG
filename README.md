# Student AI Assistant (RAG)

A Streamlit-based Generative AI app that allows users to upload any PDF and ask questions.
It uses RAG (Retrieval-Augmented Generation) to retrieve relevant document chunks and generate answers grounded in the PDF content.

## Features
- Upload any PDF
- Ask questions from the PDF
- Suggested universal questions (summary, key points, skills, interview questions, LinkedIn post)
- Shows retrieved context for transparency

## Tech Stack
- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers Embeddings
- FAISS Vector Store
- PyPDF

## How it works (RAG Pipeline)
PDF → Chunking → Embeddings → FAISS Vector Search → LLM Answer

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
