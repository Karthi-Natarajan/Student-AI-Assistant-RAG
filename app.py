import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.set_page_config(page_title="Student AI Assistant (RAG)", page_icon="📄", layout="wide")

st.title("📄 Student AI Assistant (RAG)")
st.caption("Upload any PDF and ask questions. Uses RAG (Retrieval + LLM) for document-grounded answers.")


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=embed_model_name)


def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # 🔥 IMPORTANT FIX: if no chunks, stop before FAISS crashes
    if len(chunks) == 0:
        return None, len(documents), 0

    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)

    return vs, len(documents), len(chunks)


suggested_questions = [
    "Summarize this PDF in 5 lines.",
    "What is the main purpose of this document?",
    "List the key points from this PDF.",
    "Extract important dates, names, and titles.",
    "What skills or technologies are mentioned?",
    "Generate 5 interview questions from this PDF.",
    "Write a LinkedIn post based on this PDF.",
]


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_path = "temp.pdf"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Building vector database from your PDF..."):
        vectorstore, pages, chunks = build_vectorstore(temp_path)

    if vectorstore is None:
        st.error("❌ Could not extract text from this PDF (maybe it's scanned/image-based). Try another PDF.")
        st.stop()

    st.success(f"✅ PDF processed successfully | Pages: {pages} | Chunks: {chunks}")

    st.subheader("💡 Suggested Questions")
    cols = st.columns(3)
    for i, q in enumerate(suggested_questions):
        if cols[i % 3].button(q):
            st.session_state["query"] = q

    query = st.text_input("Ask a question from the PDF:", key="query")

    if query:
        with st.spinner("Retrieving answer..."):
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content[:1000] for d in docs])

            prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the given PDF."

Context:
{context}

Question: {query}

Answer:
"""

            llm = load_llm()
            answer = llm.invoke(prompt)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("📌 Retrieved Context"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"### Chunk {i}")
                st.write(d.page_content)

else:
    st.info("Upload a PDF to start.")
