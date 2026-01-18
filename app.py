import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Student AI Assistant (RAG)", page_icon="📚", layout="wide")

st.title("📚 Student AI Assistant (RAG)")
st.caption("Upload any PDF and ask questions. Uses RAG for document-grounded answers.")

embed_model = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=embed_model)


def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    clean_chunks = []
    for c in chunks:
        if c.page_content and c.page_content.strip():
            clean_chunks.append(c)

    if len(clean_chunks) == 0:
        return None, 0, 0

    embeddings = load_embeddings()
    vs = FAISS.from_documents(clean_chunks, embeddings)

    return vs, len(documents), len(clean_chunks)


suggested_questions = [
    "Summarize this PDF in 5 lines.",
    "What is the main purpose of this document?",
    "List the key points from this PDF.",
    "Extract important dates, names, and titles.",
    "What skills or technologies are mentioned?"
]

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Building vector database from your PDF..."):
        vectorstore, pages, chunks = build_vectorstore("temp.pdf")

    if vectorstore is None:
        st.error("❌ No readable text found in this PDF. It may be scanned/image-based. Try another PDF.")
        st.stop()

    st.success(f"✅ PDF processed successfully | Pages: {pages} | Chunks: {chunks}")

    st.subheader("💡 Suggested Questions")
    cols = st.columns(2)
    for i, q in enumerate(suggested_questions):
        if cols[i % 2].button(q):
            st.session_state["query"] = q

    query = st.text_input("Ask a question from the PDF:", key="query")

    if query:
        with st.spinner("Searching relevant PDF chunks..."):
            docs = vectorstore.similarity_search(query, k=3)

        st.subheader("📌 Retrieved Context")
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(d.page_content)

        st.info("✅ Retrieval works! Now we will connect LLM in next step.")

else:
    st.info("Upload a PDF to start.")
