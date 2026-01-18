from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


pdf_path = "data/sample_notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

clean_chunks = []
for c in chunks:
    if c.page_content and c.page_content.strip():
        clean_chunks.append(c)

if len(clean_chunks) == 0:
    raise ValueError("No readable text found in the PDF")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(clean_chunks, embeddings)

vectorstore.save_local("faiss_index")

print("FAISS index created and saved successfully")
print(f"Pages loaded: {len(documents)}")
print(f"Chunks stored: {len(clean_chunks)}")
