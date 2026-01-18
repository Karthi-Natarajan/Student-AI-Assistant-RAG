from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

query = "Explain this project in simple words."

docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([d.page_content[:600] for d in docs])

prompt = f"""
You are a helpful assistant.
Explain the project clearly in 6-8 lines using ONLY the context.

Context:
{context}

Question: {query}

Answer:
"""

result = llm.invoke(prompt)

print("QUESTION:", query)
print("\nANSWER:\n")
print(result)
