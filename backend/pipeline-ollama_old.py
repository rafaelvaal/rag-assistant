# src/pipeline.py

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def ingest(pdf_path: str, persist_dir: str):
    text = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"Ingested {len(chunks)} chunks.")
    return db


def get_retriever(persist_dir: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": k})


def generate_answer(query: str, docs: list, model: str = "llama3.2") -> str:
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
    return Ollama(model=model).invoke(prompt)


def ask(query: str, persist_dir: str, k: int = 3) -> str:
    retriever = get_retriever(persist_dir, k=k)
    docs = retriever.invoke(query)
    return generate_answer(query, docs)


# --- Main ---
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\rafae\documentos\Data-Projects\RAG + LLM\rag-demo\data\2005.11401v4.pdf"
    VECTOR_DIR = r"C:\Users\rafae\documentos\Data-Projects\RAG + LLM\rag-demo\vectorstore"

    query = "What happens when the retrieval component collapses in RAG?"
    answer = ask(query, VECTOR_DIR)
    print(f"Q: {query}\nA: {answer}")