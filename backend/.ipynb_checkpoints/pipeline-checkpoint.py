import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


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


def generate_answer(query: str, docs: list) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    response = llm.invoke(prompt)
    return response.content


def ask(query: str, persist_dir: str, k: int = 3) -> str:
    retriever = get_retriever(persist_dir, k=k)
    docs = retriever.invoke(query)
    return generate_answer(query, docs)