# backend/main.py
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QuestionRequest, AnswerResponse
from pipeline import ingest, ask
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Assistant API")

# CORS — permite que el frontend se comunique con el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "RAG Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    ingest(file_path, VECTOR_DIR)
    
    return {"message": f"{file.filename} uploaded and ingested successfully"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not os.path.exists(VECTOR_DIR):
        raise HTTPException(status_code=400, detail="No documents ingested yet")
    
    session_id = request.session_id or str(uuid.uuid4())
    answer = ask(request.question, VECTOR_DIR)
    
    return AnswerResponse(
        answer=answer,
        session_id=session_id
    )