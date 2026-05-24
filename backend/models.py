# backend/models.py
from pydantic import BaseModel
from typing import Optional

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []