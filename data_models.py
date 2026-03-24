# data_models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class BookData(BaseModel):
    id: str
    title: str
    author: str
    genre: str
    tags: List[str] = []
    average_rating: float = 0.0
    cover_url: str = ""
    description: str = ""

class UserInteraction(BaseModel):
    user_id: int
    book_id: str
    rating: float = 0.0
    status: str
    timestamp: Optional[datetime] = None

class RecommendationRequest(BaseModel):
    user_id: int
    candidate_books: List[BookData]
    limit: int = 10

class RecommendationResponse(BaseModel):
    recommendations: List[BookData]
    confidence_scores: List[float]
    message: str = "Гибридные рекомендации"
    system_type: str = "гибридная"

# Модели для чата и других запросов
class ChatRequest(BaseModel):
    query: str
    user_id: int
