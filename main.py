# main.py
import asyncio
import os
import re
import json
import base64
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
from dotenv import load_dotenv

from data_models import *
from data_collector import DataCollector
from recommender import BookRecommender

load_dotenv()

# ==================== ПРЕОБРАЗОВАНИЕ URL ДЛЯ ASYNCPG ====================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")
# Если URL не содержит +asyncpg, добавляем
if "postgresql" in DATABASE_URL and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    print(f"✅ Преобразованный DATABASE_URL для asyncpg: {DATABASE_URL}")

# ==================== ЗАГРУЗКА ЛОКАЛЬНЫХ КНИГ ====================
def generate_global_id(title: str, author: str) -> str:
    """Генерирует globalId так же, как в Android-приложении."""
    base = f"{title}|{author}".lower()
    # Оставляем только буквы (русские/английские), цифры и разделитель '|'
    return re.sub(r'[^a-zа-я0-9|]', '', base)

LOCAL_BOOKS = []
try:
    with open('local_books.json', 'r', encoding='utf-8') as f:
        raw_books = json.load(f)
    for i, book in enumerate(raw_books):
        # Генерируем id как globalId
        book['id'] = generate_global_id(book['title'], book['author'])
        book['average_rating'] = book.get('averageRating', book.get('average_rating', 0.0))
        if 'averageRating' in book:
            del book['averageRating']
        book['tags'] = []
        book['description'] = book.get('description', '')
        book['cover_url'] = book.get('cover', '')   # cover -> cover_url
        LOCAL_BOOKS.append(book)
    print(f"✅ Загружено {len(LOCAL_BOOKS)} локальных книг с глобальными ID")
except Exception as e:
    print(f"❌ Ошибка загрузки локальных книг: {e}")

# ==================== ФУНКЦИИ ПОИСКА (не используются в гибридных рекомендациях, но оставлены для чата) ====================
def search_local_books(criteria: dict) -> list:
    """Поиск по локальному JSON (возвращает в формате GoogleBook)."""
    genres = [g.lower() for g in criteria.get("genres", [])]
    keywords = [k.lower() for k in criteria.get("keywords", [])]
    min_rating = criteria.get("min_rating", 0)
    max_results = criteria.get("max_results", 10)

    results = []
    for book in LOCAL_BOOKS:
        score = 0
        book_genre = book.get("genre", "").lower()
        if genres and any(g in book_genre for g in genres):
            score += 2
        text = (book.get("title", "") + " " + book.get("description", "")).lower()
        for kw in keywords:
            if kw in text:
                score += 1
        if book.get("average_rating", 0) >= min_rating:
            score += 0.5
        if score > 0:
            results.append((book, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [book_to_google_book(book) for book, _ in results[:max_results]]

def book_to_google_book(book: dict) -> dict:
    """Конвертирует локальную книгу в формат GoogleBook."""
    cover = book.get("cover", "")
    return {
        "id": book["id"],
        "volumeInfo": {
            "title": book["title"],
            "authors": [book["author"]],
            "description": book.get("description", ""),
            "categories": [book.get("genre", "Unknown")],
            "averageRating": book.get("average_rating", 0),
            "ratingsCount": book.get("ratingsCount", 0),
            "imageLinks": {"thumbnail": cover} if cover else None,
            "language": book.get("language", "ru")
        }
    }

async def search_openlibrary(criteria: dict, limit: int = 5) -> list:
    """Поиск в OpenLibrary на основе критериев (возвращает в формате GoogleBook)."""
    query_parts = []
    if criteria.get("genres"):
        for g in criteria["genres"]:
            query_parts.append(f"subject:{g}")
    if criteria.get("keywords"):
        query_parts.append(" ".join(criteria["keywords"]))
    query = " ".join(query_parts) if query_parts else "popular"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openlibrary.org/search.json",
                params={"q": query, "limit": limit, "fields": "key,title,author_name,subject,ratings_average,cover_i"}
            )
            data = response.json()
            return [openlibrary_to_google_book(doc) for doc in data.get("docs", [])]
    except Exception as e:
        print(f"OpenLibrary error: {e}")
        return []

def openlibrary_to_google_book(doc: dict) -> dict:
    """Конвертирует OpenLibrary документ в GoogleBook формат."""
    cover_id = doc.get("cover_i")
    cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None
    return {
        "id": doc.get("key", ""),
        "volumeInfo": {
            "title": doc.get("title", ""),
            "authors": doc.get("author_name", []),
            "description": None,
            "categories": doc.get("subject", [])[:3],
            "averageRating": doc.get("ratings_average", 0),
            "ratingsCount": doc.get("ratings_count", 0),
            "imageLinks": {"thumbnail": cover_url} if cover_url else None,
            "language": "ru"
        }
    }

# ==================== ФУНКЦИИ ДЛЯ GIGACHAT ====================
async def get_gigachat_token(client_id: str, client_secret: str):
    credentials = f"{client_id}:{client_secret}"
    auth_header = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"scope": "GIGACHAT_API_PERS"}
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers=headers,
            data=data
        )
    if response.status_code != 200:
        print(f"Ошибка получения токена: {response.status_code} - {response.text}")
        raise Exception("Failed to get GigaChat token")
    token_data = response.json()
    return token_data["access_token"]

async def ask_gigachat(prompt: str, token: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": "Ты помощник, который преобразует запрос пользователя о книгах в структурированный JSON для поиска. Ответ должен быть только JSON без лишнего текста. Формат: {\"genres\": [\"романтика\", \"фэнтези\"], \"keywords\": [\"легкое\", \"веселое\"], \"min_rating\": 3.5, \"max_results\": 10}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
    if response.status_code != 200:
        print(f"Ошибка GigaChat: {response.status_code} - {response.text}")
        raise Exception("GigaChat request failed")
    return response.json()

# ==================== ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ====================
app = FastAPI(
    title="Гибридная рекомендательная система книг",
    description="API для гибридных рекомендаций (популярность + контент + семантика + коллаборативная) + Чат с AI",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_collector = DataCollector(DATABASE_URL)
recommender = BookRecommender()

@app.on_event("startup")
async def startup_event():
    await data_collector.init_db()
    # Загружаем локальные книги, если их ещё нет в БД
    all_books = await data_collector.get_all_books()
    if not all_books and LOCAL_BOOKS:
        await data_collector.add_books(LOCAL_BOOKS)
        print("✅ Локальные книги загружены в базу данных")
    # Строим рекомендательную систему
    all_interactions = await data_collector.get_all_interactions()
    all_books = await data_collector.get_all_books()
    await asyncio.to_thread(recommender.build, all_interactions, all_books)
    print("✅ Рекомендательная система построена")

# ==================== ЭНДПОИНТЫ (те же, что были) ====================
# ... (все эндпоинты остаются без изменений) ...

if __name__ == "__main__":
    print("🚀 Запуск ГИБРИДНОЙ рекомендательной системы версия 3.0 с AI-чатом...")
    print("📊 Система: Гибридная (популярность + контент + семантика + коллаборативная) + AI чат")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
