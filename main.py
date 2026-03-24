# main.py
import asyncio
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import json
import base64
import uuid
from dotenv import load_dotenv
import re

from data_models import *
from data_collector import DataCollector
from recommender import BookRecommender

load_dotenv()

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
        book['cover_url'] = book.get('cover', '')   # преобразуем cover -> cover_url
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

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/dbname")
data_collector = DataCollector(DATABASE_URL)
recommender = BookRecommender()

# Стартовый event
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

# ==================== ЭНДПОИНТЫ ====================
@app.get("/")
async def root():
    return {
        "message": "Гибридная рекомендательная система книг с AI-чатом",
        "version": "3.0.0",
        "status": "работает",
        "system_type": "гибридная (популярность + контент + семантика + коллаборативная) + AI чат",
        "data_stats": await data_collector.get_all_data_stats()
    }

@app.post("/api/add_interaction_with_book")
async def add_interaction_with_book(interaction: UserInteraction, book_data: BookData):
    if interaction.book_id != book_data.id:
        raise HTTPException(status_code=400, detail="book_id in interaction must match book_data.id")
    try:
        book_dict = book_data.model_dump()
        await data_collector.add_interaction(
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            rating=interaction.rating,
            status=interaction.status,
            book_data=book_dict
        )
        # Перестраиваем рекомендательную систему
        all_interactions = await data_collector.get_all_interactions()
        all_books = await data_collector.get_all_books()
        await asyncio.to_thread(recommender.build, all_interactions, all_books)
        return {
            "status": "success",
            "message": "Взаимодействие и данные книги добавлены, система обновлена"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add_interaction")
async def add_interaction(interaction: UserInteraction):
    """Устаревший эндпоинт. Рекомендуется использовать /api/add_interaction_with_book."""
    try:
        # Проверяем, есть ли книга в метаданных
        all_books = await data_collector.get_all_books()
        if interaction.book_id not in [b["id"] for b in all_books]:
            raise HTTPException(status_code=400, detail="Книга неизвестна. Используйте /api/add_interaction_with_book")
        await data_collector.add_interaction(
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            rating=interaction.rating,
            status=interaction.status,
            book_data=None
        )
        all_interactions = await data_collector.get_all_interactions()
        all_books = await data_collector.get_all_books()
        await asyncio.to_thread(recommender.build, all_interactions, all_books)
        return {
            "status": "success",
            "message": "Взаимодействие добавлено, система обновлена"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
async def get_hybrid_recommendations(request: RecommendationRequest):
    try:
        user_id = request.user_id
        user_interactions = await data_collector.get_user_interactions(user_id)
        candidate_books = [book.model_dump() for book in request.candidate_books]

        print(f"👤 Пользователь {user_id}: {len(user_interactions)} взаимодействий")
        print(f"📚 Кандидатов: {len(candidate_books)} книг")

        recommendations, confidences = recommender.recommend(
            user_id=user_id,
            candidate_books=candidate_books,
            user_interactions=user_interactions,
            top_k=request.limit
        )

        recommended_books = [BookData(**book) for book in recommendations]

        return RecommendationResponse(
            recommendations=recommended_books,
            confidence_scores=confidences,
            training_data_size=len([i for i in user_interactions if i.get('rating',0)>0]),
            message="Гибридные рекомендации"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat_recommend")
async def chat_recommend(request: dict):
    """Обрабатывает текстовый запрос и возвращает книги через GigaChat и поиск."""
    query = request.get("query", "")
    user_id = request.get("user_id", 0)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key:
        criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
    else:
        try:
            async with httpx.AsyncClient(verify=False) as client:
                # Получение токена
                token_response = await client.post(
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    headers={
                        "Authorization": f"Basic {auth_key}",
                        "RqUID": str(uuid.uuid4()),
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    data={"scope": "GIGACHAT_API_PERS"}
                )
                if token_response.status_code != 200:
                    raise Exception(f"Failed to get token: {token_response.status_code}")
                token_data = token_response.json()
                access_token = token_data.get("access_token")

                # Запрос к GigaChat
                chat_response = await client.post(
                    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "GigaChat",
                        "messages": [
                            {"role": "system", "content": "Ты помощник, который преобразует запрос пользователя о книгах в структурированный JSON для поиска. Ответ должен быть только JSON без лишнего текста."},
                            {"role": "user", "content": f"Запрос: {query}\nФормат ответа: {{\"genres\": [\"жанр1\", \"жанр2\"], \"keywords\": [\"слово1\", \"слово2\"], \"min_rating\": 0.0, \"max_results\": 10}}"}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 512
                    }
                )
                chat_data = chat_response.json()
                if "choices" in chat_data and chat_data["choices"]:
                    criteria_text = chat_data["choices"][0]["message"]["content"]
                    start = criteria_text.find('{')
                    end = criteria_text.rfind('}') + 1
                    if start != -1 and end > start:
                        criteria = json.loads(criteria_text[start:end])
                    else:
                        criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
                else:
                    criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
        except Exception as e:
            print(f"GigaChat error: {e}")
            criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}

    # Поиск книг
    local_results = search_local_books(criteria)
    needed = max(0, criteria.get("max_results", 10) - len(local_results))
    open_library_results = await search_openlibrary(criteria, limit=needed) if needed > 0 else []
    combined = local_results + open_library_results
    return {"results": combined[:criteria.get("max_results", 10)]}

@app.get("/api/user/{user_id}/stats")
async def get_user_stats(user_id: int):
    interactions = await data_collector.get_user_interactions(user_id)
    ratings_count = len([i for i in interactions if i.get('rating', 0) > 0])
    quality = "базовая"
    if ratings_count >= 5:
        quality = "высокая"
    elif ratings_count >= 2:
        quality = "средняя"
    stats = await data_collector.get_user_stats(user_id)
    return {
        "user_id": user_id,
        "hybrid_system": True,
        "interactions": len(interactions),
        "ratings": ratings_count,
        "recommendation_quality": quality,
        **stats
    }

@app.get("/api/system/stats")
async def get_system_stats():
    stats = await data_collector.get_all_data_stats()
    return {
        "system_type": "гибридная (популярность + контент + семантика + коллаборативная) + AI чат",
        "data_collector": stats,
        "total_users": stats['unique_users'],
        "total_books": stats['unique_books'],
        "collaborative_filtering_ready": recommender.book_similarity is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_type": "гибридная + AI чат",
        "total_interactions": (await data_collector.get_all_data_stats())["total_interactions"]
    }

@app.post("/api/clear_user_data/{user_id}")
async def clear_user_data(user_id: int):
    try:
        success = await data_collector.clear_user_data(user_id)
        if success:
            # Перестраиваем систему после удаления
            all_interactions = await data_collector.get_all_interactions()
            all_books = await data_collector.get_all_books()
            await asyncio.to_thread(recommender.build, all_interactions, all_books)
            return {"status": "success", "message": f"Данные пользователя {user_id} очищены"}
        else:
            raise HTTPException(status_code=500, detail="Ошибка очистки данных")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Запуск ГИБРИДНОЙ рекомендательной системы версия 3.0 с AI-чатом...")
    print("📊 Система: Гибридная (популярность + контент + семантика + коллаборативная) + AI чат")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
