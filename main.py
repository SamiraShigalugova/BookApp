import asyncio
import os
import re
import json
import base64
import uuid
import random
import httpx
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from fastapi.responses import PlainTextResponse

from data_models import *
from data_collector import DataCollector, UserDB
from recommender import BookRecommender

load_dotenv()

# ==================== ПРЕОБРАЗОВАНИЕ URL ДЛЯ ASYNCPG ====================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")
if "postgresql" in DATABASE_URL and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    print(f"✅ Преобразованный DATABASE_URL для asyncpg: {DATABASE_URL}")

# ==================== МАППИНГ ЖАНРОВ ====================
GENRE_MAP = {
    "romance": ["романтика", "любовный роман"],
    "fantasy": ["фэнтези"],
    "science fiction": ["научная фантастика", "фантастика"],
    "mystery": ["детектив", "мистика"],
    "thriller": ["триллер"],
    "horror": ["ужасы"],
    "adventure": ["приключения"],
    "classics": ["классика"],
    "poetry": ["поэзия"],
    "biography": ["биография"],
    "history": ["история"],
    "philosophy": ["философия"],
    "self development": ["саморазвитие", "психология"],
    "business": ["бизнес"],
    "finance": ["финансы"],
}

def expand_genres(genres: List[str]) -> List[str]:
    """Расширяет список жанров, добавляя русские эквиваленты."""
    expanded = []
    for g in genres:
        g_lower = g.lower()
        expanded.append(g_lower)
        if g_lower in GENRE_MAP:
            expanded.extend(GENRE_MAP[g_lower])
    return list(set(expanded))

# ==================== ЗАГРУЗКА ЛОКАЛЬНЫХ КНИГ ====================
def generate_global_id(title: str, author: str) -> str:
    base = f"{title}|{author}".lower()
    return re.sub(r'[^a-zа-я0-9|]', '', base)

LOCAL_BOOKS = []
try:
    with open('local_books.json', 'r', encoding='utf-8') as f:
        raw_books = json.load(f)
    for book in raw_books:
        book['id'] = generate_global_id(book['title'], book['author'])
        book['average_rating'] = book.get('averageRating', book.get('average_rating', 0.0))
        if 'averageRating' in book:
            del book['averageRating']
        book['tags'] = []
        book['description'] = book.get('description', '')
        book['cover_url'] = book.get('cover', '')
        book['is_bestseller'] = book.get('is_bestseller', False) 
        LOCAL_BOOKS.append(book)
    print(f"✅ Загружено {len(LOCAL_BOOKS)} локальных книг")
except Exception as e:
    print(f"❌ Ошибка загрузки локальных книг: {e}")

# ==================== ФУНКЦИИ ПОИСКА ====================
def search_local_books(criteria: dict) -> list:
    """
    Поиск по локальным книгам.
    criteria может содержать:
      - specific_books: list[{"title":..., "author":...}]
      - genres: list[str] (на английском, будут расширены)
      - authors: list[str]
      - keywords: list[str]
      - min_rating: float
      - max_results: int
    """
    specific_books = criteria.get("specific_books", [])
    genres = criteria.get("genres", [])
    expanded_genres = expand_genres(genres) if genres else []
    keywords = [k.lower() for k in criteria.get("keywords", [])]
    authors = [a.lower() for a in criteria.get("authors", [])]
    min_rating = criteria.get("min_rating", 0)
    max_results = criteria.get("max_results", 20)

    # Если нет никаких критериев – случайные книги
    if not expanded_genres and not keywords and not authors and not specific_books:
        all_books = list(LOCAL_BOOKS)
        random.shuffle(all_books)
        return [book_to_google_book(book) for book in all_books[:max_results]]

    results = []
    for book in LOCAL_BOOKS:
        score = 0
        book_genre = book.get("genre", "").lower()
        book_title = book.get("title", "").lower()
        book_author = book.get("author", "").lower()
        book_desc = book.get("description", "").lower()
        avg_rating = book.get("average_rating", 0)

        # 1. Похожесть на конкретные книги (самый высокий вес)
        for spec in specific_books:
            spec_title = spec.get("title", "").lower()
            spec_author = spec.get("author", "").lower()
            if spec_title and spec_author:
                if spec_title in book_title or spec_author in book_author:
                    score += 10
            elif spec_title and spec_title in book_title:
                score += 8

        # 2. Жанры (с расширением)
        if expanded_genres:
            if any(g == book_genre for g in expanded_genres):
                score += 10
            elif any(g in book_genre for g in expanded_genres):
                score += 5

        # 3. Авторы
        if authors and any(a in book_author for a in authors):
            score += 5

        # 4. Ключевые слова (в заголовке, описании, жанре)
        if keywords:
            text = f"{book_title} {book_desc} {book_genre}"
            for kw in keywords:
                if kw in text:
                    score += 2

        # 5. Бонус за рейтинг
        if avg_rating >= min_rating:
            score += min(avg_rating / 2, 2)

        if score > 0:
            results.append((book, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [book_to_google_book(book) for book, _ in results[:max_results]]

def book_to_google_book(book: dict) -> dict:
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
    genres = criteria.get("genres", [])
    keywords = criteria.get("keywords", [])
    authors = criteria.get("authors", [])

    if not genres and not keywords and not authors:
        query = "popular"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://openlibrary.org/search.json",
                    params={"q": query, "limit": limit * 2, "fields": "key,title,author_name,subject,ratings_average,cover_i"}
                )
                data = response.json()
                docs = data.get("docs", [])
                random.shuffle(docs)
                return [openlibrary_to_google_book(doc) for doc in docs[:limit]]
        except Exception as e:
            print(f"OpenLibrary error: {e}")
            return []
    else:
        query_parts = []
        for g in genres:
            query_parts.append(f"subject:{g}")
        for a in authors:
            query_parts.append(f"author:{a}")
        if keywords:
            query_parts.append(" ".join(keywords))
        query = " ".join(query_parts)
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
_cached_token = None
_token_expiry = 0

async def get_gigachat_token(auth_key: str):
    headers = {
        "Authorization": f"Basic {auth_key.strip()}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "client_credentials",
        "scope": "GIGACHAT_API_PERS"
    }
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers=headers,
            data=data,
            timeout=30.0
        )
    if response.status_code != 200:
        error_details = f"status={response.status_code}, body={response.text}"
        raise Exception(f"GigaChat token error: {error_details}")
    token_data = response.json()
    return token_data["access_token"]

async def ask_gigachat(prompt: str, token: str, system_prompt: str = None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    if system_prompt is None:
        system_prompt = (
            "Ты – эксперт по книгам.\n"
            "Проанализируй запрос пользователя и верни ТОЛЬКО JSON в формате:\n"
            "{\n"
            "  \"specific_books\": [\n"
            "    {\"title\": \"Название конкретной книги\", \"author\": \"Автор\"}\n"
            "  ],\n"
            "  \"criteria\": {\n"
            "    \"genres\": [\"жанр1\", \"жанр2\"],\n"
            "    \"authors\": [\"автор1\"],\n"
            "    \"keywords\": [\"ключевое слово1\", \"ключевое слово2\"],\n"
            "    \"min_rating\": 0\n"
            "  }\n"
            "}\n"
            "Жанры указывай на английском: romance, fantasy, mystery, thriller, horror, adventure, classics, poetry, biography, history, philosophy, self development, business, finance.\n"
            "Если пользователь спрашивает про похожесть на книгу, укажи её в specific_books.\n"
            "Если просит 'лёгкое на вечер', добавь keywords: ['легкое', 'вечер'].\n"
            "Если просит 'популярное' или 'лучшее', установи min_rating: 4.0.\n"
            "Не добавляй пояснений, только JSON."
        )
    payload = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": system_prompt},
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
        raise Exception(f"GigaChat request failed: {response.status_code}")
    return response.json()

async def get_cached_token():
    global _cached_token, _token_expiry
    import time
    if _cached_token and time.time() < _token_expiry:
        return _cached_token
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key:
        raise Exception("GIGACHAT_AUTH_KEY is not set")
    token = await get_gigachat_token(auth_key)
    _cached_token = token
    _token_expiry = time.time() + 28 * 60
    print("✅ GigaChat token получен и закэширован")
    return token

# ==================== ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ====================
app = FastAPI(
    title="Гибридная рекомендательная система книг",
    description="API для гибридных рекомендаций + Чат с AI",
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
    all_books = await data_collector.get_all_books()
    print(f"📚 В базе до загрузки: {len(all_books)} книг")
    
    if LOCAL_BOOKS:
        await data_collector.add_books(LOCAL_BOOKS)
        print("✅ Локальные книги загружены в базу данных")
        all_books = await data_collector.get_all_books()
        print(f"📚 После загрузки в базе {len(all_books)} книг")
    else:
        print("⚠️ Нет локальных книг для загрузки")
    
    all_interactions = await data_collector.get_all_interactions()
    all_books = await data_collector.get_all_books()
    await asyncio.to_thread(recommender.build, all_interactions, all_books)
    print("✅ Рекомендательная система построена")

# ==================== ЭНДПОИНТЫ ПОЛЬЗОВАТЕЛЕЙ ====================
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    user_id: int  
    username: str
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@app.post("/api/register", response_model=UserResponse)
async def register(request: RegisterRequest):
    existing = await data_collector.get_user_by_username(request.username)
    if existing:
        raise HTTPException(status_code=400, detail="Имя пользователя уже занято")
    existing_email = await data_collector.get_user_by_email(request.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email уже используется")

    password_hash = hash_password(request.password)
    user_id = await data_collector.create_user(request.username, request.email, password_hash)
    user = await data_collector.get_user_by_id(user_id)
    return UserResponse(
        user_id=user.id, 
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        last_login=user.last_login
    )

@app.post("/api/login")
async def login(request: LoginRequest):
    user = await data_collector.get_user_by_username(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")
    if user.password_hash != hash_password(request.password):
        raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")
    await data_collector.update_last_login(user.id)
    return {"user_id": user.id, "username": user.username}

@app.get("/api/user/{user_id}/books")
async def get_user_books(user_id: int):
    interactions = await data_collector.get_user_interactions(user_id)
    books = {b["id"]: b for b in await data_collector.get_all_books()}
    result = []
    for inter in interactions:
        book = books.get(inter["book_id"])
        if book:
            result.append({
                "book_id": book["id"],
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "cover_url": book["cover_url"],
                "average_rating": book["average_rating"],
                "status": inter["status"],
                "user_rating": inter["rating"],
                "global_id": book["id"]
            })
    return result

class UserBookUpdate(BaseModel):
    global_id: str
    title: str
    author: str
    genre: str
    cover_url: str = ""
    description: str = ""
    status: str
    rating: float = 0.0

@app.post("/api/user/{user_id}/book")
async def update_user_book(user_id: int, request: UserBookUpdate):
    book_data = {
        "id": request.global_id,
        "title": request.title,
        "author": request.author,
        "genre": request.genre,
        "cover_url": request.cover_url,
        "description": request.description,
        "average_rating": 0.0,
        "tags": []
    }
    await data_collector.add_interaction(
        user_id=user_id,
        book_id=request.global_id,
        rating=request.rating,
        status=request.status,
        book_data=book_data
    )
    all_interactions = await data_collector.get_all_interactions()
    all_books = await data_collector.get_all_books()
    await asyncio.to_thread(recommender.build, all_interactions, all_books)
    return {"status": "ok"}

# ==================== ОСТАЛЬНЫЕ ЭНДПОИНТЫ ====================
@app.get("/")
async def root():
    return {
        "message": "Гибридная рекомендательная система книг с AI-чатом",
        "version": "3.0.0",
        "status": "работает",
        "system_type": "гибридная",
        "data_stats": await data_collector.get_all_data_stats()
    }

@app.post("/api/add_interaction_with_book")
async def add_interaction_with_book(interaction: UserInteraction, book_data: BookData):
    if interaction.book_id != book_data.id:
        raise HTTPException(status_code=400, detail="book_id mismatch")
    try:
        book_dict = book_data.model_dump()
        await data_collector.add_interaction(
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            rating=interaction.rating,
            status=interaction.status,
            book_data=book_dict
        )
        all_interactions = await data_collector.get_all_interactions()
        all_books = await data_collector.get_all_books()
        await asyncio.to_thread(recommender.build, all_interactions, all_books)
        return {"status": "success", "message": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books")
async def get_all_books():
    books = await data_collector.get_all_books()
    return books

@app.post("/api/add_interaction")
async def add_interaction(interaction: UserInteraction):
    try:
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
        return {"status": "success", "message": "ok"}
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
    query = request.get("query", "")
    user_id = request.get("user_id", 0)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    print(f"📝 Запрос чата: {query}, user_id={user_id}")

    # ================= УЛУЧШЕННЫЙ ПРОМПТ ДЛЯ GIGACHAT =================
    system_prompt = """
Ты — эксперт по книгам. Твоя задача — преобразовать запрос пользователя в структурированные критерии для поиска книг.

ВАЖНО: Всегда указывай жанры, даже если пользователь их не назвал. Для неявных запросов (например, "лёгкое на вечер") определи подходящие жанры: romance, comedy, adventure, short stories, poetry, self development.

Верни ТОЛЬКО JSON в формате:
{
  "specific_books": [
    {"title": "Название", "author": "Автор"}
  ],
  "criteria": {
    "genres": ["жанр1", "жанр2"],
    "authors": ["автор1"],
    "keywords": ["слово1", "слово2"],
    "min_rating": 0
  }
}

Жанры указывай на английском из списка: romance, fantasy, science fiction, mystery, thriller, horror, adventure, classics, poetry, biography, history, philosophy, self development, business, finance, comedy, short stories.

Если пользователь просит "лёгкое на вечер" — genres: ["romance", "comedy", "adventure"], keywords: ["легкое", "вечер"].
Если просит "популярное" или "лучшее" — установи min_rating: 4.0.
Если спрашивает про похожесть на книгу — укажи её в specific_books.

Не добавляй пояснений, только JSON.
"""
    try:
        token = await get_cached_token()
        response = await ask_gigachat(query, token, system_prompt)
        content = response["choices"][0]["message"]["content"]
        print(f"🤖 GigaChat: {content}")
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            specific_books = parsed.get("specific_books", [])
            criteria = parsed.get("criteria", {})
        else:
            specific_books = []
            criteria = {}
    except Exception as e:
        print(f"❌ GigaChat error: {e}")
        specific_books = []
        criteria = {"keywords": query.lower().split()[:5]}

    # ================= ПОИСК =================
    results = []

    # 1. Поиск по конкретным книгам
    for spec in specific_books:
        for book in LOCAL_BOOKS:
            if (spec.get("title", "").lower() in book["title"].lower() or
                spec.get("author", "").lower() in book["author"].lower()):
                results.append(book_to_google_book(book))
                break

    # 2. Поиск по локальным книгам через search_local_books (уже улучшенную)
    if len(results) < 10:
        local_found = search_local_books(criteria)
        for book in local_found:
            if book not in results:
                results.append(book)
                if len(results) >= 20:
                    break

    # 3. Если всё ещё мало (меньше 5) — идём в OpenLibrary
    if len(results) < 5:
        try:
            ol_books = await search_openlibrary(criteria, limit=15)
            for book in ol_books:
                if book not in results:
                    results.append(book)
        except Exception as e:
            print(f"OpenLibrary error: {e}")

    # 4. Финальный fallback — книги с высоким рейтингом (но не одни и те же)
    if not results:
        all_books = await data_collector.get_all_books()
        if all_books:
            # Берём не просто популярные, а случайные из топ-100
            top_books = sorted(all_books, key=lambda x: x.get("average_rating", 0), reverse=True)[:100]
            random.shuffle(top_books)
            results = [book_to_google_book(b) for b in top_books[:10]]

    # ================= ВОЗВРАТ (без сортировки по рейтингу) =================
    # Ограничиваем 10 книгами, но не меняем порядок
    final = results[:10]
    print(f"📤 Возвращаем {len(final)} книг")
    for i, b in enumerate(final[:3]):
        print(f"   {i+1}. {b['volumeInfo']['title']}")
    return {
        "results": final,
        "source": "llm+search",
        "is_fallback": len(final) == 0
    }

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
        "system_type": "гибридная",
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
            all_interactions = await data_collector.get_all_interactions()
            all_books = await data_collector.get_all_books()
            await asyncio.to_thread(recommender.build, all_interactions, all_books)
            return {"status": "success", "message": f"Данные пользователя {user_id} очищены"}
        else:
            raise HTTPException(status_code=500, detail="Ошибка очистки данных")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProfileUpdateRequest(BaseModel):
    username: str
    email: str

@app.put("/api/user/{user_id}/profile")
async def update_profile(user_id: int, request: ProfileUpdateRequest):
    try:
        print(f"🔵 ПОЛУЧЕН ЗАПРОС на обновление user_id={user_id}")
        user = await data_collector.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        existing_user = await data_collector.get_user_by_username(request.username)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(status_code=400, detail="Имя пользователя уже занято")
        if request.email:
            existing_email = await data_collector.get_user_by_email(request.email)
            if existing_email and existing_email.id != user_id:
                raise HTTPException(status_code=400, detail="Email уже используется")
        async with data_collector.async_session() as session:
            db_user = await session.get(UserDB, user_id)
            if db_user:
                db_user.username = request.username
                db_user.email = request.email
                await session.commit()
                await session.refresh(db_user)
                return {
                    "id": db_user.id,
                    "username": db_user.username,
                    "email": db_user.email,
                    "created_at": db_user.created_at,
                    "last_login": db_user.last_login
                }
            else:
                raise HTTPException(status_code=404, detail="User not found in session")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Критическая ошибка: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# Путь к папке с текстами (создайте её в корне проекта)
TEXTS_DIR = "book_texts"

@app.get("/api/book/{book_id}/content", response_class=PlainTextResponse)
async def get_book_content(book_id: str):
    print(f"🔍 Ищу текст для book_id = '{book_id}'")
    # Защита от подстановки путей
    safe_id = book_id.replace("/", "").replace("\\", "").replace("..", "")
    file_path = os.path.join(TEXTS_DIR, f"{safe_id}.txt")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Текст книги не найден")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Запуск ГИБРИДНОЙ рекомендательной системы на порту {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
