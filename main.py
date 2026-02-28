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
from gigachat import GigaChat

from data_models import *
from data_collector import DataCollector
from neural_model import HybridRecommender

load_dotenv()

# ==================== ЗАГРУЗКА ЛОКАЛЬНЫХ КНИГ ====================
LOCAL_BOOKS = []
try:
    with open('local_books.json', 'r', encoding='utf-8') as f:
        LOCAL_BOOKS = json.load(f)
    print(f"✅ Загружено {len(LOCAL_BOOKS)} локальных книг")
except Exception as e:
    print(f"❌ Ошибка загрузки локальных книг: {e}")

# ==================== ФУНКЦИИ ПОИСКА ====================
def search_local_books(criteria: dict) -> list:
    """Поиск по локальному JSON"""
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
        if book.get("averageRating", 0) >= min_rating:
            score += 0.5
        if score > 0:
            results.append((book, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [book_to_google_book(book) for book, _ in results[:max_results]]

def book_to_google_book(book: dict) -> dict:
    """Конвертирует локальную книгу в формат GoogleBook"""
    cover = book.get("cover", "")
    return {
        "id": f"local_{book['title']}_{book['author']}".replace(" ", "_"),
        "volumeInfo": {
            "title": book["title"],
            "authors": [book["author"]],
            "description": book.get("description", ""),
            "categories": [book.get("genre", "Unknown")],
            "averageRating": book.get("averageRating", 0),
            "ratingsCount": book.get("ratingsCount", 0),
            "imageLinks": {"thumbnail": cover} if cover else None,
            "language": book.get("language", "ru")
        }
    }

async def search_openlibrary(criteria: dict, limit: int = 5) -> list:
    """Поиск в OpenLibrary на основе критериев"""
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
    """Конвертирует OpenLibrary документ в GoogleBook формат"""
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
    description="API для гибридных рекомендаций (нейросеть + контентная фильтрация + коллаборативная) + Чат с AI",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_collector = DataCollector(data_dir="user_data")
hybrid_recommender = HybridRecommender()

try:
    if hasattr(hybrid_recommender, 'load_all_models'):
        hybrid_recommender.load_all_models()
        print("✅ Модели загружены")
    if hasattr(hybrid_recommender, 'update_collaborative_matrix'):
        hybrid_recommender.update_collaborative_matrix(data_collector)
        print("✅ Коллаборативная матрица обновлена")
except Exception as e:
    print(f"⚠️  Ошибка при инициализации: {e}")

# ==================== ЭНДПОИНТЫ ====================
@app.get("/")
async def root():
    collaborative_ready = False
    if hasattr(hybrid_recommender, 'collaborative_filter'):
        collaborative_ready = hasattr(hybrid_recommender.collaborative_filter, 'user_book_matrix') \
                              and hybrid_recommender.collaborative_filter.user_book_matrix is not None
    return {
        "message": "Гибридная рекомендательная система книг с AI-чатом",
        "version": "2.1.0",
        "status": "работает",
        "system_type": "гибридная (нейросеть + контент + коллаборативная) + AI чат",
        "collaborative_ready": collaborative_ready,
        "data_stats": data_collector.get_all_data_stats()
    }

@app.post("/api/add_interaction")
async def add_interaction(interaction: UserInteraction):
    try:
        data_collector.add_interaction(
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            rating=interaction.rating,
            status=interaction.status,
            book_data=None
        )
        collaborative_updated = False
        if hasattr(hybrid_recommender, 'update_collaborative_matrix'):
            hybrid_recommender.update_collaborative_matrix(data_collector)
            collaborative_updated = True
        if interaction.rating > 0:
            X, y = data_collector.prepare_training_data(interaction.user_id)
            if X is not None and y is not None and hasattr(hybrid_recommender, 'train_for_user'):
                loss = hybrid_recommender.train_for_user(
                    user_id=interaction.user_id,
                    X=X, y=y,
                    epochs=30
                )
                return {
                    "status": "success",
                    "message": "Взаимодействие добавлено, гибридная модель обновлена",
                    "training_loss": loss,
                    "collaborative_updated": collaborative_updated
                }
        return {
            "status": "success",
            "message": "Взаимодействие добавлено",
            "collaborative_updated": collaborative_updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add_interaction_with_book")
async def add_interaction_with_book(interaction: UserInteraction, book_data: BookData):
    try:
        data_collector.add_interaction(
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            rating=interaction.rating,
            status=interaction.status,
            book_data=book_data.model_dump()
        )
        collaborative_updated = False
        if hasattr(hybrid_recommender, 'update_collaborative_matrix'):
            hybrid_recommender.update_collaborative_matrix(data_collector)
            collaborative_updated = True
        if interaction.rating > 0:
            X, y = data_collector.prepare_training_data(interaction.user_id)
            if X is not None and y is not None and hasattr(hybrid_recommender, 'train_for_user'):
                loss = hybrid_recommender.train_for_user(
                    user_id=interaction.user_id,
                    X=X, y=y,
                    epochs=30
                )
                return {
                    "status": "success",
                    "message": "Взаимодействие и данные книги добавлены",
                    "training_loss": loss,
                    "collaborative_updated": collaborative_updated
                }
        return {
            "status": "success",
            "message": "Взаимодействие и данные книги добавлены",
            "collaborative_updated": collaborative_updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
async def get_hybrid_recommendations(request: RecommendationRequest):
    try:
        user_id = request.user_id
        user_interactions = data_collector.get_user_interactions(user_id)
        candidate_books = [book.model_dump() for book in request.candidate_books]
        print(f"👤 Пользователь {user_id}: {len(user_interactions)} взаимодействий")
        print(f"📚 Кандидатов: {len(candidate_books)} книг")
        recommendations, confidences = hybrid_recommender.get_hybrid_recommendations(
            user_id=user_id,
            candidate_books=candidate_books,
            user_interactions=user_interactions,
            data_collector=data_collector,
            top_k=request.limit
        )
        recommended_books = [BookData(**book) for book in recommendations]
        return RecommendationResponse(
            recommendations=recommended_books,
            confidence_scores=confidences,
            training_data_size=len(user_interactions),
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

    # Получаем Authorization key из переменных окружения
    # Получаем Authorization key из переменных окружения
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    print(f"1. Auth key loaded: {'Yes' if auth_key else 'No'}")

    if not auth_key:
        print("2. GigaChat auth key not set, using fallback search")
        criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
    else:
        try:
            print("3. Attempting to get token from GigaChat...")
            async with httpx.AsyncClient(verify=False) as client:
                # Шаг 1: Получение токена
                token_response = await client.post(
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    headers={
                        "Authorization": f"Basic {auth_key}",
                        "RqUID": str(uuid.uuid4()),
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    data={"scope": "GIGACHAT_API_PERS"}
                )

                print(f"4. Token response status: {token_response.status_code}")

                if token_response.status_code != 200:
                    print(f"5. Token error body: {token_response.text}")
                    raise Exception(f"Failed to get token: {token_response.status_code}")

                token_data = token_response.json()
                access_token = token_data.get("access_token")
                print(f"6. Token obtained: {'Yes' if access_token else 'No'}")

                if not access_token:
                    raise Exception("No access_token in response")

                # Шаг 2: Запрос к GigaChat
                print("7. Sending request to GigaChat API...")
                chat_response = await client.post(
                    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "GigaChat",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Ты помощник, который преобразует запрос пользователя о книгах в структурированный JSON для поиска. Ответ должен быть только JSON без лишнего текста."
                            },
                            {
                                "role": "user",
                                "content": f"Запрос: {query}\nФормат ответа: {{\"genres\": [\"жанр1\", \"жанр2\"], \"keywords\": [\"слово1\", \"слово2\"], \"min_rating\": 0.0, \"max_results\": 10}}"
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 512
                    }
                )

                print(f"8. Chat response status: {chat_response.status_code}")

                if chat_response.status_code != 200:
                    print(f"9. Chat error body: {chat_response.text}")
                    raise Exception(f"Chat request failed: {chat_response.status_code}")

                chat_data = chat_response.json()
                print(f"10. Chat response data keys: {chat_data.keys()}")

                if "choices" in chat_data and len(chat_data["choices"]) > 0:
                    criteria_text = chat_data["choices"][0]["message"]["content"]
                    print(f"11. GigaChat response text: {criteria_text}")

                    # Извлекаем JSON
                    start = criteria_text.find('{')
                    end = criteria_text.rfind('}') + 1
                    if start != -1 and end > start:
                        criteria = json.loads(criteria_text[start:end])
                        print(f"12. Parsed criteria: {criteria}")
                    else:
                        print("13. No JSON found in response")
                        criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
                else:
                    print("14. No choices in response")
                    criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}

        except Exception as e:
            print(f"15. GigaChat error: {e}")
            import traceback
            traceback.print_exc()
            criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}

    # Поиск книг (такой же, как был)
    local_results = search_local_books(criteria)
    needed = max(0, criteria.get("max_results", 10) - len(local_results))
    open_library_results = await search_openlibrary(criteria, limit=needed) if needed > 0 else []
    combined = local_results + open_library_results
    return {"results": combined[:criteria.get("max_results", 10)]}

@app.get("/api/user/{user_id}/stats")
async def get_user_stats(user_id: int):
    stats = data_collector.get_user_stats(user_id)
    if not stats:
        return {
            "user_id": user_id,
            "hybrid_system": True,
            "interactions": 0,
            "ratings": 0,
            "recommendation_quality": "базовая (нет истории)"
        }
    interactions = data_collector.get_user_interactions(user_id)
    ratings_count = len([i for i in interactions if i['rating'] > 0])
    quality = "базовая"
    if ratings_count >= 5:
        quality = "высокая"
    elif ratings_count >= 2:
        quality = "средняя"
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
    collaborative_ready = False
    collaborative_users = 0
    collaborative_books = 0
    if hasattr(hybrid_recommender, 'collaborative_filter'):
        collaborative_ready = hasattr(hybrid_recommender.collaborative_filter, 'user_book_matrix') \
                              and hybrid_recommender.collaborative_filter.user_book_matrix is not None
        if collaborative_ready:
            collaborative_users = len(hybrid_recommender.collaborative_filter.user_ids)
            collaborative_books = len(hybrid_recommender.collaborative_filter.book_ids)
    return {
        "system_type": "гибридная (нейросеть + контент + коллаборативная) + AI чат",
        "data_collector": data_collector.get_all_data_stats(),
        "neural_models_trained": len(hybrid_recommender.models) if hasattr(hybrid_recommender, 'models') else 0,
        "total_users": data_collector.stats['unique_users'],
        "collaborative_filtering_ready": collaborative_ready,
        "collaborative_users": collaborative_users,
        "collaborative_books": collaborative_books
    }

@app.get("/health")
async def health_check():
    collaborative_ready = False
    if hasattr(hybrid_recommender, 'collaborative_filter'):
        collaborative_ready = hasattr(hybrid_recommender.collaborative_filter, 'user_book_matrix') \
                              and hybrid_recommender.collaborative_filter.user_book_matrix is not None
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_type": "гибридная + AI чат",
        "neural_models": len(hybrid_recommender.models) if hasattr(hybrid_recommender, 'models') else 0,
        "collaborative_ready": collaborative_ready,
        "total_interactions": data_collector.stats['total_interactions']
    }

@app.post("/api/clear_user_data/{user_id}")
async def clear_user_data(user_id: int):
    try:
        success = data_collector.clear_user_data(user_id)
        if success:
            if hasattr(hybrid_recommender, 'models'):
                hybrid_recommender.models.pop(user_id, None)
            model_path = f"models/user_{user_id}.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
            return {
                "status": "success",
                "message": f"Данные пользователя {user_id} очищены"
            }
        else:
            raise HTTPException(status_code=500, detail="Ошибка очистки данных")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Запуск ГИБРИДНОЙ рекомендательной системы версия 2.1 с AI-чатом...")
    print("📊 Система: Гибридная (нейросеть + контентная + коллаборативная фильтрация) + AI чат")
    print("📊 Статистика данных:", data_collector.get_all_data_stats())
    if hasattr(hybrid_recommender, 'models'):
        print(f"🤖 Загружено нейросетевых моделей: {len(hybrid_recommender.models)}")
    else:
        print("🤖 Нейросетевые модели: не доступны")
    if hasattr(hybrid_recommender, 'collaborative_filter'):
        if hasattr(hybrid_recommender.collaborative_filter, 'user_book_matrix') \
                and hybrid_recommender.collaborative_filter.user_book_matrix is not None:
            print("🤝 Коллаборативная фильтрация: ГОТОВА")
            print(f"   Пользователей в матрице: {len(hybrid_recommender.collaborative_filter.user_ids)}")
            print(f"   Книг в матрице: {len(hybrid_recommender.collaborative_filter.book_ids)}")
        else:
            print("🤝 Коллаборативная фильтрация: НЕ ГОТОВА (нужно больше данных)")
    else:
        print("🤝 Коллаборативная фильтрация: не доступна")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")