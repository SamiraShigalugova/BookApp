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
if "postgresql" in DATABASE_URL and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    print(f"✅ Преобразованный DATABASE_URL для asyncpg: {DATABASE_URL}")

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
        LOCAL_BOOKS.append(book)
    print(f"✅ Загружено {len(LOCAL_BOOKS)} локальных книг")
except Exception as e:
    print(f"❌ Ошибка загрузки локальных книг: {e}")

# ==================== ФУНКЦИИ ПОИСКА ====================
def search_local_books(criteria: dict) -> list:
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
            {"role": "system", "content": """
Ты – помощник, который преобразует запрос пользователя о книгах в структурированный JSON для поиска. 
Твоя задача – извлечь из текста запроса ТОЛЬКО жанры, ключевые слова и минимальный рейтинг. 
Ты НЕ должен предлагать конкретные книги, НЕ должен писать текст, только JSON.

=== СПИСОК ДОСТУПНЫХ ЖАНРОВ ===
"романтика", "любовный роман", "фэнтези", "эпическое фэнтези", "темное фэнтези", "городское фэнтези", 
"научная фантастика", "киберпанк", "стимпанк", "космическая опера", "постапокалипсис", "антиутопия", 
"детектив", "полицейский детектив", "исторический детектив", "триллер", "психологический триллер", 
"юридический триллер", "ужасы", "мистика", "готика", "приключения", "классика", "русская классика", 
"зарубежная классика", "исторический роман", "военный роман", "биография", "автобиография", 
"мемуары", "поэзия", "драма", "трагедия", "комедия", "юмор", "сатира", "философия", 
"психология", "саморазвитие", "мотивация", "карьера", "бизнес", "финансы", "инвестиции", 
"маркетинг", "менеджмент", "научно-популярная литература", "популярная наука", "физика", 
"биология", "химия", "астрономия", "история", "археология", "искусство", "архитектура", 
"музыка", "кино", "театр", "программирование", "алгоритмы", "искусственный интеллект", 
"математика", "алгебра", "геометрия", "статистика", "кулинария", "путешествия", "туризм", 
"спорт", "фитнес", "здоровье", "медицина", "эзотерика", "астрология", "таро", 
"религия", "духовность", "буддизм", "христианство", "ислам", "этика", "политология", 
"социология", "экономика", "экология", "феминизм", "психотерапия", "семейная психология", 
"детская литература", "сказки", "young adult", "подростковая литература", "графический роман", 
"комиксы", "манга", "сборник рассказов", "новелла", "эссе"

=== ПРАВИЛА СОПОСТАВЛЕНИЯ ЗАПРОСОВ С ЖАНРАМИ ===

Если пользователь пишет что-то из левой колонки, используй жанры из правой:

=== ЛЮБОВЬ И ОТНОШЕНИЯ ===
"любовь", "романтика", "роман", "нежные чувства", "влюбленность", "свидания" → ["романтика"]
"сложные отношения", "измена", "развод", "расставание" → ["романтика", "драма"]
"семья", "дети", "родители", "брак" → ["семейная психология", "драма"]

=== ФЭНТЕЗИ И МАГИЯ ===
"магия", "волшебство", "колдовство", "чародейство" → ["фэнтези"]
"эльфы", "гномы", "орки", "драконы", "маги" → ["эпическое фэнтези"]
"вампиры", "оборотни", "нежить", "демоны" → ["темное фэнтези", "ужасы"]
"современная магия", "городское фэнтези", "магия в городе" → ["городское фэнтези"]

=== НАУЧНАЯ ФАНТАСТИКА ===
"фантастика", "будущее", "технологии", "роботы", "андроиды" → ["научная фантастика"]
"киберпанк", "хакеры", "киберпространство" → ["киберпанк"]
"паровые машины", "дирижабли", "викторианская эпоха" → ["стимпанк"]
"космос", "звездолеты", "инопланетяне", "галактика" → ["космическая опера"]
"апокалипсис", "конец света", "зомби", "выживание" → ["постапокалипсис"]
"тоталитаризм", "большой брат", "контроль", "свобода" → ["антиутопия"]

=== ДЕТЕКТИВЫ И ТРИЛЛЕРЫ ===
"детектив", "загадка", "расследование", "преступление", "убийство", "преступник" → ["детектив"]
"полиция", "следователь", "прокурор" → ["полицейский детектив"]
"историческое преступление", "средневековье" → ["исторический детектив"]
"триллер", "напряжение", "погоня", "опасность" → ["триллер"]
"психология", "маньяк", "психиатр", "безумие" → ["психологический триллер"]
"суд", "адвокат", "присяжные", "правосудие" → ["юридический триллер"]

=== УЖАСЫ И МИСТИКА ===
"страшное", "жуткое", "пугающее", "хоррор", "ужасы" → ["ужасы"]
"мистика", "таинственное", "сверхъестественное" → ["мистика"]
"готика", "замок", "призраки", "старый особняк" → ["готика"]

=== ПРИКЛЮЧЕНИЯ ===
"приключения", "путешествия", "походы", "экшн", "опасные приключения" → ["приключения"]
"пираты", "клад", "морские путешествия" → ["приключения", "исторический роман"]
"джунгли", "сафари", "экспедиция" → ["приключения", "путешествия"]

=== КЛАССИКА ===
"классика", "великие произведения", "мировая литература" → ["классика"]
"русская литература", "русские писатели", "достоевский", "толстой" → ["русская классика"]
"шекспир", "диккенс", "остин", "гюго" → ["зарубежная классика"]

=== ИСТОРИЧЕСКИЕ ===
"история", "исторический", "средневековье", "античность" → ["исторический роман"]
"война", "битва", "солдаты", "военные действия" → ["военный роман", "исторический роман"]

=== БИОГРАФИИ И МЕМУАРЫ ===
"биография", "жизнь", "о человеке", "история жизни" → ["биография"]
"мемуары", "воспоминания", "дневники" → ["мемуары", "автобиография"]
"автобиография", "о себе" → ["автобиография"]

=== ПОЭЗИЯ И ДРАМА ===
"стихи", "поэзия", "рифмы", "сонеты" → ["поэзия"]
"драма", "трагедия", "тяжелая судьба" → ["драма", "трагедия"]
"смешное", "веселое", "юмор", "комедия" → ["комедия", "юмор"]
"сатира", "ирония", "высмеивание" → ["сатира", "юмор"]

=== ФИЛОСОФИЯ И ПСИХОЛОГИЯ ===
"философия", "смысл жизни", "бытие", "экзистенциализм" → ["философия"]
"психология", "человек", "личность", "характер" → ["психология"]
"саморазвитие", "самореализация", "стать лучше", "развитие личности" → ["саморазвитие", "психология"]
"мотивация", "успех", "достижение целей" → ["мотивация", "саморазвитие"]
"карьера", "работа", "профессия", "карьерный рост" → ["карьера", "бизнес"]
"бизнес", "предпринимательство", "стартап" → ["бизнес", "менеджмент"]
"финансы", "деньги", "инвестиции", "капитал" → ["финансы", "инвестиции"]
"маркетинг", "продажи", "реклама" → ["маркетинг", "бизнес"]
"управление", "лидерство", "команда" → ["менеджмент"]

=== НАУЧНО-ПОПУЛЯРНАЯ ЛИТЕРАТУРА ===
"наука", "научная", "популярная наука" → ["научно-популярная литература"]
"физика", "квантовая механика", "теория относительности" → ["физика", "научно-популярная литература"]
"биология", "эволюция", "генетика", "днк" → ["биология", "научно-популярная литература"]
"химия", "элементы", "реакции" → ["химия", "научно-популярная литература"]
"астрономия", "космос", "звезды", "галактики", "вселенная" → ["астрономия", "научно-популярная литература"]
"история", "историческая", "древний мир" → ["история", "научно-популярная литература"]
"археология", "раскопки", "цивилизации" → ["археология", "история"]

=== ИСКУССТВО И КУЛЬТУРА ===
"искусство", "живопись", "скульптура" → ["искусство"]
"архитектура", "здания", "города" → ["архитектура"]
"музыка", "композиторы", "инструменты" → ["музыка"]
"кино", "фильмы", "режиссеры" → ["кино"]
"театр", "спектакли", "актеры" → ["театр"]

=== ПРОГРАММИРОВАНИЕ И ТЕХНОЛОГИИ ===
"программирование", "код", "разработка", "айти", "it" → ["программирование"]
"алгоритмы", "структуры данных", "сортировка" → ["алгоритмы", "программирование"]
"искусственный интеллект", "нейросети", "ai", "машинное обучение" → ["искусственный интеллект", "программирование"]

=== МАТЕМАТИКА ===
"математика", "алгебра", "геометрия", "тригонометрия" → ["математика"]
"статистика", "вероятность", "анализ данных" → ["статистика", "математика"]

=== ХОББИ И ОБРАЗ ЖИЗНИ ===
"кулинария", "рецепты", "готовка", "еда" → ["кулинария"]
"путешествия", "туризм", "отпуск", "страны" → ["путешествия", "туризм"]
"спорт", "фитнес", "тренировки", "здоровый образ жизни" → ["спорт", "фитнес"]
"здоровье", "медицина", "лечение", "болезни" → ["здоровье", "медицина"]

=== ЭЗОТЕРИКА И ДУХОВНОСТЬ ===
"эзотерика", "тайные знания", "магия" → ["эзотерика", "мистика"]
"астрология", "гороскопы", "зодиак" → ["астрология"]
"таро", "карты", "гадание" → ["таро", "эзотерика"]
"религия", "бог", "вера", "церковь" → ["религия"]
"духовность", "медитация", "осознанность", "дзен" → ["духовность", "буддизм"]
"буддизм", "даосизм" → ["буддизм", "духовность"]
"христианство", "православие", "католицизм" → ["христианство", "религия"]

=== ОБЩЕСТВО И ПОЛИТИКА ===
"политика", "власть", "государство", "правительство" → ["политология"]
"социология", "общество", "социальные группы" → ["социология"]
"экономика", "рынок", "капитализм", "деньги" → ["экономика"]
"экология", "природа", "окружающая среда", "климат" → ["экология"]
"феминизм", "гендер", "равенство" → ["феминизм", "социология"]

=== ПСИХОТЕРАПИЯ И САМОПОМОЩЬ ===
"психотерапия", "терапия", "психолог" → ["психотерапия", "психология"]
"семейная психология", "отношения", "конфликты" → ["семейная психология"]
"депрессия", "тревога", "стресс" → ["психология", "психотерапия"]
"уверенность", "самооценка", "комплексы" → ["саморазвитие", "психология"]

=== ДЕТСКАЯ ЛИТЕРАТУРА ===
"детские", "для детей", "сказки", "малышам" → ["детская литература", "сказки"]
"подростки", "для подростков", "тинейджеры" → ["young adult", "подростковая литература"]
"комиксы", "графический роман", "манга" → ["комиксы", "графический роман", "манга"]

=== ФОРМАТЫ ПРОИЗВЕДЕНИЙ ===
"рассказы", "сборник рассказов" → ["сборник рассказов"]
"новелла", "повесть" → ["новелла"]
"эссе", "размышления" → ["эссе"]

=== ПРАВИЛА ДЛЯ КЛЮЧЕВЫХ СЛОВ ===
1. Извлекай ВСЕ значимые слова из запроса:
   - Настроение: "легкое", "веселое", "грустное", "задумчивое", "мотивирующее"
   - Темы: "магия", "драконы", "космос", "война", "любовь", "дружба", "предательство"
   - Имена авторов: Толстой, Достоевский, Пушкин, Лермонтов, Булгаков, Ремарк, Оруэлл
   - Части названий: если пользователь вспомнил часть названия книги
   - Персонажи: "Гарри Поттер", "Шерлок Холмс", "Ведьмак"

=== ПРАВИЛА ДЛЯ РЕЙТИНГА ===
- "рейтинг выше 4", "хорошие", "популярные", "с высоким рейтингом" → 4.0
- "рейтинг выше 4.5", "отличные", "шедевры", "лучшие" → 4.5
- "средние", "неплохие", "нормальные" → 3.5
- "выше среднего" → 3.8
- "топ", "бестселлеры" → 4.2
- иначе → 0

=== ВЫХОДНОЙ ФОРМАТ ===
Ответ должен быть ТОЛЬКО JSON в формате:
{"genres": ["жанр1", "жанр2"], "keywords": ["слово1", "слово2"], "min_rating": число, "max_results": 10}

=== ПРИМЕРЫ ===
Запрос: "хочу что-то страшное" → {"genres": ["ужасы", "триллер"], "keywords": ["страшное"], "min_rating": 0, "max_results": 10}
Запрос: "посоветуй романтическую книгу про любовь" → {"genres": ["романтика"], "keywords": ["любовь"], "min_rating": 0, "max_results": 10}
Запрос: "научная фантастика про космос и будущее, рейтинг выше 4" → {"genres": ["научная фантастика"], "keywords": ["космос", "будущее"], "min_rating": 4.0, "max_results": 10}
Запрос: "классика, которую должен прочитать каждый" → {"genres": ["классика"], "keywords": [], "min_rating": 0, "max_results": 10}
Запрос: "Толстой, война и мир" → {"genres": [], "keywords": ["Толстой", "война и мир"], "min_rating": 0, "max_results": 10}
Запрос: "книги по программированию для начинающих" → {"genres": ["программирование"], "keywords": ["для начинающих"], "min_rating": 0, "max_results": 10}
Запрос: "саморазвитие, мотивация, успех" → {"genres": ["саморазвитие", "мотивация"], "keywords": ["успех"], "min_rating": 0, "max_results": 10}
Запрос: "легкая книга для отдыха" → {"genres": ["юмор", "комедия"], "keywords": ["легкая", "отдых"], "min_rating": 0, "max_results": 10}
Запрос: "про любовь и войну" → {"genres": ["романтика", "военный роман"], "keywords": ["любовь", "война"], "min_rating": 0, "max_results": 10}

Теперь обработай запрос пользователя и выдай ТОЛЬКО JSON, без пояснений и текста.
"""},
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
        raise Exception("GigaChat request failed")
    return response.json()

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
    if not all_books and LOCAL_BOOKS:
        await data_collector.add_books(LOCAL_BOOKS)
        print("✅ Локальные книги загружены в базу данных")
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

    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key:
        criteria = {"genres": [], "keywords": [], "min_rating": 0, "max_results": 10}
    else:
        try:
            async with httpx.AsyncClient(verify=False) as client:
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
                    raise Exception("Failed to get token")
                token_data = token_response.json()
                access_token = token_data.get("access_token")

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

if __name__ == "__main__":
    print("🚀 Запуск ГИБРИДНОЙ рекомендательной системы...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
