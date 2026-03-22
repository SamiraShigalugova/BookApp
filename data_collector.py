# data_collector.py
import json
import os
import glob
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Any
import psycopg2
from psycopg2.extras import Json


class DataCollector:
    """
    Собирает данные отдельно для каждого пользователя и хранит метаданные книг.
    Поддерживает PostgreSQL (основное хранилище) и файловое хранилище (fallback).
    """

    def __init__(self, data_dir: str = "user_data"):
        self.data_dir = data_dir
        self.user_interactions: Dict[int, List[dict]] = defaultdict(list)
        self.books_metadata: Dict[str, dict] = {}
        self.conn = None
        self.use_postgres = False

        self.stats = {
            'total_interactions': 0,
            'unique_users': 0,
            'unique_books': 0,
            'avg_rating': 0.0
        }

        # Пытаемся подключиться к PostgreSQL
        self._init_database()
        
        # Если PostgreSQL не работает, используем файловое хранилище
        if not self.use_postgres:
            os.makedirs(data_dir, exist_ok=True)
            self._load_all_data()

    def _init_database(self):
        """Инициализация подключения к PostgreSQL"""
        database_url = os.environ.get('DATABASE_URL')
        
        if not database_url:
            print("⚠️ DATABASE_URL не найдена, использую файловое хранилище")
            return
        
        try:
            self.conn = psycopg2.connect(database_url)
            self.use_postgres = True
            self._create_tables()
            self._load_from_postgres()
            print("✅ PostgreSQL подключена успешно")
        except Exception as e:
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            print("⚠️ Переключаюсь на файловое хранилище")
            self.use_postgres = False
            self.conn = None

    def _create_tables(self):
        """Создание таблиц в PostgreSQL"""
        try:
            with self.conn.cursor() as cur:
                # Таблица пользователей
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL UNIQUE,
                        username VARCHAR(255),
                        created_at TIMESTAMP DEFAULT NOW(),
                        last_updated TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Таблица взаимодействий
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        book_id TEXT NOT NULL,
                        rating FLOAT DEFAULT 0,
                        status TEXT,
                        book_data JSONB,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        UNIQUE(user_id, book_id)
                    )
                """)
                
                # Таблица метаданных книг
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS books_metadata (
                        book_id TEXT PRIMARY KEY,
                        title TEXT,
                        author TEXT,
                        genre TEXT,
                        tags TEXT[],
                        description TEXT,
                        average_rating FLOAT,
                        cover_url TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Индексы для ускорения запросов
                cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_book_id ON interactions(book_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_books_metadata_genre ON books_metadata(genre)")
                
                self.conn.commit()
                print("✅ Таблицы PostgreSQL созданы")
        except Exception as e:
            print(f"❌ Ошибка создания таблиц: {e}")
            self.conn.rollback()

    def _load_from_postgres(self):
        """Загрузка данных из PostgreSQL"""
        try:
            with self.conn.cursor() as cur:
                # Загружаем метаданные книг
                cur.execute("SELECT book_id, title, author, genre, tags, description, average_rating, cover_url FROM books_metadata")
                rows = cur.fetchall()
                for row in rows:
                    self.books_metadata[row[0]] = {
                        'title': row[1],
                        'author': row[2],
                        'genre': row[3],
                        'tags': row[4] if row[4] else [],
                        'description': row[5] if row[5] else '',
                        'average_rating': row[6] if row[6] else 0.0,
                        'cover_url': row[7] if row[7] else ''
                    }
                
                # Загружаем взаимодействия
                cur.execute("SELECT user_id, book_id, rating, status, book_data, timestamp FROM interactions ORDER BY timestamp")
                rows = cur.fetchall()
                for row in rows:
                    user_id = row[0]
                    interaction = {
                        'user_id': user_id,
                        'book_id': row[1],
                        'rating': row[2],
                        'status': row[3],
                        'timestamp': row[5].isoformat() if row[5] else datetime.now().isoformat()
                    }
                    if row[4]:
                        interaction['book_data'] = row[4]
                    self.user_interactions[user_id].append(interaction)
                
                print(f"📚 Загружено из PostgreSQL: {len(self.books_metadata)} книг, {len(self.user_interactions)} пользователей")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки из PostgreSQL: {e}")
            self.use_postgres = False

    def _load_all_data(self):
        """Загрузка всех сохранённых данных (файловое хранилище)"""
        # Загружаем метаданные книг
        metadata_file = os.path.join(self.data_dir, "books_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.books_metadata = {k: v for k, v in loaded.items()}
                print(f"📚 Загружено метаданных книг: {len(self.books_metadata)}")
            except Exception as e:
                print(f"❌ Ошибка загрузки метаданных книг: {e}")

        # Загружаем данные каждого пользователя
        user_files = glob.glob(os.path.join(self.data_dir, "user_*.json"))
        for file in user_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'user_id' in data:
                    user_id = data['user_id']
                    interactions = data.get('interactions', [])
                    for inter in interactions:
                        inter['book_id'] = str(inter['book_id'])
                    self.user_interactions[user_id] = interactions
                    print(f"👤 Загружены взаимодействия для пользователя {user_id}: {len(interactions)} книг")
            except Exception as e:
                print(f"❌ Ошибка загрузки файла {file}: {e}")

        self._update_stats()
        print(f"💾 Данные загружены: {len(self.user_interactions)} пользователей, всего взаимодействий: {self.stats['total_interactions']}")

    def save_all_data(self):
        """Сохранение всех данных"""
        if self.use_postgres:
            # PostgreSQL автоматически сохраняет данные, ничего делать не нужно
            print(f"💾 Данные уже в PostgreSQL: {len(self.user_interactions)} пользователей")
            return
        
        # Файловое хранилище
        metadata_file = os.path.join(self.data_dir, "books_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.books_metadata, f, indent=2, default=str, ensure_ascii=False)
        print(f"💾 Сохранены метаданные книг: {len(self.books_metadata)} записей")

        for user_id, interactions in self.user_interactions.items():
            user_file = os.path.join(self.data_dir, f"user_{user_id}.json")
            data = {
                'user_id': user_id,
                'interactions': interactions,
                'last_updated': datetime.now().isoformat()
            }
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            print(f"💾 Сохранены взаимодействия пользователя {user_id}: {len(interactions)} записей")

        self._update_stats()
        print(f"💾 Все данные сохранены: {len(self.user_interactions)} пользователей")

    def add_books(self, books: List[dict]):
        """Добавляет список книг в метаданные."""
        for book in books:
            book_id = book["id"]
            self.books_metadata[book_id] = {
                'title': book.get('title', ''),
                'author': book.get('author', ''),
                'genre': book.get('genre', ''),
                'tags': book.get('tags', []),
                'description': book.get('description', ''),
                'average_rating': book.get('average_rating', 0.0),
                'cover_url': book.get('cover_url', '')
            }
            
            if self.use_postgres and self.conn:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO books_metadata (book_id, title, author, genre, tags, description, average_rating, cover_url)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (book_id) DO UPDATE SET
                                title = EXCLUDED.title,
                                author = EXCLUDED.author,
                                genre = EXCLUDED.genre,
                                tags = EXCLUDED.tags,
                                description = EXCLUDED.description,
                                average_rating = EXCLUDED.average_rating,
                                cover_url = EXCLUDED.cover_url,
                                updated_at = NOW()
                        """, (
                            book_id,
                            book.get('title', ''),
                            book.get('author', ''),
                            book.get('genre', ''),
                            book.get('tags', []),
                            book.get('description', ''),
                            book.get('average_rating', 0.0),
                            book.get('cover_url', '')
                        ))
                    self.conn.commit()
                except Exception as e:
                    print(f"❌ Ошибка сохранения книги {book_id}: {e}")
                    self.conn.rollback()
        
        print(f"📚 Добавлено книг в метаданные: {len(books)}")
        if not self.use_postgres:
            self.save_all_data()

    def add_interaction(self, user_id: int, book_id: str, rating: float,
                        status: str, book_data: dict = None):
        """Добавление нового взаимодействия пользователя с книгой."""
        print(f"👤 Добавление взаимодействия для пользователя {user_id} с рейтингом {rating}")

        interaction = {
            'user_id': user_id,
            'book_id': str(book_id),
            'rating': rating,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        }

        if book_data:
            self.books_metadata[book_id] = {
                'title': book_data.get('title', ''),
                'author': book_data.get('author', ''),
                'genre': book_data.get('genre', ''),
                'tags': book_data.get('tags', []),
                'description': book_data.get('description', ''),
                'average_rating': book_data.get('average_rating', 0.0),
                'cover_url': book_data.get('cover_url', '')
            }
            interaction['book_data'] = book_data

        if self.use_postgres and self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (user_id) VALUES (%s)
                        ON CONFLICT (user_id) DO UPDATE SET last_updated = NOW()
                    """, (user_id,))
                    
                    cur.execute("""
                        INSERT INTO interactions (user_id, book_id, rating, status, book_data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, book_id) DO UPDATE SET
                            rating = EXCLUDED.rating,
                            status = EXCLUDED.status,
                            book_data = EXCLUDED.book_data,
                            timestamp = NOW()
                    """, (user_id, book_id, rating, status, Json(book_data) if book_data else None))
                    
                    if book_data:
                        cur.execute("""
                            INSERT INTO books_metadata (book_id, title, author, genre, tags, description, average_rating, cover_url)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (book_id) DO UPDATE SET
                                title = EXCLUDED.title,
                                author = EXCLUDED.author,
                                genre = EXCLUDED.genre,
                                tags = EXCLUDED.tags,
                                description = EXCLUDED.description,
                                average_rating = EXCLUDED.average_rating,
                                cover_url = EXCLUDED.cover_url,
                                updated_at = NOW()
                        """, (
                            book_id,
                            book_data.get('title', ''),
                            book_data.get('author', ''),
                            book_data.get('genre', ''),
                            book_data.get('tags', []),
                            book_data.get('description', ''),
                            book_data.get('average_rating', 0.0),
                            book_data.get('cover_url', '')
                        ))
                    
                    self.conn.commit()
                    print(f"✅ Взаимодействие сохранено в PostgreSQL")
            except Exception as e:
                print(f"❌ Ошибка сохранения в PostgreSQL: {e}")
                self.conn.rollback()
                self._save_interaction_to_file(user_id, book_id, rating, status, book_data, interaction)
        else:
            self._save_interaction_to_file(user_id, book_id, rating, status, book_data, interaction)

        self._update_stats()

    def _save_interaction_to_file(self, user_id, book_id, rating, status, book_data, interaction):
        """Сохранение взаимодействия в файл (fallback)"""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []

        self.user_interactions[user_id] = [
            i for i in self.user_interactions[user_id]
            if not (i['book_id'] == book_id and i.get('user_id') == user_id)
        ]
        self.user_interactions[user_id].append(interaction)
        self.save_all_data()
        print(f"✅ Взаимодействие добавлено: пользователь {user_id}, книга {book_id}")

    def _update_stats(self):
        """Обновление статистики."""
        total_interactions = 0
        all_book_ids = set()
        all_ratings = []

        for user_id, interactions in self.user_interactions.items():
            total_interactions += len(interactions)
            for i in interactions:
                all_book_ids.add(i['book_id'])
                if i.get('rating', 0) > 0:
                    all_ratings.append(i['rating'])

        self.stats['total_interactions'] = total_interactions
        self.stats['unique_users'] = len(self.user_interactions)
        self.stats['unique_books'] = len(all_book_ids)
        self.stats['avg_rating'] = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    def get_user_interactions(self, user_id: int) -> List[dict]:
        """Получение всех взаимодействий пользователя."""
        if self.use_postgres and self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT book_id, rating, status, book_data, timestamp
                        FROM interactions
                        WHERE user_id = %s
                        ORDER BY timestamp DESC
                    """, (user_id,))
                    rows = cur.fetchall()
                    return [
                        {
                            'book_id': row[0],
                            'rating': row[1],
                            'status': row[2],
                            'book_data': row[3] if row[3] else {},
                            'timestamp': row[4].isoformat() if row[4] else datetime.now().isoformat()
                        }
                        for row in rows
                    ]
            except Exception as e:
                print(f"❌ Ошибка получения взаимодействий: {e}")
        
        return self.user_interactions.get(user_id, [])

    def get_all_interactions(self) -> List[dict]:
        """Получение всех взаимодействий всех пользователей."""
        if self.use_postgres and self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_id, book_id, rating, status, book_data, timestamp
                        FROM interactions
                        ORDER BY timestamp DESC
                    """)
                    rows = cur.fetchall()
                    return [
                        {
                            'user_id': row[0],
                            'book_id': row[1],
                            'rating': row[2],
                            'status': row[3],
                            'book_data': row[4] if row[4] else {},
                            'timestamp': row[5].isoformat() if row[5] else datetime.now().isoformat()
                        }
                        for row in rows
                    ]
            except Exception as e:
                print(f"❌ Ошибка получения всех взаимодействий: {e}")
        
        all_interactions = []
        for interactions in self.user_interactions.values():
            all_interactions.extend(interactions)
        return all_interactions

    def get_all_books(self) -> List[dict]:
        """Возвращает список всех книг из метаданных."""
        if self.use_postgres and self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT book_id, title, author, genre, tags, description, average_rating, cover_url
                        FROM books_metadata
                    """)
                    rows = cur.fetchall()
                    return [
                        {
                            'id': row[0],
                            'title': row[1],
                            'author': row[2],
                            'genre': row[3],
                            'tags': row[4] if row[4] else [],
                            'description': row[5] if row[5] else '',
                            'average_rating': row[6] if row[6] else 0.0,
                            'cover_url': row[7] if row[7] else ''
                        }
                        for row in rows
                    ]
            except Exception as e:
                print(f"❌ Ошибка получения книг: {e}")
        
        books = []
        for book_id, meta in self.books_metadata.items():
            books.append({
                'id': book_id,
                'title': meta.get('title', ''),
                'author': meta.get('author', ''),
                'genre': meta.get('genre', ''),
                'tags': meta.get('tags', []),
                'description': meta.get('description', ''),
                'average_rating': meta.get('average_rating', 0.0),
                'cover_url': meta.get('cover_url', '')
            })
        return books

    def get_user_stats(self, user_id: int) -> dict:
        """Получение статистики пользователя."""
        interactions = self.get_user_interactions(user_id)
        ratings = [i['rating'] for i in interactions if i.get('rating', 0) > 0]
        return {
            'total_books': len(interactions),
            'ratings_count': len(ratings),
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0.0
        }

    def get_all_data_stats(self) -> dict:
        """Получение общей статистики."""
        self._update_stats()
        return self.stats

    def clear_user_data(self, user_id: int) -> bool:
        """Очистка всех данных пользователя."""
        try:
            if self.use_postgres and self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("DELETE FROM interactions WHERE user_id = %s", (user_id,))
                    cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                    self.conn.commit()
                print(f"🧹 Данные пользователя {user_id} очищены из PostgreSQL")
                return True
            
            if user_id in self.user_interactions:
                del self.user_interactions[user_id]

            user_file = os.path.join(self.data_dir, f"user_{user_id}.json")
            if os.path.exists(user_file):
                os.remove(user_file)

            self.save_all_data()
            print(f"🧹 Данные пользователя {user_id} очищены")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка очистки данных: {e}")
            return False
