import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import os
import glob


class DataCollector:
    def __init__(self, data_dir: str = "user_data"):
        self.data_dir = data_dir
        self.user_interactions: Dict[int, List[dict]] = defaultdict(list)
        self.books_metadata: Dict[str, dict] = {} 

        self.stats = {
            'total_interactions': 0,
            'unique_users': 0,
            'unique_books': 0,
            'avg_rating': 0.0
        }

        os.makedirs(data_dir, exist_ok=True)
        self._load_all_data()

    def _load_all_data(self):

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
                    # Преобразуем book_id в строку
                    for inter in interactions:
                        inter['book_id'] = str(inter['book_id'])
                    self.user_interactions[user_id] = interactions
                    print(f"👤 Загружены взаимодействия для пользователя {user_id}: {len(interactions)} книг")
                else:
                    print(f"⚠️ Файл {file} не содержит user_id, пропускаем")
            except Exception as e:
                print(f"❌ Ошибка загрузки файла {file}: {e}")

        self._update_stats()
        print(f"💾 Данные загружены: {len(self.user_interactions)} пользователей, всего взаимодействий: {self.stats['total_interactions']}")

    def save_all_data(self):
        """Сохранение всех данных."""
        # Сохраняем метаданные книг
        metadata_file = os.path.join(self.data_dir, "books_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.books_metadata, f, indent=2, default=str)
        print(f"💾 Сохранены метаданные книг: {len(self.books_metadata)} записей")

        # Сохраняем данные каждого пользователя
        for user_id, interactions in self.user_interactions.items():
            user_file = os.path.join(self.data_dir, f"user_{user_id}.json")
            data = {
                'user_id': user_id,
                'interactions': interactions,
                'last_updated': datetime.now().isoformat()
            }
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Сохранены взаимодействия пользователя {user_id}: {len(interactions)} записей")

        self._update_stats()
        print(f"💾 Все данные сохранены: {len(self.user_interactions)} пользователей")

    def add_books(self, books: List[dict]):
        """
        Добавляет список книг в метаданные.
        Ожидается, что каждая книга содержит поля:
        id (str), title, author, genre, tags, description, average_rating.
        """
        for book in books:
            book_id = book["id"]
            self.books_metadata[book_id] = {
                'title': book.get('title', ''),
                'author': book.get('author', ''),
                'genre': book.get('genre', ''),
                'tags': book.get('tags', []),
                'description': book.get('description', ''),
                'average_rating': book.get('average_rating', 0.0)
            }
        print(f"📚 Добавлено книг в метаданные: {len(books)}")

    def add_interaction(self, user_id: int, book_id: str, rating: float,
                        status: str, book_data: dict = None):
        """
        Добавление нового взаимодействия пользователя с книгой.
        Если передан book_data, метаданные книги обновляются.
        """
        print(f"👤 Добавление взаимодействия для пользователя {user_id} с рейтингом {rating}")

        # Создаём запись о взаимодействии
        interaction = {
            'user_id': user_id,
            'book_id': str(book_id),  # всегда строка
            'rating': rating,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        }

        # Если переданы данные книги, сохраняем их
        if book_data:
            self.books_metadata[book_id] = {
                'title': book_data.get('title', ''),
                'author': book_data.get('author', ''),
                'genre': book_data.get('genre', ''),
                'tags': book_data.get('tags', []),
                'description': book_data.get('description', ''),
                'average_rating': book_data.get('average_rating', 0.0)
            }
            interaction['book_data'] = book_data  # для истории

        # Добавляем во взаимодействия пользователя
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []

        # Удаляем старую запись, если была (чтобы не дублировать)
        self.user_interactions[user_id] = [
            i for i in self.user_interactions[user_id]
            if not (i['book_id'] == book_id and i.get('user_id') == user_id)
        ]

        self.user_interactions[user_id].append(interaction)

        # Сохраняем все данные
        self.save_all_data()

        print(f"✅ Взаимодействие добавлено: пользователь {user_id}, книга {book_id}")
        print(f"📊 Всего у пользователя {user_id}: {len(self.user_interactions[user_id])} книг")

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
        return self.user_interactions.get(user_id, [])

    def get_all_interactions(self) -> List[dict]:
        """Получение всех взаимодействий всех пользователей."""
        all_interactions = []
        for interactions in self.user_interactions.values():
            all_interactions.extend(interactions)
        return all_interactions

    def get_all_books(self) -> List[dict]:
        """
        Возвращает список всех книг из метаданных в формате, пригодном для Recommender.
        """
        books = []
        for book_id, meta in self.books_metadata.items():
            books.append({
                'id': book_id,
                'title': meta.get('title', ''),
                'author': meta.get('author', ''),
                'genre': meta.get('genre', ''),
                'tags': meta.get('tags', []),
                'description': meta.get('description', ''),
                'average_rating': meta.get('average_rating', 0.0)
            })
        return books

    def get_user_stats(self, user_id: int) -> dict:
        """Получение статистики пользователя."""
        interactions = self.user_interactions.get(user_id, [])
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
            if user_id in self.user_interactions:
                del self.user_interactions[user_id]

            # Удаляем файл пользователя
            user_file = os.path.join(self.data_dir, f"user_{user_id}.json")
            if os.path.exists(user_file):
                os.remove(user_file)

            self.save_all_data()
            print(f"🧹 Данные пользователя {user_id} очищены")
            return True
        except Exception as e:
            print(f"❌ Ошибка очистки данных: {e}")
            return False
