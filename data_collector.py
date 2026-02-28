import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import os
import shutil
import glob


class DataCollector:
    """
    НОВАЯ версия: собирает данные отдельно для каждого пользователя
    """

    def __init__(self, data_dir: str = "user_data"):
        self.data_dir = data_dir
        self.user_interactions: Dict[int, List[dict]] = defaultdict(list)
        self.books_metadata: Dict[int, dict] = {}
        self.user_profiles: Dict[int, dict] = {}

        self.stats = {
            'total_interactions': 0,
            'unique_users': 0,
            'unique_books': 0,
            'avg_rating': 0.0
        }

        # Создаем директорию для данных пользователей
        os.makedirs(data_dir, exist_ok=True)

        # Загружаем существующие данные
        self._load_all_data()

    def _load_all_data(self):
        """Загрузка всех сохраненных данных"""
        try:
            # Загружаем метаданные книг
            metadata_file = os.path.join(self.data_dir, "books_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.books_metadata = {int(k): v for k, v in loaded.items()}
                print(f"📚 Загружено метаданных книг: {len(self.books_metadata)}")
        except Exception as e:
            print(f"❌ Ошибка загрузки метаданных книг: {e}")

        try:
            # Загружаем профили пользователей
            profiles_file = os.path.join(self.data_dir, "user_profiles.json")
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.user_profiles = {int(k): v for k, v in loaded.items()}
                print(f"👤 Загружено профилей пользователей: {len(self.user_profiles)}")
        except Exception as e:
            print(f"❌ Ошибка загрузки профилей пользователей: {e}")

        try:
            # Загружаем данные каждого пользователя
            user_files = glob.glob(os.path.join(self.data_dir, "user_*.json"))
            for file in user_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'user_id' in data:
                        user_id = data['user_id']
                        self.user_interactions[user_id] = data.get('interactions', [])
                        print(
                            f"👤 Загружены взаимодействия для пользователя {user_id}: {len(self.user_interactions[user_id])} книг")
                    else:
                        # Возможно, файл повреждён – пропускаем
                        print(f"⚠️ Файл {file} не содержит user_id, пропускаем")
                except Exception as e:
                    print(f"❌ Ошибка загрузки файла {file}: {e}")

            self._update_stats()
            print(
                f"💾 Данные загружены: {len(self.user_interactions)} пользователей, всего взаимодействий: {self.stats['total_interactions']}")
        except Exception as e:
            print(f"❌ Общая ошибка загрузки данных: {e}")

    def save_all_data(self):
        """Сохранение всех данных"""
        try:
            # Сохраняем метаданные книг
            metadata_file = os.path.join(self.data_dir, "books_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.books_metadata, f, indent=2, default=str)
            print(f"💾 Сохранены метаданные книг: {len(self.books_metadata)} записей")

            # Сохраняем профили пользователей
            profiles_file = os.path.join(self.data_dir, "user_profiles.json")
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, indent=2, default=str)
            print(f"💾 Сохранены профили пользователей: {len(self.user_profiles)}")

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

        except Exception as e:
            print(f"❌ Ошибка сохранения данных: {e}")

    def add_interaction(self, user_id: int, book_id: int, rating: float,
                        status: str, book_data: dict = None):
        """
        Добавление нового взаимодействия пользователя с книгой
        """
        print(f"👤 Добавление взаимодействия для пользователя {user_id} с рейтингом {rating}")

        # Создаем запись о взаимодействии
        interaction = {
            'user_id': user_id,
            'book_id': book_id,
            'rating': rating,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'book_data': book_data
        }

        # Добавляем во взаимодействия пользователя
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []

        # Удаляем старую запись если есть
        self.user_interactions[user_id] = [
            i for i in self.user_interactions[user_id]
            if not (i['book_id'] == book_id and i['user_id'] == user_id)
        ]

        # Добавляем новую
        self.user_interactions[user_id].append(interaction)

        # Сохраняем метаданные книги
        if book_data:
            self.books_metadata[book_id] = {
                'title': book_data.get('title', ''),
                'author': book_data.get('author', ''),
                'genre': book_data.get('genre', ''),
                'tags': book_data.get('tags', []),
                'average_rating': book_data.get('average_rating', 0.0)
            }

        # Обновляем профиль пользователя
        self._update_user_profile(user_id)

        # Сохраняем все данные
        self.save_all_data()

        print(f"✅ Взаимодействие добавлено: пользователь {user_id}, книга {book_id}")
        print(f"📊 Всего у пользователя {user_id}: {len(self.user_interactions[user_id])} книг")

    def _update_user_profile(self, user_id: int):
        """Обновление профиля пользователя"""
        user_interactions = self.user_interactions.get(user_id, [])

        if not user_interactions:
            self.user_profiles[user_id] = {
                'avg_rating': 0.0,
                'total_books': 0,
                'preferred_genres': [],
                'last_active': datetime.now().isoformat()
            }
            return

        # Считаем средний рейтинг
        ratings = [i['rating'] for i in user_interactions if i['rating'] > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        # Определяем любимые жанры
        genre_counts = defaultdict(int)
        for interaction in user_interactions:
            book_id = interaction['book_id']
            if book_id in self.books_metadata:
                genre = self.books_metadata[book_id].get('genre', '')
                if genre:
                    genre_counts[genre] += 1

        # Топ-3 жанра
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        preferred_genres = [genre for genre, count in top_genres]

        # Сохраняем профиль
        self.user_profiles[user_id] = {
            'avg_rating': avg_rating,
            'total_books': len(user_interactions),
            'preferred_genres': preferred_genres,
            'last_active': datetime.now().isoformat()
        }

    def _update_stats(self):
        """Обновление статистики"""
        total_interactions = 0
        all_book_ids = set()

        for user_id, interactions in self.user_interactions.items():
            total_interactions += len(interactions)
            all_book_ids.update(i['book_id'] for i in interactions)

        self.stats['total_interactions'] = total_interactions
        self.stats['unique_users'] = len(self.user_interactions)
        self.stats['unique_books'] = len(all_book_ids)

        # Средний рейтинг по всем пользователям
        all_ratings = []
        for interactions in self.user_interactions.values():
            all_ratings.extend([i['rating'] for i in interactions if i['rating'] > 0])

        self.stats['avg_rating'] = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    def get_user_interactions(self, user_id: int) -> List[dict]:
        """Получение всех взаимодействий пользователя"""
        return self.user_interactions.get(user_id, [])

    def get_all_interactions(self):
        """Получение всех взаимодействий всех пользователей"""
        all_interactions = []
        for user_id, interactions in self.user_interactions.items():
            all_interactions.extend(interactions)
        return all_interactions

    def get_user_stats(self, user_id: int) -> dict:
        """Получение статистики пользователя"""
        return self.user_profiles.get(user_id, {})

    def get_all_data_stats(self) -> dict:
        """Получение общей статистики"""
        self._update_stats()
        return self.stats

    def prepare_training_data(self, user_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Подготовка данных для обучения для конкретного пользователя
        """
        user_interactions = self.user_interactions.get(user_id, [])

        if len(user_interactions) < 3:
            return None, None

        X = []
        y = []

        for interaction in user_interactions:
            book_id = interaction['book_id']
            rating = interaction['rating']

            if book_id in self.books_metadata:
                features = self._extract_book_features(book_id)
                if features is not None and rating > 0:
                    X.append(features)
                    y.append(rating)

        if len(X) < 3:
            return None, None

        return np.array(X), np.array(y)

    def _extract_book_features(self, book_id: int) -> Optional[np.ndarray]:
        """Извлечение признаков книги"""
        if book_id not in self.books_metadata:
            return None

        book = self.books_metadata[book_id]

        # Список всех возможных жанров
        all_genres = [
            'fiction', 'science fiction', 'fantasy', 'mystery', 'romance',
            'thriller', 'horror', 'historical fiction', 'biography', 'science',
            'philosophy', 'poetry', 'drama', 'comedy', 'adventure',
            'children', 'young adult', 'classics', 'russian literature'
        ]

        # One-hot кодирование жанра
        genre_vector = np.zeros(len(all_genres))
        book_genre = book.get('genre', '').lower()
        for i, genre in enumerate(all_genres):
            if genre in book_genre or book_genre in genre:
                genre_vector[i] = 1.0

        # Нормализованный средний рейтинг
        avg_rating = book.get('average_rating', 0.0) / 5.0

        # Количество тегов (нормализованное)
        tags = book.get('tags', [])
        tags_count = min(len(tags) / 10.0, 1.0)

        # Объединяем все признаки
        features = np.concatenate([
            genre_vector,
            [avg_rating, tags_count]
        ])

        return features

    def clear_user_data(self, user_id: int):
        """Очистка всех данных пользователя"""
        try:
            # Удаляем взаимодействия пользователя
            if user_id in self.user_interactions:
                del self.user_interactions[user_id]

            # Удаляем профиль пользователя
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]

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