import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import os
from collections import defaultdict
import glob
import re
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import math

def is_nan(x):
    return isinstance(x, float) and math.isnan(x)

class BookRecommenderNN(nn.Module):
    """
    Нейронная сеть для рекомендации книг
    """

    def __init__(self, input_size: int):
        super(BookRecommenderNN, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов сети"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x) * 5
        return x


class CollaborativeFiltering:
    """
    Коллаборативная фильтрация на основе пользователь-книга матрицы
    """

    def __init__(self):
        self.user_book_matrix = None
        self.user_similarity = None
        self.book_similarity = None
        self.user_ids = []
        self.book_ids = []
        self.book_popularity = {}  # Для бестселлеров

    def build_matrix(self, interactions, user_ids, book_ids):
        """
        Строит матрицу пользователь-книга
        """
        self.user_ids = list(user_ids)
        self.book_ids = list(book_ids)

        # Создаем маппинги
        user_to_idx = {user_id: i for i, user_id in enumerate(self.user_ids)}
        book_to_idx = {book_id: i for i, book_id in enumerate(self.book_ids)}

        # Строим разреженную матрицу
        rows, cols, data = [], [], []

        # Считаем популярность книг
        book_rating_counts = defaultdict(int)
        book_rating_sums = defaultdict(float)

        for interaction in interactions:
            user_idx = user_to_idx.get(interaction['user_id'])
            book_idx = book_to_idx.get(interaction['book_id'])
            rating = interaction.get('rating', 0)

            if rating > 0:
                book_rating_counts[interaction['book_id']] += 1
                book_rating_sums[interaction['book_id']] += rating

            if user_idx is not None and book_idx is not None and rating > 0:
                rows.append(user_idx)
                cols.append(book_idx)
                data.append(rating)

        # Вычисляем популярность (средний рейтинг * логарифм количества оценок)
        for book_id in book_rating_counts:
            avg_rating = book_rating_sums[book_id] / book_rating_counts[book_id]
            self.book_popularity[book_id] = avg_rating * np.log1p(book_rating_counts[book_id])

        if rows:
            self.user_book_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(len(self.user_ids), len(self.book_ids))
            )

            # Вычисляем схожести
            if len(self.user_ids) > 1:
                self.user_similarity = cosine_similarity(self.user_book_matrix)
            if len(self.book_ids) > 1:
                self.book_similarity = cosine_similarity(self.user_book_matrix.T)

    def get_user_based_recommendations(self, user_id, candidate_books, top_k=10):
        """
        Коллаборативная фильтрация на основе пользователей
        """
        if user_id not in self.user_ids or self.user_similarity is None:
            return []

        user_idx = self.user_ids.index(user_id)

        # Находим похожих пользователей
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:6]  # топ-5 похожих

        recommendations = {}
        for similar_user_idx in similar_users:
            books_rated = self.user_book_matrix[similar_user_idx].nonzero()[1]

            for book_idx in books_rated:
                if book_idx < len(self.book_ids):
                    book_id = self.book_ids[book_idx]

                    if any(b['id'] == book_id for b in candidate_books):
                        similarity = self.user_similarity[user_idx, similar_user_idx]
                        rating = self.user_book_matrix[similar_user_idx, book_idx]

                        if book_id not in recommendations:
                            recommendations[book_id] = 0
                        recommendations[book_id] += rating * similarity

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [book_id for book_id, score in sorted_recs]

    def get_popular_books(self, candidate_books, top_k=10):
        candidate_popularity = []
        for book in candidate_books:
            if self.book_popularity and book['id'] in self.book_popularity:
                candidate_popularity.append((book, self.book_popularity[book['id']]))
            else:
                # Используем рейтинг книги из кандидата, если нет данных о популярности
                candidate_popularity.append((book, book.get('average_rating', 3.0)))
        # Сортируем по убыванию популярности/рейтинга
        sorted_books = sorted(candidate_popularity, key=lambda x: x[1], reverse=True)[:top_k]
        return [book for book, _ in sorted_books]


class HybridRecommender:
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.models: Dict[int, BookRecommenderNN] = {}
        self.collaborative_filter = CollaborativeFiltering()
        self.input_size = 21

        os.makedirs(model_path, exist_ok=True)

    def load_all_models(self):
        """Загрузка всех сохраненных моделей"""
        model_files = glob.glob(f"{self.model_path}user_*.pth")

        print(f"📂 Поиск моделей в {self.model_path}")
        print(f"📂 Найдено файлов: {len(model_files)}")

        loaded_count = 0
        for file_path in model_files:
            match = re.search(r'user_(\d+)\.pth', file_path)
            if match:
                user_id = int(match.group(1))
                if self.load_model(user_id):
                    loaded_count += 1

        print(f"✅ Загружено моделей: {loaded_count}")

    def load_model(self, user_id: int) -> bool:
        """Загрузка модели пользователя"""
        model_path = f"{self.model_path}user_{user_id}.pth"

        try:
            if not os.path.exists(model_path):
                return False

            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            input_size = checkpoint.get('input_size', self.input_size)
            model = BookRecommenderNN(input_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.models[user_id] = model
            model.eval()
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def save_model(self, user_id: int):
        """Сохранение модели пользователя"""
        if user_id not in self.models:
            return

        model_path = f"{self.model_path}user_{user_id}.pth"
        torch.save({
            'model_state_dict': self.models[user_id].state_dict(),
            'input_size': self.input_size,
            'timestamp': datetime.now().isoformat()
        }, model_path)

    def train_for_user(self, user_id: int, X: np.ndarray, y: np.ndarray,
                       epochs: int = 30, learning_rate: float = 0.001) -> float:
        """
        Обучение модели для конкретного пользователя
        """
        print(f"🎯 Обучение модели для пользователя {user_id}")

        if len(X) == 0 or len(y) == 0:
            return 0.0

        # Преобразуем данные в тензоры
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        # Создаем или загружаем модель
        if user_id not in self.models:
            self.models[user_id] = BookRecommenderNN(self.input_size)

        model = self.models[user_id]
        model.train()

        # Оптимизатор и функция потерь
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Обучение
        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        # Сохраняем модель
        self.save_model(user_id)

        final_loss = loss_history[-1] if loss_history else 0.0
        return final_loss


    def update_collaborative_matrix(self, data_collector):
        """
        Обновляет матрицу коллаборативной фильтрации
        """
        all_interactions = data_collector.get_all_interactions()
        all_user_ids = list(set(i['user_id'] for i in all_interactions))
        all_book_ids = list(set(i['book_id'] for i in all_interactions))

        if len(all_user_ids) >= 2 and len(all_book_ids) >= 2:
            self.collaborative_filter.build_matrix(all_interactions, all_user_ids, all_book_ids)
            print(f"✅ Коллаборативная матрица обновлена: {len(all_user_ids)} пользователей, {len(all_book_ids)} книг")

    def get_hybrid_recommendations(self, user_id: int, candidate_books: List[dict],
                                   user_interactions: List[dict], data_collector=None,
                                   top_k: int = 10) -> Tuple[List[dict], List[float]]:
        """
        Гибридные рекомендации с защитой от ошибок
        """
        try:
            # Шаг 1: Определяем стадию пользователя
            ratings_count = len([i for i in user_interactions if i.get('rating', 0) > 0])

            if ratings_count < 3:
                # ХОЛОДНЫЙ СТАРТ: новые пользователи или мало данных
                print(f"❄️  Холодный старт для пользователя {user_id} (оценок: {ratings_count})")

                try:
                    # Если совсем нет данных - предлагаем популярные книги
                    if ratings_count == 0:
                        popular_books = self.collaborative_filter.get_popular_books(candidate_books, top_k)
                        confidences = [0.9] * len(popular_books)
                        return popular_books, confidences

                    # Если есть 1-2 оценки - смесь контентной фильтрации и популярных книг
                    content_scores = self._get_content_scores(user_id, candidate_books, user_interactions,
                                                              data_collector)

                    # Получаем популярные книги
                    popular_books = self.collaborative_filter.get_popular_books(candidate_books, top_k)
                    popular_book_ids = {book['id'] for book in popular_books}

                    # Комбинируем: 60% контентная, 40% популярные
                    hybrid_scores = []
                    for i, book in enumerate(candidate_books):
                        content_score = content_scores[i] if i < len(content_scores) else 0.5
                        is_popular = book['id'] in popular_book_ids
                        hybrid_score = (content_score * 0.6) + (0.4 if is_popular else 0.0)
                        hybrid_scores.append((book, hybrid_score))

                    sorted_books = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_k]
                    recommendations = [book for book, _ in sorted_books]
                    confidences = [score for _, score in sorted_books]

                    return recommendations, confidences

                except Exception as e:
                    print(f"❌ Ошибка в холодном старте: {e}")
                    # Fallback на популярные книги
                    popular_books = self.collaborative_filter.get_popular_books(candidate_books, top_k)
                    return popular_books, [0.9] * len(popular_books)

            else:
                # ГИБРИДНАЯ СИСТЕМА: достаточно данных для всех компонентов
                print(f"🤖 Гибридная система для пользователя {user_id} (оценок: {ratings_count})")

                try:
                    # 1. Контентные рекомендации
                    try:
                        content_scores = self._get_content_scores(user_id, candidate_books, user_interactions,
                                                                  data_collector)
                        # Заменяем NaN на 0.5
                        content_scores = [0.5 if np.isnan(x) else x for x in content_scores]
                    except Exception as e:
                        print(f"⚠️ Ошибка в content scores: {e}")
                        content_scores = [book.get('average_rating', 3.0) / 5.0 for book in candidate_books]

                    # 2. Нейросетевые оценки
                    try:
                        neural_scores = self._get_neural_scores(user_id, candidate_books)
                        if neural_scores is not None:
                            # Заменяем NaN на content_scores для соответствующих позиций
                            for i in range(len(neural_scores)):
                                if np.isnan(neural_scores[i]):
                                    neural_scores[i] = content_scores[i]
                        else:
                            neural_scores = content_scores.copy()
                    except Exception as e:
                        print(f"⚠️ Ошибка в neural scores: {e}")
                        neural_scores = content_scores.copy()

                    # 3. Коллаборативные рекомендации
                    collaborative_recommendations = set()
                    try:
                        if hasattr(self.collaborative_filter,
                                   'user_ids') and user_id in self.collaborative_filter.user_ids:
                            collaborative_book_ids = self.collaborative_filter.get_user_based_recommendations(
                                user_id, candidate_books, top_k=top_k
                            )
                            collaborative_recommendations = set(collaborative_book_ids)
                    except Exception as e:
                        print(f"⚠️ Ошибка в collaborative: {e}")

                    # 4. Гибридный скор с динамическими весами
                    hybrid_scores = []
                    for i, book in enumerate(candidate_books):
                        content_score = content_scores[i] if i < len(content_scores) else 0.5
                        neural_score = neural_scores[i] if i < len(neural_scores) else content_score

                        # Динамические веса в зависимости от количества оценок
                        base_content_weight = 0.4
                        base_neural_weight = 0.4
                        collaborative_weight = 0.2

                        if ratings_count > 10:
                            base_neural_weight = 0.5
                            base_content_weight = 0.3

                        is_collaborative = 1.0 if book['id'] in collaborative_recommendations else 0.0

                        # Итоговый скор
                        hybrid_score = (
                                content_score * base_content_weight +
                                neural_score * base_neural_weight +
                                is_collaborative * collaborative_weight
                        )

                        hybrid_scores.append((book, hybrid_score))

                    # Сортировка и возврат
                    sorted_books = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_k]
                    recommendations = [book for book, _ in sorted_books]
                    confidences = [score for _, score in sorted_books]

                    return recommendations, confidences

                except Exception as e:
                    print(f"❌ Критическая ошибка в гибридном режиме: {e}")
                    # Ultimate fallback - популярные книги
                    popular_books = self.collaborative_filter.get_popular_books(candidate_books, top_k)
                    return popular_books, [0.9] * len(popular_books)

        except Exception as e:
            print(f"💥 Непредвиденная ошибка в get_hybrid_recommendations: {e}")
            # Самый крайний случай - вернуть первые top_k книг
            return candidate_books[:top_k], [0.5] * min(top_k, len(candidate_books))

    def _get_content_scores(self, user_id: int, candidate_books: List[dict],
                            user_interactions: List[dict], data_collector) -> List[float]:
        """
        Вычисление контентных оценок на основе истории пользователя
        """
        if not user_interactions:
            return [book.get('average_rating', 3.0) / 5.0 for book in candidate_books]

        # Строим профиль пользователя
        user_profile = self._build_user_profile(user_interactions, data_collector)

        # Вычисляем схожесть каждой книги с профилем
        scores = []
        for book in candidate_books:
            similarity = self._calculate_content_similarity(book, user_profile)
            scores.append(similarity)

        return scores

    def _build_user_profile(self, user_interactions: List[dict], data_collector) -> Dict:
        """
        Построение контентного профиля пользователя
        """
        profile = {
            'genres': defaultdict(float),
            'tags': defaultdict(float),
            'avg_rating': 0.0
        }

        ratings = []
        for interaction in user_interactions:
            book_id = interaction['book_id']
            rating = interaction.get('rating', 0)

            if data_collector and book_id in data_collector.books_metadata:
                book_meta = data_collector.books_metadata[book_id]

                genre = book_meta.get('genre', '')
                if genre and rating > 0:
                    profile['genres'][genre.lower()] += rating

                tags = book_meta.get('tags', [])
                for tag in tags:
                    profile['tags'][tag.lower()] += 1

            if rating > 0:
                ratings.append(rating)

        profile['avg_rating'] = sum(ratings) / len(ratings) if ratings else 3.0

        # Нормализуем
        for key in ['genres', 'tags']:
            total = sum(profile[key].values())
            if total > 0:
                for item in profile[key]:
                    profile[key][item] /= total

        return profile

    def _calculate_content_similarity(self, book: Dict, user_profile: Dict) -> float:
        """
        Вычисление схожести книги с профилем пользователя
        """
        similarity = 0.0

        # Жанры (вес 0.6)
        book_genre = book.get('genre', '').lower()
        if book_genre in user_profile['genres']:
            similarity += user_profile['genres'][book_genre] * 0.6

        # Теги (вес 0.3)
        book_tags = [tag.lower() for tag in book.get('tags', [])]
        for tag in book_tags:
            if tag in user_profile['tags']:
                similarity += user_profile['tags'][tag] * 0.3

        # Рейтинг книги (вес 0.1)
        book_rating = book.get('average_rating', 3.0)
        if book_rating >= user_profile['avg_rating']:
            similarity += 0.1

        return min(similarity, 1.0)

    def _get_neural_scores(self, user_id: int, candidate_books: List[dict]) -> Optional[np.ndarray]:
        if user_id not in self.models:
            return None

        X_candidates = []
        valid_indices = []
        for i, book in enumerate(candidate_books):
            features = self._extract_book_features(book)
            if features is not None:
                X_candidates.append(features)
                valid_indices.append(i)

        if not X_candidates:
            return None

        try:
            model = self.models[user_id]
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(np.array(X_candidates))
                predictions = model(X_tensor)
                scores = predictions.numpy().flatten() / 5.0

            # Создаём массив размером с candidate_books, заполняя NaN для пропущенных
            full_scores = np.full(len(candidate_books), np.nan)
            for idx, score in zip(valid_indices, scores):
                full_scores[idx] = score
            return full_scores
        except Exception as e:
            print(f"Ошибка при предсказании нейросети: {e}")
            return None

    def _extract_book_features(self, book: dict) -> Optional[np.ndarray]:
        try:
            all_genres = [
                'fiction', 'science fiction', 'fantasy', 'mystery', 'romance',
                'thriller', 'horror', 'historical fiction', 'biography', 'science',
                'philosophy', 'poetry', 'drama', 'comedy', 'adventure',
                'children', 'young adult', 'classics', 'russian literature',
                'criticism', 'german fiction'
            ]
            genre_vector = np.zeros(len(all_genres))
            book_genre = book.get('genre', '').lower()
            for i, genre in enumerate(all_genres):
                if genre in book_genre or book_genre in genre:
                    genre_vector[i] = 1.0

            avg_rating = book.get('average_rating', 3.0) / 5.0
            tags = book.get('tags', [])
            tags_count = min(len(tags) / 10.0, 1.0)

            return np.concatenate([genre_vector, [avg_rating, tags_count]])
        except Exception as e:
            print(f"Ошибка извлечения признаков для книги {book.get('id')}: {e}")
            return None