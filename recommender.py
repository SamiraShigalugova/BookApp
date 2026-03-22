# recommender.py - коллаборативная + контентная фильтрация (без scipy/sklearn)
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any


class BookRecommender:
    """
    Гибридная рекомендательная система, объединяющая:
    - популярность (fallback)
    - контентную близость (жанры/теги)
    - коллаборативную фильтрацию (item-based на чистом NumPy)
    """

    def __init__(self):
        # Данные для коллаборативной фильтрации
        self.user_ids: List[int] = []
        self.book_ids: List[str] = []
        self.user_to_idx: Dict[int, int] = {}
        self.book_to_idx: Dict[str, int] = {}

        self.user_book_matrix: Optional[np.ndarray] = None
        self.book_similarity: Optional[np.ndarray] = None

        # Популярность книг
        self.book_popularity: Dict[str, float] = {}

        # Метаданные книг
        self.books_metadata: Dict[str, Dict] = {}

    def build(self, interactions: List[Dict], books: List[Dict]):
        """Полностью перестраивает систему."""
        self.books_metadata = {b["id"]: b for b in books}
        self.book_ids = sorted(set(i["book_id"] for i in interactions) | set(self.books_metadata.keys()))
        self.user_ids = sorted(set(i["user_id"] for i in interactions if i.get("rating", 0) > 0))
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.book_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}

        self._build_user_book_matrix(interactions)
        self._build_book_similarity()  # коллаборативная матрица
        self._build_popularity(interactions)

        print(f"✅ Система построена: {len(self.user_ids)} пользователей, {len(self.book_ids)} книг")
        if self.book_similarity is not None:
            print(f"   Матрица сходства книг: {self.book_similarity.shape}")
        else:
            print("   Матрица сходства книг: не построена (недостаточно данных)")

    def _build_user_book_matrix(self, interactions: List[Dict]):
        """Строит матрицу user × book с оценками (плотная)."""
        n_users = len(self.user_ids)
        n_books = len(self.book_ids)
        matrix = np.zeros((n_users, n_books), dtype=np.float32)

        for inter in interactions:
            rating = inter.get("rating", 0)
            if rating > 0:
                user_id = inter["user_id"]
                book_id = inter["book_id"]
                if user_id in self.user_to_idx and book_id in self.book_to_idx:
                    u_idx = self.user_to_idx[user_id]
                    b_idx = self.book_to_idx[book_id]
                    matrix[u_idx, b_idx] = rating

        self.user_book_matrix = matrix

    def _build_book_similarity(self):
        """Строит матрицу косинусного сходства между книгами (на чистом NumPy)."""
        if self.user_book_matrix is None:
            self.book_similarity = None
            return

        n_books = len(self.book_ids)
        if n_books < 2:
            self.book_similarity = None
            return

        # Транспонируем: строки = книги, столбцы = пользователи
        book_vectors = self.user_book_matrix.T  # shape: (n_books, n_users)

        # Проверяем, есть ли ненулевые оценки
        if np.sum(book_vectors) == 0:
            self.book_similarity = None
            return

        # Нормализуем векторы
        norms = np.linalg.norm(book_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = book_vectors / norms

        # Косинусное сходство через матричное умножение
        similarity = np.dot(normalized, normalized.T)
        np.fill_diagonal(similarity, 0)  # убираем сходство книги с собой

        self.book_similarity = similarity

    def _build_popularity(self, interactions: List[Dict]):
        """Вычисляет популярность книг."""
        rating_counts = defaultdict(int)
        rating_sums = defaultdict(float)

        for inter in interactions:
            rating = inter.get("rating", 0)
            if rating > 0:
                book_id = inter["book_id"]
                rating_counts[book_id] += 1
                rating_sums[book_id] += rating

        self.book_popularity = {}
        for book_id, cnt in rating_counts.items():
            avg = rating_sums[book_id] / cnt
            self.book_popularity[book_id] = avg * np.log1p(cnt)

    def recommend(
        self,
        user_id: int,
        candidate_books: List[Dict],
        user_interactions: List[Dict],
        top_k: int = 10
    ) -> Tuple[List[Dict], List[float]]:
        """Возвращает рекомендации."""
        positive_ratings = [i for i in user_interactions if i.get("rating", 0) > 0]
        rated_book_ids = {i["book_id"] for i in positive_ratings}
        n_ratings = len(positive_ratings)

        # Нет оценок -> популярность
        if n_ratings == 0:
            return self._popular_books(candidate_books, top_k, exclude=rated_book_ids)

        # Строим контентный профиль
        profile = self._build_content_profile(positive_ratings)

        # Оцениваем каждую книгу
        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in rated_book_ids:
                continue

            content_score = self._content_score(book, profile)

            if n_ratings < 3:
                score = content_score * 0.7
            else:
                collab_score = self._collaborative_score(user_id, book_id) / 5.0
                score = content_score * 0.6 + collab_score * 0.4

            scored.append((book, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    def _popular_books(self, candidate_books: List[Dict], top_k: int, exclude: set):
        """Популярные книги."""
        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in exclude:
                continue
            score = self.book_popularity.get(book_id, book.get("average_rating", 3.0))
            scored.append((book, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    def _build_content_profile(self, positive_ratings: List[Dict]) -> Dict:
        """Профиль пользователя по жанрам."""
        profile = defaultdict(float)

        for inter in positive_ratings:
            rating = inter["rating"]
            book_id = inter["book_id"]
            meta = self.books_metadata.get(book_id, {})
            genre = meta.get("genre", "").lower()

            if genre:
                profile[genre] += rating

        total = sum(profile.values())
        if total > 0:
            for k in profile:
                profile[k] /= total

        return profile

    def _content_score(self, book: Dict, profile: Dict) -> float:
        """Контентная близость."""
        genre = book.get("genre", "").lower()
        return profile.get(genre, 0.0)

    def _collaborative_score(self, user_id: int, book_id: str) -> float:
        """
        Коллаборативная оценка через item-based подход.
        """
        if self.book_similarity is None:
            return 0.0

        if user_id not in self.user_to_idx or book_id not in self.book_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        b_idx = self.book_to_idx[book_id]

        if self.user_book_matrix is None or u_idx >= len(self.user_book_matrix):
            return 0.0

        # Оценки пользователя
        user_ratings = self.user_book_matrix[u_idx]  # shape: (n_books,)

        # Индексы книг, которые пользователь оценил
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return 0.0

        # Берём сходство целевой книги с оценёнными
        sims = self.book_similarity[b_idx, rated_indices]

        # Взвешенная сумма
        weighted_sum = np.sum(sims * user_ratings[rated_indices])
        total_weight = np.sum(sims[sims > 0])

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight
