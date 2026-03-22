import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:
    """
    Гибридная рекомендательная система, объединяющая:
    - популярность (fallback)
    - контентную близость (жанры/теги)
    - коллаборативную фильтрацию (item-based)
    """

    def __init__(self):
        # Данные для коллаборативной фильтрации
        self.user_ids: List[int] = []
        self.book_ids: List[str] = []               # строковые идентификаторы книг
        self.user_to_idx: Dict[int, int] = {}
        self.book_to_idx: Dict[str, int] = {}       # отображение строкового book_id → индекс

        self.user_book_matrix: Optional[csr_matrix] = None
        self.book_similarity: Optional[np.ndarray] = None

        # Популярность книг
        self.book_popularity: Dict[str, float] = {}

        # Метаданные книг (для контентной фильтрации)
        self.books_metadata: Dict[str, Dict] = {}

    # =========================
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # =========================

    def build(self, interactions: List[Dict], books: List[Dict]):
        """
        Полностью перестраивает систему на основе переданных взаимодействий и книг.
        Вызывается при старте или периодически для обновления.
        """
        # Сохраняем метаданные книг для быстрого доступа
        self.books_metadata = {b["id"]: b for b in books}

        # Формируем списки пользователей и книг
        self.user_ids = sorted(set(i["user_id"] for i in interactions if i.get("rating", 0) > 0))
        self.book_ids = sorted(set(i["book_id"] for i in interactions) | set(self.books_metadata.keys()))

        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.book_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}

        # Строим разреженную матрицу оценок
        self._build_user_book_matrix(interactions)

        # Строим матрицу сходства книг (item-item)
        self._build_book_similarity()

        # Строим популярность книг
        self._build_popularity(interactions)

        print(f"✅ Система построена: {len(self.user_ids)} пользователей, {len(self.book_ids)} книг")
        if self.book_similarity is not None:
            print(f"   Матрица сходства книг: {self.book_similarity.shape}")
        else:
            print(f"   Матрица сходства книг: не построена (недостаточно данных)")

    def recommend(
        self,
        user_id: int,
        candidate_books: List[Dict],
        user_interactions: List[Dict],
        top_k: int = 10
    ) -> Tuple[List[Dict], List[float]]:
        """
        Возвращает список рекомендованных книг и соответствующие оценки уверенности.
        :param user_id: идентификатор пользователя
        :param candidate_books: список книг-кандидатов (словари с полем id и др.)
        :param user_interactions: история взаимодействий данного пользователя
        :param top_k: количество рекомендаций
        :return: (список книг, список оценок)
        """
        # Фильтруем только книги с оценками > 0
        positive_ratings = [i for i in user_interactions if i.get("rating", 0) > 0]
        rated_book_ids = {i["book_id"] for i in positive_ratings}
        n_ratings = len(positive_ratings)

        # Если нет оценок — используем только популярность
        if n_ratings == 0:
            return self._popular_books(candidate_books, top_k, exclude=rated_book_ids)

        # Строим контентный профиль пользователя (жанры/теги)
        profile = self._build_content_profile(positive_ratings)

        # Оцениваем каждую книгу-кандидат
        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in rated_book_ids:
                continue  # не предлагаем уже оценённые книги

            # Контентный скор (жанры/теги) [0..1]
            content = self._content_score(book, profile)

            # Базовый скор для холодного старта (мало оценок)
            if n_ratings < 3:
                score = content * 0.7  # больше веса на контент
            else:
                # Коллаборативный скор (item-based) -> нормируем делением на 5
                collab = self._collaborative_score(user_id, book_id) / 5.0
                score = content * 0.6 + collab * 0.4

            scored.append((book, score))

        # Сортируем по убыванию скора и возвращаем top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    # =========================
    # ВНУТРЕННИЕ МЕТОДЫ ДЛЯ ПОСТРОЕНИЯ
    # =========================

    def _build_user_book_matrix(self, interactions: List[Dict]):
        """Строит разреженную матрицу user × book с оценками."""
        n_users = len(self.user_ids)
        n_books = len(self.book_ids)

        matrix = lil_matrix((n_users, n_books), dtype=np.float32)

        for inter in interactions:
            user_id = inter["user_id"]
            book_id = inter["book_id"]
            rating = inter.get("rating", 0)
            if rating <= 0:
                continue
            if user_id in self.user_to_idx and book_id in self.book_to_idx:
                u_idx = self.user_to_idx[user_id]
                b_idx = self.book_to_idx[book_id]
                matrix[u_idx, b_idx] = rating

        self.user_book_matrix = matrix.tocsr()

    def _build_book_similarity(self):
        """Строит матрицу косинусного сходства между книгами (item-item)."""
        if self.user_book_matrix is None:
            self.book_similarity = None
            return

        if self.user_book_matrix.shape[1] < 2:
            self.book_similarity = None
            return

        if self.user_book_matrix.nnz == 0:
            self.book_similarity = None
            return

        book_vectors = self.user_book_matrix.T.tocsr()

        if book_vectors.shape[0] == 0 or book_vectors.shape[1] == 0:
            self.book_similarity = None
            return

        try:
            self.book_similarity = cosine_similarity(book_vectors, dense_output=True)
        except ValueError as e:
            print(f"⚠️ Не удалось вычислить сходство книг: {e}")
            self.book_similarity = None

    def _build_popularity(self, interactions: List[Dict]):
        """Вычисляет популярность книг на основе количества оценок и среднего рейтинга."""
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

        if not self.book_popularity:
            print("⚠️ Нет данных для вычисления популярности книг")

    # =========================
    # ВНУТРЕННИЕ МЕТОДЫ ДЛЯ РЕКОМЕНДАЦИЙ
    # =========================

    def _popular_books(self, candidate_books: List[Dict], top_k: int, exclude: set) -> Tuple[List[Dict], List[float]]:
        """Возвращает топ популярных книг из кандидатов, исключая уже оценённые."""
        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in exclude:
                continue
            if book_id in self.book_popularity:
                score = self.book_popularity[book_id]
            else:
                score = book.get("average_rating", 3.0)  # fallback
            scored.append((book, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    def _build_content_profile(self, positive_ratings: List[Dict]) -> Dict:
        """Строит профиль пользователя на основе жанров и тегов книг, которые он высоко оценил."""
        profile = {
            "genres": defaultdict(float),
            "tags": defaultdict(float)
        }

        for inter in positive_ratings:
            rating = inter["rating"]
            book_id = inter["book_id"]
            meta = self.books_metadata.get(book_id, {})
            if not meta:
                continue

            genre_field = meta.get("genre", "")
            if isinstance(genre_field, str):
                genres = [genre_field.lower()] if genre_field else []
            elif isinstance(genre_field, list):
                genres = [g.lower() for g in genre_field if g]
            else:
                genres = []

            for g in genres:
                profile["genres"][g] += rating

            tags = meta.get("tags", [])
            for tag in tags:
                if isinstance(tag, str):
                    profile["tags"][tag.lower()] += 1

        for key in ["genres", "tags"]:
            total = sum(profile[key].values())
            if total > 0:
                for k in profile[key]:
                    profile[key][k] /= total

        return profile

    def _content_score(self, book: Dict, profile: Dict) -> float:
        """Вычисляет контентную близость книги к профилю (жанры 0.7, теги 0.3)."""
        score = 0.0

        genre_field = book.get("genre", "")
        if isinstance(genre_field, str):
            genres = [genre_field.lower()] if genre_field else []
        elif isinstance(genre_field, list):
            genres = [g.lower() for g in genre_field if g]
        else:
            genres = []

        genre_weight = 0.7
        for g in genres:
            if g in profile["genres"]:
                score += profile["genres"][g] * genre_weight
                break

        tags = book.get("tags", [])
        tag_weight = 0.3
        for tag in tags:
            tag_low = tag.lower() if isinstance(tag, str) else ""
            if tag_low in profile["tags"]:
                score += profile["tags"][tag_low] * tag_weight

        return min(score, 1.0)

    def _collaborative_score(self, user_id: int, book_id: str) -> float:
        """Вычисляет коллаборативную оценку для книги на основе item-based подхода."""
        if self.book_similarity is None:
            return 0.0
        if user_id not in self.user_to_idx or book_id not in self.book_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        b_idx = self.book_to_idx[book_id]

        if self.user_book_matrix is None or self.user_book_matrix.shape[0] <= u_idx:
            return 0.0

        try:
            rated_books = self.user_book_matrix[u_idx].indices
            ratings = self.user_book_matrix[u_idx].data
        except (IndexError, AttributeError):
            return 0.0

        if len(rated_books) == 0:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for idx, r in zip(rated_books, ratings):
            if idx >= self.book_similarity.shape[1]:
                continue
            sim = self.book_similarity[b_idx, idx]
            if sim > 0:
                weighted_sum += sim * r
                total_weight += sim

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight
