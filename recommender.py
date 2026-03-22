# recommender.py - с fasttext эмбеддингами
import numpy as np
import fasttext
import fasttext.util
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os


class EmbeddingModel:
    """
    Управление семантическими эмбеддингами книг на основе fasttext.
    """

    def __init__(self):
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self._load_model()

    def _load_model(self):
        """Загружает модель fasttext (скачивает если нет)."""
        try:
            # Проверяем, есть ли уже загруженная модель
            model_path = 'cc.en.300.bin'
            if not os.path.exists(model_path):
                print("📥 Скачиваем модель fasttext...")
                fasttext.util.download_model('ru', if_exists='ignore')
                print("✅ Модель fasttext загружена")

            self.model = fasttext.load_model(model_path)
            print("✅ fasttext модель готова к использованию")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки fasttext модели: {e}")
            self.model = None

    def build_embeddings(self, books: List[Dict[str, Any]]):
        """
        Строит эмбеддинги для списка книг.
        """
        if self.model is None:
            print("⚠️ fasttext модель не загружена, пропускаем эмбеддинги")
            return

        for book in books:
            book_id = book["id"]
            # Составляем текст для эмбеддинга
            parts = [
                book.get("title", ""),
                book.get("author", ""),
                book.get("genre", "") if isinstance(book.get("genre"), str) else " ".join(book.get("genre", [])),
                " ".join(book.get("tags", [])),
                book.get("description", "")
            ]
            text = " ".join(parts).strip()

            if text:
                # Получаем эмбеддинг через fasttext
                embedding = self._get_text_embedding(text)
                if embedding is not None:
                    self.embeddings[book_id] = embedding

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Получает эмбеддинг для текста через fasttext."""
        if self.model is None:
            return None

        try:
            # Разбиваем на слова и усредняем их эмбеддинги
            words = text.lower().split()
            vectors = []

            for word in words:
                # fasttext возвращает вектор для слова
                vec = self.model.get_word_vector(word)
                vectors.append(vec)

            if vectors:
                # Усредняем все векторы слов
                return np.mean(vectors, axis=0)
            else:
                # Если нет слов, возвращаем нулевой вектор
                return np.zeros(300)
        except Exception as e:
            print(f"⚠️ Ошибка получения эмбеддинга: {e}")
            return None

    def get(self, book_id: str) -> Optional[np.ndarray]:
        """Возвращает эмбеддинг книги."""
        return self.embeddings.get(book_id)


class BookRecommender:
    """
    Гибридная рекомендательная система, объединяющая:
    - популярность
    - контентную близость (жанры/теги)
    - семантическую близость (fasttext эмбеддинги)
    - коллаборативную фильтрацию
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()

        self.user_ids: List[int] = []
        self.book_ids: List[str] = []
        self.user_to_idx: Dict[int, int] = {}
        self.book_to_idx: Dict[str, int] = {}
        self.user_book_matrix: Optional[csr_matrix] = None
        self.book_similarity: Optional[np.ndarray] = None
        self.book_popularity: Dict[str, float] = {}
        self.books_metadata: Dict[str, Dict] = {}

    def build(self, interactions: List[Dict], books: List[Dict]):
        """Полностью перестраивает систему."""
        self.books_metadata = {b["id"]: b for b in books}
        self.book_ids = sorted(set(i["book_id"] for i in interactions) | set(self.books_metadata.keys()))
        self.user_ids = sorted(set(i["user_id"] for i in interactions if i.get("rating", 0) > 0))
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.book_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}

        self._build_user_book_matrix(interactions)
        self._build_book_similarity()
        self._build_popularity(interactions)
        self.embedding_model.build_embeddings(books)

        print(f"✅ Система построена: {len(self.user_ids)} пользователей, {len(self.book_ids)} книг")

    def _build_user_book_matrix(self, interactions: List[Dict]):
        """Строит матрицу оценок."""
        n_users = len(self.user_ids)
        n_books = len(self.book_ids)
        matrix = lil_matrix((n_users, n_books), dtype=np.float32)

        for inter in interactions:
            rating = inter.get("rating", 0)
            if rating > 0:
                user_id = inter["user_id"]
                book_id = inter["book_id"]
                if user_id in self.user_to_idx and book_id in self.book_to_idx:
                    u_idx = self.user_to_idx[user_id]
                    b_idx = self.book_to_idx[book_id]
                    matrix[u_idx, b_idx] = rating

        self.user_book_matrix = matrix.tocsr()

    def _build_book_similarity(self):
        """Строит матрицу сходства книг."""
        if self.user_book_matrix is None or self.user_book_matrix.shape[1] < 2:
            self.book_similarity = None
            return

        if self.user_book_matrix.nnz == 0:
            self.book_similarity = None
            return

        book_vectors = self.user_book_matrix.T.tocsr()
        try:
            self.book_similarity = cosine_similarity(book_vectors, dense_output=True)
        except Exception as e:
            print(f"⚠️ Не удалось вычислить сходство: {e}")
            self.book_similarity = None

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

        for book_id, cnt in rating_counts.items():
            avg = rating_sums[book_id] / cnt
            self.book_popularity[book_id] = avg * np.log1p(cnt)

    def recommend(self, user_id: int, candidate_books: List[Dict],
                  user_interactions: List[Dict], top_k: int = 10):
        """Возвращает рекомендации."""
        positive_ratings = [i for i in user_interactions if i.get("rating", 0) > 0]
        rated_book_ids = {i["book_id"] for i in positive_ratings}
        n_ratings = len(positive_ratings)

        if n_ratings == 0:
            return self._popular_books(candidate_books, top_k, exclude=rated_book_ids)

        profile = self._build_content_profile(positive_ratings)
        user_embedding = self._build_user_embedding(positive_ratings)

        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in rated_book_ids:
                continue

            content_score = self._content_score(book, profile)
            semantic_score = self._semantic_score(book_id, user_embedding)
            semantic_norm = (semantic_score + 1) / 2

            if n_ratings < 3:
                score = content_score * 0.6 + semantic_norm * 0.4
            else:
                collab_score = self._collaborative_score(user_id, book_id) / 5.0
                score = content_score * 0.35 + semantic_norm * 0.45 + collab_score * 0.20

            scored.append((book, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    def _popular_books(self, candidate_books: List[Dict], top_k: int, exclude: set):
        """Возвращает популярные книги."""
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

    def _build_content_profile(self, positive_ratings: List[Dict]):
        """Строит профиль пользователя по жанрам/тегам."""
        profile = {"genres": defaultdict(float), "tags": defaultdict(float)}

        for inter in positive_ratings:
            rating = inter["rating"]
            book_id = inter["book_id"]
            meta = self.books_metadata.get(book_id, {})

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
        """Контентная близость."""
        score = 0.0
        genre_field = book.get("genre", "")
        if isinstance(genre_field, str):
            genres = [genre_field.lower()] if genre_field else []
        elif isinstance(genre_field, list):
            genres = [g.lower() for g in genre_field if g]
        else:
            genres = []

        for g in genres:
            if g in profile["genres"]:
                score += profile["genres"][g] * 0.7
                break

        tags = book.get("tags", [])
        for tag in tags:
            tag_low = tag.lower() if isinstance(tag, str) else ""
            if tag_low in profile["tags"]:
                score += profile["tags"][tag_low] * 0.3

        return min(score, 1.0)

    def _build_user_embedding(self, positive_ratings: List[Dict]) -> Optional[np.ndarray]:
        """Средний эмбеддинг пользователя."""
        vectors = []
        weights = []

        for inter in positive_ratings:
            book_id = inter["book_id"]
            rating = inter["rating"]
            vec = self.embedding_model.get(book_id)
            if vec is not None:
                vectors.append(vec)
                weights.append(rating)

        if not vectors:
            return None

        try:
            vectors = np.array(vectors)
            weights = np.array(weights) / np.sum(weights)
            return np.average(vectors, axis=0, weights=weights)
        except Exception:
            return None

    def _semantic_score(self, book_id: str, user_embedding: Optional[np.ndarray]) -> float:
        """Семантическая близость через fasttext."""
        if user_embedding is None:
            return 0.0

        book_vec = self.embedding_model.get(book_id)
        if book_vec is None:
            return 0.0

        norm1 = np.linalg.norm(book_vec)
        norm2 = np.linalg.norm(user_embedding)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_sim = np.dot(book_vec, user_embedding) / (norm1 * norm2)
        return float(cos_sim)

    def _collaborative_score(self, user_id: int, book_id: str) -> float:
        """Коллаборативный скор."""
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
