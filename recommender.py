# recommender.py — с gensim вместо fasttext
import numpy as np
from gensim.models import FastText
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
import os

class EmbeddingModel:
    def __init__(self):
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}

    def _load_or_create_model(self):
        model_path = 'fasttext_model.bin'
        if os.path.exists(model_path):
            try:
                self.model = FastText.load(model_path)
                print("✅ Модель fasttext загружена")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                self.model = None

    def build_embeddings(self, books: List[Dict[str, Any]]):
        if self.model is None:
            self._load_or_create_model()
        
        if self.model is None:
            print("⚠️ Модель не загружена, создаём новую")
            self.model = FastText(vector_size=100, window=5, min_count=1, workers=4)
        
        sentences = []
        for book in books:
            parts = [
                book.get("title", ""),
                book.get("author", ""),
                book.get("genre", ""),
                " ".join(book.get("tags", [])),
                book.get("description", "")
            ]
            text = " ".join(parts).strip()
            if text:
                words = text.lower().split()
                if words:
                    sentences.append(words)
        
        if sentences:
            self.model.build_vocab(sentences)
            self.model.train(sentences, total_examples=len(sentences), epochs=10)
            self.model.save('fasttext_model.bin')
            print(f"✅ Модель обучена на {len(sentences)} текстах")
        
        for book in books:
            book_id = book["id"]
            parts = [
                book.get("title", ""),
                book.get("author", ""),
                book.get("genre", ""),
                " ".join(book.get("tags", [])),
                book.get("description", "")
            ]
            text = " ".join(parts).strip()
            if text:
                words = text.lower().split()
                vectors = []
                for word in words:
                    if word in self.model.wv:
                        vectors.append(self.model.wv[word])
                if vectors:
                    self.embeddings[book_id] = np.mean(vectors, axis=0)

    def get(self, book_id: str) -> Optional[np.ndarray]:
        return self.embeddings.get(book_id)


class BookRecommender:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.user_ids: List[int] = []
        self.book_ids: List[str] = []
        self.user_to_idx: Dict[int, int] = {}
        self.book_to_idx: Dict[str, int] = {}
        self.user_book_matrix = None
        self.book_popularity: Dict[str, float] = {}
        self.books_metadata: Dict[str, Dict] = {}

    def build(self, interactions: List[Dict], books: List[Dict]):
        self.books_metadata = {b["id"]: b for b in books}
        self.book_ids = sorted(set(i["book_id"] for i in interactions) | set(self.books_metadata.keys()))
        self.user_ids = sorted(set(i["user_id"] for i in interactions if i.get("rating", 0) > 0))
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.book_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        self._build_user_book_matrix(interactions)
        self._build_popularity(interactions)
        self.embedding_model.build_embeddings(books)
        print(f"✅ Система построена: {len(self.user_ids)} пользователей, {len(self.book_ids)} книг")

    def _build_user_book_matrix(self, interactions: List[Dict]):
        n_users = len(self.user_ids)
        n_books = len(self.book_ids)
        matrix = [[0.0] * n_books for _ in range(n_users)]
        
        for inter in interactions:
            rating = inter.get("rating", 0)
            if rating > 0:
                user_id = inter["user_id"]
                book_id = inter["book_id"]
                if user_id in self.user_to_idx and book_id in self.book_to_idx:
                    u_idx = self.user_to_idx[user_id]
                    b_idx = self.book_to_idx[book_id]
                    matrix[u_idx][b_idx] = rating
        
        self.user_book_matrix = np.array(matrix, dtype=np.float32)

    def _build_popularity(self, interactions: List[Dict]):
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
        positive = [i for i in user_interactions if i.get("rating", 0) > 0]
        rated_ids = {i["book_id"] for i in positive}
        n_ratings = len(positive)
        
        if n_ratings == 0:
            return self._popular_books(candidate_books, top_k, rated_ids)
        
        profile = self._build_content_profile(positive)
        user_emb = self._build_user_embedding(positive)
        
        scored = []
        for book in candidate_books:
            book_id = book["id"]
            if book_id in rated_ids:
                continue
            
            content = self._content_score(book, profile)
            semantic = self._semantic_score(book_id, user_emb)
            semantic = (semantic + 1) / 2
            
            score = content * 0.6 + semantic * 0.4 if n_ratings < 3 else content * 0.35 + semantic * 0.45 + self._collaborative_score(user_id, book_id) * 0.2 / 5.0
            scored.append((book, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            return [], []
        books, scores = zip(*top)
        return list(books), list(scores)

    def _popular_books(self, books: List[Dict], top_k: int, exclude: set):
        scored = [(b, self.book_popularity.get(b["id"], b.get("average_rating", 3.0))) for b in books if b["id"] not in exclude]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return ([b for b, _ in top], [s for _, s in top]) if top else ([], [])

    def _build_content_profile(self, ratings: List[Dict]):
        profile = {"genres": defaultdict(float), "tags": defaultdict(float)}
        for inter in ratings:
            rating = inter["rating"]
            meta = self.books_metadata.get(inter["book_id"], {})
            genre = meta.get("genre", "").lower()
            if genre:
                profile["genres"][genre] += rating
            for tag in meta.get("tags", []):
                if isinstance(tag, str):
                    profile["tags"][tag.lower()] += 1
        for key in ["genres", "tags"]:
            total = sum(profile[key].values())
            if total > 0:
                for k in profile[key]:
                    profile[key][k] /= total
        return profile

    def _content_score(self, book: Dict, profile: Dict) -> float:
        score = 0.0
        genre = book.get("genre", "").lower()
        if genre and genre in profile["genres"]:
            score += profile["genres"][genre] * 0.7
        tags = book.get("tags", [])
        for tag in tags:
            tag_low = tag.lower() if isinstance(tag, str) else ""
            if tag_low in profile["tags"]:
                score += profile["tags"][tag_low] * 0.3
        return min(score, 1.0)

    def _build_user_embedding(self, ratings: List[Dict]) -> Optional[np.ndarray]:
        vectors, weights = [], []
        for inter in ratings:
            vec = self.embedding_model.get(inter["book_id"])
            if vec is not None:
                vectors.append(vec)
                weights.append(inter["rating"])
        if not vectors:
            return None
        vectors = np.array(vectors)
        weights = np.array(weights) / np.sum(weights)
        return np.average(vectors, axis=0, weights=weights)

    def _semantic_score(self, book_id: str, user_emb: Optional[np.ndarray]) -> float:
        if user_emb is None:
            return 0.0
        book_vec = self.embedding_model.get(book_id)
        if book_vec is None:
            return 0.0
        norm1, norm2 = np.linalg.norm(book_vec), np.linalg.norm(user_emb)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(book_vec, user_emb) / (norm1 * norm2))

    def _collaborative_score(self, user_id: int, book_id: str) -> float:
        if user_id not in self.user_to_idx or book_id not in self.book_to_idx:
            return 0.0
        u_idx = self.user_to_idx[user_id]
        b_idx = self.book_to_idx[book_id]
        if self.user_book_matrix is None or u_idx >= len(self.user_book_matrix):
            return 0.0
        rated = [(idx, r) for idx, r in enumerate(self.user_book_matrix[u_idx]) if r > 0]
        if not rated:
            return 0.0
        sim_scores = []
        for idx, r in rated:
            if idx < len(self.user_book_matrix) and idx != b_idx:
                sim_scores.append(r)
        return sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
