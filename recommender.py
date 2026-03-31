import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


class BookRecommender:

    def __init__(self):
        self.user_ids = []
        self.book_ids = []
        self.user_to_idx = {}
        self.book_to_idx = {}

        self.user_book_matrix = None
        self.book_similarity = None
        self.book_popularity = {}
        self.books_metadata = {}

        # для TF-IDF-light
        self.feature_doc_freq = defaultdict(int)
        self.total_books = 0

        self.stopwords = {
            "и","в","на","с","по","за","от","до","что","как",
            "это","его","ее","их","но","а","или","же","для",
            "при","под","из","к","о","у"
        }

    # ================= BUILD =================

    def build(self, interactions: List[Dict], books: List[Dict]):
        self.books_metadata = {b["id"]: b for b in books}
        self.total_books = len(books)

        self.book_ids = sorted(
            set(i["book_id"] for i in interactions) | set(self.books_metadata.keys())
        )

        self.user_ids = sorted(
            set(i["user_id"] for i in interactions if i.get("rating", 0) > 0)
        )

        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.book_to_idx = {b: i for i, b in enumerate(self.book_ids)}

        self._build_feature_stats(books)
        self._build_matrix(interactions)
        self._build_similarity()
        self._build_popularity(interactions)

    # ================= FEATURES =================

    def _tokenize(self, text: str):
        words = text.lower().split()
        result = []

        for w in words[:50]:
            w = w.strip(".,!?()\"«»:-")

            if len(w) < 4:
                continue
            if w in self.stopwords:
                continue

            result.append(w)

        return result

    def _extract_features(self, book: Dict):
        features = []

        if book.get("genre"):
            features.append(f"genre:{book['genre'].lower().strip()}")

        if book.get("author"):
            features.append(f"author:{book['author'].lower().strip()}")

        desc = book.get("description", "")
        if desc:
            words = self._tokenize(desc)
            for w in words:
                features.append(f"word:{w}")

        return features

    def _build_feature_stats(self, books: List[Dict]):
        df = defaultdict(set)

        for book in books:
            features = set(self._extract_features(book))
            for f in features:
                df[f].add(book["id"])

        self.feature_doc_freq = {
            f: len(ids) for f, ids in df.items()
        }

    def _feature_weight(self, f: str):
        df = self.feature_doc_freq.get(f, 1)

        # фильтр шума
        if df < 2:
            return 0.0
        if df > self.total_books * 0.8:
            return 0.0

        return np.log(self.total_books / df)

    # ================= MATRIX =================

    def _build_matrix(self, interactions):
        n_users = len(self.user_ids)
        n_books = len(self.book_ids)

        matrix = np.zeros((n_users, n_books), dtype=np.float32)

        for inter in interactions:
            r = inter.get("rating", 0)
            if r > 0:
                u = inter["user_id"]
                b = inter["book_id"]

                if u in self.user_to_idx and b in self.book_to_idx:
                    matrix[self.user_to_idx[u], self.book_to_idx[b]] = r

        self.user_book_matrix = matrix

    # ================= COLLAB =================

    def _build_similarity(self):
        if self.user_book_matrix is None:
            return

        matrix = self.user_book_matrix
        mask = matrix > 0

        user_means = np.divide(
            matrix.sum(axis=1),
            mask.sum(axis=1),
            out=np.zeros(matrix.shape[0]),
            where=mask.sum(axis=1) != 0
        )

        centered = matrix - user_means[:, None]
        centered[~mask] = 0

        book_vectors = centered.T

        norms = np.linalg.norm(book_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1

        normalized = book_vectors / norms
        sim = np.dot(normalized, normalized.T)

        # 🔥 shrinkage (ключевое улучшение)
        co_counts = np.dot(mask.T, mask.T.T)
        sim = sim * (co_counts / (co_counts + 10))

        np.fill_diagonal(sim, 0)

        self.book_similarity = sim

    def _collab_score(self, user_id, book_id):
        if self.book_similarity is None:
            return 0.0

        if user_id not in self.user_to_idx or book_id not in self.book_to_idx:
            return 0.0

        u_idx = self.user_to_idx[user_id]
        b_idx = self.book_to_idx[book_id]

        user_ratings = self.user_book_matrix[u_idx]
        rated_idx = np.where(user_ratings > 0)[0]

        if len(rated_idx) == 0:
            return 0.0

        sims = self.book_similarity[b_idx, rated_idx]

        weights = sims[sims > 0]
        if len(weights) == 0:
            return 0.0

        return np.sum(sims * user_ratings[rated_idx]) / (np.sum(weights) + 1e-8)

    # ================= CONTENT =================

    def _build_profile(self, interactions):
        profile = defaultdict(float)

        for inter in interactions:
            r = inter["rating"]
            book = self.books_metadata.get(inter["book_id"], {})

            weight = r - 3

            for f in self._extract_features(book):
                profile[f] += weight * self._feature_weight(f)

        norm = sum(abs(v) for v in profile.values())
        if norm > 0:
            for k in profile:
                profile[k] /= norm

        return profile

    def _content_score(self, book, profile):
        score = 0.0

        for f in self._extract_features(book):
            score += profile.get(f, 0.0) * self._feature_weight(f)

        return score

    # ================= POPULAR =================

    def _build_popularity(self, interactions):
        cnt = defaultdict(int)
        s = defaultdict(float)

        for i in interactions:
            r = i.get("rating", 0)
            if r > 0:
                b = i["book_id"]
                cnt[b] += 1
                s[b] += r

        for b in cnt:
            avg = s[b] / cnt[b]
            self.book_popularity[b] = avg * np.log1p(cnt[b])

    # ================= RECOMMEND =================

    def recommend(self, user_id, candidate_books, user_interactions, top_k=10):

        if not candidate_books:
            return [], []

        positives = [i for i in user_interactions if i.get("rating", 0) > 0]
        rated_ids = {i["book_id"] for i in positives}

        if not positives:
            return self._popular(candidate_books, rated_ids, top_k)

        profile = self._build_profile(positives)

        scored = []
        for book in candidate_books:
            b_id = book["id"]

            if b_id in rated_ids:
                continue

            content = self._content_score(book, profile)
            collab = self._collab_score(user_id, b_id) / 5.0

            n = len(positives)

            score = content if n < 3 else 0.6 * content + 0.4 * collab

            scored.append((book, score))

        if not scored:
            return [], []

        # нормализация
        vals = np.array([s for _, s in scored])
        min_v, max_v = vals.min(), vals.max()

        if max_v > min_v:
            vals = (vals - min_v) / (max_v - min_v)
        else:
            vals = np.zeros_like(vals)

        scored = [(scored[i][0], vals[i]) for i in range(len(scored))]

        # ================= MMR =================
        selected = []
        used = set()

        lambda_param = 0.7

        while len(selected) < top_k and scored:
            best = None
            best_score = -1

            for book, score in scored:
                if book["id"] in used:
                    continue

                penalty = 0.0

                for sel in selected:
                    if sel.get("genre") == book.get("genre"):
                        penalty += 0.2

                mmr_score = lambda_param * score - (1 - lambda_param) * penalty

                if mmr_score > best_score:
                    best = (book, score)
                    best_score = mmr_score

            if best is None:
                break

            selected.append(best[0])
            used.add(best[0]["id"])

        scores = [s for _, s in scored[:len(selected)]]

        return selected, scores

    # ================= POPULAR =================

    def _popular(self, books, exclude, top_k):
        scored = []

        for b in books:
            if b["id"] in exclude:
                continue

            score = self.book_popularity.get(
                b["id"], b.get("average_rating", 3.0)
            )
            scored.append((b, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[:top_k]
        if not top:
            return [], []

        books, scores = zip(*top)
        return list(books), list(scores)

    # ================= METRICS =================

    def precision_at_k(self, recommendations, relevant, k=10):
        rec_ids = [b["id"] for b in recommendations[:k]]
        rel_set = set(relevant)

        hits = sum(1 for r in rec_ids if r in rel_set)
        return hits / k if k > 0 else 0.0
