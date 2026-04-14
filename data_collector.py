import json
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, delete, func, and_
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func as sql_func
from datetime import datetime, timezone

# --- Модели SQLAlchemy ---
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now())
    last_login: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)


class BookDB(Base):
    __tablename__ = "books"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String)
    genre: Mapped[str] = mapped_column(String)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    average_rating: Mapped[float] = mapped_column(Float, default=0.0)
    cover_url: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    is_bestseller: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "genre": self.genre,
            "tags": self.tags or [],
            "average_rating": self.average_rating,
            "cover_url": self.cover_url,
            "description": self.description,
            "is_bestseller": self.is_bestseller,
        }


class InteractionDB(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    book_id: Mapped[str] = mapped_column(String, ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now())

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "book_id": self.book_id,
            "rating": self.rating,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class ChatHistoryDB(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    message: Mapped[str] = mapped_column(String, nullable=False)
    is_from_user: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now())
    session_id: Mapped[str] = mapped_column(String, nullable=True)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "message": self.message,
            "is_from_user": self.is_from_user,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


# --- DataCollector ---
class DataCollector:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        self._stats_cache = {"total_interactions": 0, "unique_users": 0, "unique_books": 0}

    async def init_db(self):
        """Создаёт таблицы, если их нет."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._refresh_stats()

    async def _refresh_stats(self):
        """Обновляет кэш статистики."""
        async with self.async_session() as session:
            total = await session.scalar(select(func.count(InteractionDB.id)))
            users = await session.scalar(select(func.count(InteractionDB.user_id.distinct())))
            books = await session.scalar(select(func.count(InteractionDB.book_id.distinct())))
            self._stats_cache = {
                "total_interactions": total or 0,
                "unique_users": users or 0,
                "unique_books": books or 0
            }

    # --- Методы для книг и взаимодействий ---
    async def add_books(self, books: List[Dict]):
        """Добавляет или обновляет книги."""
        async with self.async_session() as session:
            for b in books:
                book = await session.get(BookDB, b["id"])
                if not book:
                    book = BookDB(
                        id=b["id"],
                        title=b["title"],
                        author=b.get("author", ""),
                        genre=b.get("genre", ""),
                        tags=b.get("tags", []),
                        average_rating=b.get("average_rating", 0.0),
                        cover_url=b.get("cover_url", ""),
                        description=b.get("description", ""),
                        is_bestseller=b.get("is_bestseller", False)
                    )
                    session.add(book)
                else:
                    # обновляем (на случай, если изменились метаданные)
                    book.title = b["title"]
                    book.author = b.get("author", "")
                    book.genre = b.get("genre", "")
                    book.tags = b.get("tags", [])
                    book.average_rating = b.get("average_rating", 0.0)
                    book.cover_url = b.get("cover_url", "")
                    book.description = b.get("description", "")
                    book.is_bestseller = b.get("is_bestseller", False)
            await session.commit()
            print(f"✅ Добавлено {len(books)} книг в базу")
        await self._refresh_stats()

    async def add_interaction(
        self,
        user_id: int,
        book_id: str,
        rating: float = 0.0,
        status: str = "rated",
        book_data: Dict = None
    ):
        async with self.async_session() as session:
            # Сохраняем/обновляем книгу
            if book_data:
                book = await session.get(BookDB, book_id)
                if not book:
                    book = BookDB(
                        id=book_id,
                        title=book_data["title"],
                        author=book_data.get("author", ""),
                        genre=book_data.get("genre", ""),
                        tags=book_data.get("tags", []),
                        average_rating=book_data.get("average_rating", 0.0),
                        cover_url=book_data.get("cover_url", ""),
                        description=book_data.get("description", ""),
                        is_bestseller=book_data.get("is_bestseller", False)
                    )
                    session.add(book)
                else:
                    # обновляем метаданные
                    book.title = book_data["title"]
                    book.author = book_data.get("author", "")
                    book.genre = book_data.get("genre", "")
                    book.tags = book_data.get("tags", [])
                    book.average_rating = book_data.get("average_rating", 0.0)
                    book.cover_url = book_data.get("cover_url", "")
                    book.description = book_data.get("description", "")
                    book.is_bestseller = book_data.get("is_bestseller", False)

            # Проверяем существующее взаимодействие
            from sqlalchemy import select
            stmt = select(InteractionDB).where(
                InteractionDB.user_id == user_id,
                InteractionDB.book_id == book_id
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Обновляем
                existing.rating = rating
                existing.status = status
                # Если хотите обновлять timestamp, добавьте:
                # existing.timestamp = datetime.now(timezone.utc)
            else:
                # Создаём новое
                interaction = InteractionDB(
                    user_id=user_id,
                    book_id=book_id,
                    rating=rating,
                    status=status
                )
                session.add(interaction)

            await session.commit()
        await self._refresh_stats()

    async def get_user_interactions(self, user_id: int) -> List[Dict]:
        """Возвращает все взаимодействия пользователя."""
        async with self.async_session() as session:
            result = await session.execute(
                select(InteractionDB).where(InteractionDB.user_id == user_id)
            )
            interactions = result.scalars().all()
            return [i.to_dict() for i in interactions]

    async def get_all_interactions(self) -> List[Dict]:
        """Возвращает все взаимодействия (для построения рекомендательной системы)."""
        async with self.async_session() as session:
            result = await session.execute(select(InteractionDB))
            interactions = result.scalars().all()
            return [i.to_dict() for i in interactions]

    async def get_all_books(self) -> List[Dict]:
        """Возвращает все книги."""
        async with self.async_session() as session:
            result = await session.execute(select(BookDB))
            books = result.scalars().all()
            return [b.to_dict() for b in books]

    async def get_book_by_id(self, book_id: str) -> Optional[Dict]:
        """Возвращает книгу по id."""
        async with self.async_session() as session:
            book = await session.get(BookDB, book_id)
            return book.to_dict() if book else None

    async def get_user_stats(self, user_id: int) -> Dict:
        """Статистика пользователя."""
        async with self.async_session() as session:
            interactions = await session.execute(
                select(InteractionDB).where(InteractionDB.user_id == user_id)
            )
            interactions = interactions.scalars().all()
            ratings = [i.rating for i in interactions if i.rating > 0]
            return {
                "total_interactions": len(interactions),
                "ratings_count": len(ratings),
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "unique_books": len(set(i.book_id for i in interactions))
            }

    async def get_all_data_stats(self) -> Dict:
        """Общая статистика (кэшированная)."""
        return self._stats_cache

    async def clear_user_data(self, user_id: int) -> bool:
        """Удаляет все взаимодействия пользователя."""
        try:
            async with self.async_session() as session:
                await session.execute(
                    delete(InteractionDB).where(InteractionDB.user_id == user_id)
                )
                await session.commit()
            await self._refresh_stats()
            return True
        except Exception:
            return False

    # --- Методы для работы с пользователями ---
    async def create_user(self, username: str, email: str, password_hash: str) -> int:
        async with self.async_session() as session:
            user = UserDB(username=username, email=email, password_hash=password_hash)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id

    async def get_user_by_username(self, username: str) -> Optional[UserDB]:
        async with self.async_session() as session:
            result = await session.execute(select(UserDB).where(UserDB.username == username))
            return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int) -> Optional[UserDB]:
        async with self.async_session() as session:
            return await session.get(UserDB, user_id)

    async def update_last_login(self, user_id: int):
        async with self.async_session() as session:
            user = await session.get(UserDB, user_id)
            if user:
                user.last_login = datetime.now(timezone.utc)
                await session.commit()

    async def get_user_by_email(self, email: str) -> Optional[UserDB]:
        async with self.async_session() as session:
            result = await session.execute(select(UserDB).where(UserDB.email == email))
            return result.scalar_one_or_none()

    # --- Методы для работы с историей чата ---
    async def save_chat_message(self, user_id: int, message: str, is_from_user: bool, session_id: str = None, data: Dict = None) -> int:
        async with self.async_session() as session:
            chat = ChatHistoryDB(
                user_id=user_id,
                message=message,
                is_from_user=is_from_user,
                session_id=session_id,
                data=data or {}
            )
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return chat.id

    async def get_chat_history(self, user_id: int, limit: int = 20, session_id: str = None) -> List[Dict]:
        async with self.async_session() as session:
            query = select(ChatHistoryDB).where(ChatHistoryDB.user_id == user_id)
            if session_id:
                query = query.where(ChatHistoryDB.session_id == session_id)
            query = query.order_by(ChatHistoryDB.timestamp.asc()).limit(limit)
            result = await session.execute(query)
            return [row.to_dict() for row in result.scalars().all()]

    async def clear_chat_history(self, user_id: int, session_id: str = None) -> bool:
        async with self.async_session() as session:
            query = delete(ChatHistoryDB).where(ChatHistoryDB.user_id == user_id)
            if session_id:
                query = query.where(ChatHistoryDB.session_id == session_id)
            await session.execute(query)
            await session.commit()
            return True
