import json
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, delete, func
from database import Base, Book, Interaction
from collections import defaultdict

class DataCollector:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        # Для быстрой статистики (можно пересчитывать каждый раз)
        self.stats = {"total_interactions": 0, "unique_users": 0, "unique_books": 0}

    async def init_db(self):
        """Создаёт таблицы, если их нет."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Обновляем статистику
        await self._refresh_stats()

    async def _refresh_stats(self):
        async with self.async_session() as session:
            total = await session.scalar(select(func.count(Interaction.id)))
            users = await session.scalar(select(func.count(Interaction.user_id.distinct())))
            books = await session.scalar(select(func.count(Interaction.book_id.distinct())))
            self.stats = {
                "total_interactions": total or 0,
                "unique_users": users or 0,
                "unique_books": books or 0
            }

    async def add_books(self, books: List[Dict]):
        """Добавляет или обновляет книги."""
        async with self.async_session() as session:
            for b in books:
                book = await session.get(Book, b["id"])
                if not book:
                    book = Book(
                        id=b["id"],
                        title=b["title"],
                        author=b.get("author", ""),
                        genre=b.get("genre", ""),
                        tags=b.get("tags", []),
                        average_rating=b.get("average_rating", 0.0),
                        cover_url=b.get("cover_url", ""),
                        description=b.get("description", ""),
                    )
                    session.add(book)
                else:
                    # обновляем поля (если нужно)
                    book.title = b["title"]
                    book.author = b.get("author", "")
                    book.genre = b.get("genre", "")
                    book.tags = b.get("tags", [])
                    book.average_rating = b.get("average_rating", 0.0)
                    book.cover_url = b.get("cover_url", "")
                    book.description = b.get("description", "")
            await session.commit()
        await self._refresh_stats()

    async def add_interaction(
        self,
        user_id: int,
        book_id: str,
        rating: float = 0.0,
        status: str = "rated",
        book_data: Dict = None
    ):
        """
        Добавляет взаимодействие. Если передан book_data, книга сохраняется/обновляется.
        """
        async with self.async_session() as session:
            # если переданы данные книги, сохраняем
            if book_data:
                book = await session.get(Book, book_id)
                if not book:
                    book = Book(
                        id=book_id,
                        title=book_data["title"],
                        author=book_data.get("author", ""),
                        genre=book_data.get("genre", ""),
                        tags=book_data.get("tags", []),
                        average_rating=book_data.get("average_rating", 0.0),
                        cover_url=book_data.get("cover_url", ""),
                        description=book_data.get("description", ""),
                    )
                    session.add(book)
                else:
                    # обновляем существующую книгу (например, если изменились поля)
                    book.title = book_data["title"]
                    book.author = book_data.get("author", "")
                    book.genre = book_data.get("genre", "")
                    book.tags = book_data.get("tags", [])
                    book.average_rating = book_data.get("average_rating", 0.0)
                    book.cover_url = book_data.get("cover_url", "")
                    book.description = book_data.get("description", "")

            # добавляем взаимодействие
            interaction = Interaction(
                user_id=user_id,
                book_id=book_id,
                rating=rating,
                status=status
            )
            session.add(interaction)
            await session.commit()
        await self._refresh_stats()

    async def get_user_interactions(self, user_id: int) -> List[Dict]:
        """Возвращает список взаимодействий пользователя."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Interaction).where(Interaction.user_id == user_id)
            )
            interactions = result.scalars().all()
            return [i.to_dict() for i in interactions]

    async def get_all_interactions(self) -> List[Dict]:
        """Возвращает все взаимодействия."""
        async with self.async_session() as session:
            result = await session.execute(select(Interaction))
            interactions = result.scalars().all()
            return [i.to_dict() for i in interactions]

    async def get_all_books(self) -> List[Dict]:
        """Возвращает все книги."""
        async with self.async_session() as session:
            result = await session.execute(select(Book))
            books = result.scalars().all()
            return [b.to_dict() for b in books]

    async def get_user_stats(self, user_id: int) -> Dict:
        """Статистика пользователя."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Interaction).where(Interaction.user_id == user_id)
            )
            interactions = result.scalars().all()
            ratings = [i.rating for i in interactions if i.rating > 0]
            return {
                "total_interactions": len(interactions),
                "ratings_count": len(ratings),
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "unique_books": len(set(i.book_id for i in interactions))
            }

    async def get_all_data_stats(self) -> Dict:
        """Общая статистика."""
        return self.stats

    async def clear_user_data(self, user_id: int) -> bool:
        """Удаляет все взаимодействия пользователя."""
        try:
            async with self.async_session() as session:
                await session.execute(
                    delete(Interaction).where(Interaction.user_id == user_id)
                )
                await session.commit()
            await self._refresh_stats()
            return True
        except Exception:
            return False
