package com.samira.bookapp.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface BookDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertBook(book: Book): Long

    @Query("SELECT * FROM books WHERE id = :bookId AND userId = :userId")
    suspend fun getBookById(bookId: Long, userId: Long): Book?

    @Query("SELECT * FROM books WHERE title = :title AND author = :author AND userId = :userId LIMIT 1")
    suspend fun getBookByTitleAndAuthor(title: String, author: String, userId: Long): Book?

    @Query("SELECT * FROM books WHERE userId = :userId")
    suspend fun getAllBooksForUser(userId: Long): List<Book>

    @Query("SELECT b.* FROM books b INNER JOIN user_books ub ON b.id = ub.bookId WHERE ub.userId = :userId AND ub.status = :status AND b.userId = :userId")
    suspend fun getUserBooksByStatus(userId: Long, status: BookStatus): List<Book>

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAllBooks(books: List<Book>)

    @Query("DELETE FROM books WHERE id = :bookId AND userId = :userId")
    suspend fun deleteBook(bookId: Long, userId: Long)

    @Query("SELECT * FROM books WHERE userId != :userId ORDER BY averageRating DESC LIMIT 10")
    suspend fun getOtherUsersBooks(userId: Long): List<Book>

    @Query("""
        SELECT * FROM books 
        WHERE userId != :currentUserId
        AND averageRating >= 3.0 
        AND ratingsCount >= 5
        ORDER BY averageRating DESC, ratingsCount DESC 
        LIMIT :limit
    """)
    suspend fun getPopularBooks(currentUserId: Long, limit: Int = 50): List<Book>

    @Query("""
        SELECT * FROM books 
        WHERE language = 'ru' 
        AND userId = :currentUserId
        AND averageRating >= 3.5
        ORDER BY averageRating DESC 
        LIMIT :limit
    """)
    suspend fun getRussianBooks(currentUserId: Long, limit: Int = 20): List<Book>


    @Query("""
        SELECT * FROM books 
        WHERE title LIKE '%' || :query || '%' 
           OR author LIKE '%' || :query || '%'
           OR genre LIKE '%' || :query || '%'
        LIMIT 20
    """)
    suspend fun searchBooksLocally(query: String): List<Book>


    @Query("""
    SELECT * FROM books 
    WHERE userId = :userId 
    AND (title LIKE '%' || :query || '%' 
         OR author LIKE '%' || :query || '%' 
         OR genre LIKE '%' || :query || '%')
    LIMIT 20
""")
    suspend fun searchBooksLocally(userId: Long, query: String): List<Book>

    @Query("SELECT COUNT(*) FROM books WHERE userId = :userId")
    suspend fun getUserBookCount(userId: Long): Int

    @Query("SELECT * FROM books WHERE globalId = :globalId")
    suspend fun getBooksByGlobalId(globalId: String): List<Book>

    @Query("UPDATE books SET averageRating = :rating, ratingsCount = :count WHERE id = :bookId")
    suspend fun updateBookRating(bookId: Long, rating: Float, count: Int)

    @Query("""
    SELECT b.* FROM books b
    WHERE b.userId = :userId
    AND b.id NOT IN (SELECT ub.bookId FROM user_books ub WHERE ub.userId = :userId)
    ORDER BY b.averageRating DESC, b.ratingsCount DESC
    LIMIT :limit
""")
    suspend fun getRecommendedBooksForUser(userId: Long, limit: Int): List<Book>

    @Query("SELECT * FROM books WHERE userId = :userId AND globalId = :globalId LIMIT 1")
    suspend fun getBookByGlobalIdForUser(userId: Long, globalId: String): Book?
    // BookDao.kt
    @Query("SELECT * FROM books WHERE globalId = :globalId AND userId = :userId LIMIT 1")
    suspend fun getBookByGlobalId(globalId: String, userId: Long): Book?




}