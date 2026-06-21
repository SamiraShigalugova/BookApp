package com.samira.bookapp.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface UserBookDao {
    @Update
    suspend fun updateUserBook(userBook: UserBook)
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUserBook(userBook: UserBook)

    @Query("SELECT * FROM user_books WHERE userId = :userId AND bookId = :bookId")
    suspend fun getUserBook(userId: Long, bookId: Long): UserBook?

    @Query("SELECT * FROM user_books WHERE userId = :userId AND status = :status")
    suspend fun getUserBooksByStatus(userId: Long, status: BookStatus): List<UserBook>

    @Query("UPDATE user_books SET status = :status, userRating = :rating WHERE userId = :userId AND bookId = :bookId")
    suspend fun updateUserBookStatusAndRating(userId: Long, bookId: Long, status: BookStatus, rating: Float)

    @Query("UPDATE user_books SET userRating = :rating WHERE userId = :userId AND bookId = :bookId")
    suspend fun updateBookRating(userId: Long, bookId: Long, rating: Float)

    @Delete
    suspend fun deleteUserBook(userBook: UserBook)

    @Query("DELETE FROM user_books WHERE userId = :userId AND bookId = :bookId")
    suspend fun deleteUserBook(userId: Long, bookId: Long)

    @Query("SELECT * FROM user_books WHERE userId = :userId")
    suspend fun getAllUserBooks(userId: Long): List<UserBook>

    @Query("SELECT * FROM user_books WHERE userId = :userId AND userRating > 0")
    suspend fun getUserBooksWithRatings(userId: Long): List<UserBook>

    @Query("SELECT COUNT(*) FROM user_books WHERE userId = :userId")
    suspend fun getUserBookCount(userId: Long): Int

    @Query("SELECT bookId FROM user_books WHERE userId = :userId")
    suspend fun getUserBookIds(userId: Long): List<Long>
    @Query("SELECT * FROM user_books WHERE bookId = :bookId")
    suspend fun getUserBooksForBook(bookId: Long): List<UserBook>

    @Query("SELECT EXISTS(SELECT 1 FROM user_books WHERE userId = :userId AND bookId = :bookId)")
    suspend fun isBookInUserLibrary(userId: Long, bookId: Long): Boolean


}