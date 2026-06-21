package com.samira.bookapp.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(
    tableName = "user_books",
    primaryKeys = ["userId", "bookId"]
)
data class UserBook(
    val userId: Long,
    val bookId: Long,
    val status: BookStatus,
    val userRating: Float = 0.0f,
    val userReview: String = "",
    val addedDate: Long = System.currentTimeMillis()
)

enum class BookStatus {
    WANT_TO_READ,
    READING,
    FINISHED,
    DROPPED
}