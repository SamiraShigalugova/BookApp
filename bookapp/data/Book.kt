package com.samira.bookapp.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "books")
data class Book(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val userId: Long,
    val title: String,
    val author: String,
    val genre: String,
    val description: String,
    val coverUrl: String = "",
    var averageRating: Float = 0.0f,
    val tags: String = "",
    val language: String = "en",
    var ratingsCount: Int = 0,
    val globalId: String = "",
    val isBestseller: Boolean = false,
    val playlistUrl: String = "",
    val sourceFile: String = ""
)