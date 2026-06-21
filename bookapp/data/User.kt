package com.samira.bookapp.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date

@Entity(tableName = "users")
data class User(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val username: String,
    val email: String? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val lastLogin: Long = 0L,
    val preferences: String = "",
    val profileImageUrl: String = "",
    val isActive: Boolean = true,
    val passwordHash: String = "",
    val isGuest: Boolean = false,
) {
    companion object {
        const val DEFAULT_USER_ID = 1L
    }
}