package com.samira.bookapp.data

import androidx.room.*

@Dao
interface UserDao {
    @Insert
    suspend fun insertUser(user: User): Long

    @Update
    suspend fun updateUser(user: User)

    @Delete
    suspend fun deleteUser(user: User)

    @Query("SELECT * FROM users WHERE id = :userId")
    suspend fun getUserById(userId: Long): User?

    @Query("SELECT * FROM users WHERE username = :username")
    suspend fun getUserByUsername(username: String): User?

    @Query("SELECT * FROM users WHERE email = :email")
    suspend fun getUserByEmail(email: String): User?

    @Query("SELECT COUNT(*) FROM users WHERE username = :username")
    suspend fun isUsernameTaken(username: String): Int

    @Query("SELECT COUNT(*) FROM users WHERE email = :email")
    suspend fun isEmailTaken(email: String): Int

    @Query("UPDATE users SET lastLogin = :timestamp WHERE id = :userId")
    suspend fun updateLastLogin(userId: Long, timestamp: Long)

    @Query("SELECT * FROM users LIMIT 1")
    suspend fun getFirstUser(): User?
    @Query("UPDATE users SET username = :username WHERE id = :userId")
    suspend fun updateUsername(userId: Long, username: String)

    @Query("UPDATE users SET email = :email WHERE id = :userId")
    suspend fun updateEmail(userId: Long, email: String)

    @Query("SELECT COUNT(*) FROM users WHERE email = :email AND id != :userId")
    suspend fun isEmailTakenByOtherUser(email: String, userId: Long): Int
}