package com.samira.bookapp.data

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.samira.bookapp.network.*

class HybridNeuralRepository {

    suspend fun checkServerHealth(): Boolean {
        return try {
            withContext(Dispatchers.IO) {
                val response = NeuralRecommendationApi.retrofit.checkHealth()
                Log.d("HYBRID", "✅ Сервер доступен: ${response.status}")
                response.status == "healthy"
            }
        } catch (e: Exception) {
            Log.e("HYBRID", "❌ Сервер недоступен: ${e.message}")
            false
        }
    }

    suspend fun getHybridRecommendationsFromServer(
        userId: Long,
        candidateBooks: List<NeuralBookData>,
        limit: Int = 10
    ): List<NeuralBookData> {
        return withContext(Dispatchers.IO) {
            try {
                val request = NeuralRecommendationRequest(
                    user_id = userId,
                    candidate_books = candidateBooks,
                    limit = limit
                )
                val response = NeuralRecommendationApi.retrofit.getRecommendations(request)
                Log.d("HYBRID", "✅ Получены рекомендации с сервера: ${response.recommendations.size}")
                response.recommendations
            } catch (e: Exception) {
                Log.e("HYBRID", "❌ Ошибка получения рекомендаций с сервера: ${e.message}")
                // Fallback на локальные
                candidateBooks
                    .sortedByDescending { it.average_rating }
                    .take(limit)
            }
        }
    }

    suspend fun sendInteraction(
        userId: Long,
        bookId: String,
        rating: Float,
        status: String,
        bookData: NeuralBookData? = null
    ): Boolean {
        return try {
            withContext(Dispatchers.IO) {
                val interaction = NeuralInteraction(
                    user_id = userId,
                    book_id = bookId,
                    rating = rating,
                    status = status
                )

                val response = if (bookData != null) {
                    val request = CombinedInteractionRequest(
                        interaction = interaction,
                        book_data = bookData
                    )
                    NeuralRecommendationApi.retrofit.addInteractionWithBook(request)
                } else {
                    NeuralRecommendationApi.retrofit.addInteraction(interaction)
                }

                val success = response.isSuccessful
                if (success) {
                    Log.d("HYBRID", "✅ Взаимодействие отправлено в гибридную систему")
                }
                success
            }
        } catch (e: Exception) {
            Log.e("HYBRID", "❌ Ошибка отправки взаимодействия: ${e.message}")
            false
        }
    }

    suspend fun getHybridRecommendations(
        userId: Long,
        candidateBooks: List<NeuralBookData>,
        limit: Int = 10
    ): List<NeuralBookData> {
        return try {
            withContext(Dispatchers.IO) {
                val request = NeuralRecommendationRequest(
                    user_id = userId,
                    candidate_books = candidateBooks,
                    limit = limit
                )

                try {
                    val response = NeuralRecommendationApi.retrofit.getRecommendations(request)
                    Log.d("HYBRID", "✅ Получены гибридные рекомендации: ${response.recommendations.size} книг")
                    response.recommendations
                } catch (e: Exception) {
                    Log.e("HYBRID", "❌ Ошибка гибридных рекомендаций: ${e.message}")
                    // Fallback: простые рекомендации по рейтингу
                    candidateBooks
                        .sortedByDescending { it.average_rating }
                        .take(limit)
                }
            }
        } catch (e: Exception) {
            Log.e("HYBRID", "❌ Ошибка получения гибридных рекомендаций: ${e.message}")
            emptyList()
        }
    }

    suspend fun getUserStats(userId: Long): Map<String, Any> {
        return try {
            withContext(Dispatchers.IO) {
                try {
                    val response = NeuralRecommendationApi.retrofit.getUserStats(userId)
                    if (response.isSuccessful && response.body() != null) {
                        response.body()!!.toMutableMap().apply {
                            put("system_type", "гибридная")
                        }
                    } else {
                        mapOf(
                            "user_id" to userId,
                            "system_type" to "гибридная",
                            "interactions" to 0,
                            "ratings" to 0,
                            "recommendation_quality" to "базовая"
                        )
                    }
                } catch (e: Exception) {
                    Log.e("HYBRID", "❌ Ошибка статистики: ${e.message}")
                    mapOf(
                        "user_id" to userId,
                        "system_type" to "гибридная",
                        "recommendation_quality" to "базовая"
                    )
                }
            }
        } catch (e: Exception) {
            Log.e("HYBRID", "❌ Ошибка получения статистики: ${e.message}")
            emptyMap()
        }
    }

    suspend fun clearUserData(userId: Long): Boolean {
        return try {
            withContext(Dispatchers.IO) {
                try {
                    val response = NeuralRecommendationApi.retrofit.clearUserData(userId)
                    response.isSuccessful
                } catch (e: Exception) {
                    Log.d("HYBRID", "🧹 Очистка данных пользователя $userId (локально)")
                    true
                }
            }
        } catch (e: Exception) {
            Log.e("HYBRID", "❌ Ошибка очистки данных: ${e.message}")
            false
        }
    }
}