package com.samira.bookapp.network

import kotlinx.serialization.Serializable
import retrofit2.Response
import retrofit2.http.*
import javax.net.ssl.*
import java.security.cert.X509Certificate
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit

@Serializable
data class NeuralInteraction(
    val user_id: Long,
    val book_id: String,
    val rating: Float,
    val status: String,
    val timestamp: String? = null
)

@Serializable
data class BookData(
    val id: String,
    val title: String,
    val author: String,
    val genre: String,
    val tags: List<String> = emptyList(),
    val average_rating: Float = 0.0f,
    val cover_url: String = "",
    val description: String = "",
    val is_bestseller: Boolean = false,
    val playlist_url: String = ""
)

@Serializable
data class NeuralBookData(
    val id: String,
    val title: String,
    val author: String,
    val genre: String,
    val tags: List<String> = emptyList(),
    val average_rating: Float = 0.0f,
    val cover_url: String = "",
    val description: String = ""
)

@Serializable
data class NeuralRecommendationRequest(
    val user_id: Long,
    val candidate_books: List<NeuralBookData>,
    val limit: Int = 10
)

@Serializable
data class NeuralRecommendationResponse(
    val recommendations: List<NeuralBookData>,
    val confidence_scores: List<Float>,
    val model_version: String = "1.0",
    val training_data_size: Int = 0,
    val message: String = ""
)

@Serializable
data class HealthResponse(
    val status: String,
    val timestamp: String,
    val models_loaded: Int,
    val total_interactions: Int
)

@Serializable
data class CombinedInteractionRequest(
    val interaction: NeuralInteraction,
    val book_data: NeuralBookData
)

@Serializable
data class InteractionResponse(
    val status: String,
    val message: String,
    val training_loss: Float? = null
)

@Serializable
data class ChatRecommendResponse(
    val results: List<GoogleBook>,
    val is_fallback: Boolean = false,
    val fallback_message: String? = null
)


@Serializable
data class ChatRequest(
    val query: String,
    val user_id: Long,
    val session_id: String? = null
)



@Serializable
data class RegisterRequest(
    val username: String,
    val email: String,
    val password: String
)

@Serializable
data class LoginRequest(
    val username: String,
    val password: String
)

@Serializable
data class UserResponse(
    val user_id: Int,
    val id: Int,
    val username: String,
    val email: String,
    val created_at: String,
    val last_login: String? = null
)

@Serializable
data class UserBookDto(
    val book_id: String,
    val title: String,
    val author: String,
    val genre: String,
    val cover_url: String,
    val average_rating: Float,
    val status: String,
    val user_rating: Float,
    val global_id: String,
    val playlist_url: String = ""
)

@Serializable
data class UserBookUpdateRequest(
    val global_id: String,
    val title: String,
    val author: String,
    val genre: String,
    val cover_url: String = "",
    val description: String = "",
    val status: String,
    val rating: Float = 0.0f,
    val playlist_url: String = ""
)

@Serializable
data class ProfileUpdateRequest(
    val username: String,
    val email: String
)
@Serializable
data class ChatHistoryResponse(
    val history: List<ChatHistoryItem>
)

@Serializable
data class ChatHistoryItem(
    val message: String,
    val is_from_user: Boolean,
    val timestamp: String,
    val data: ChatMessageData? = null
)

@Serializable
data class ChatMessageData(
    val type: String,
    val text: String? = null,
    val books: List<GoogleBook>? = null
)
@Serializable
data class SearchResponse(
    val results: List<GoogleBook>
)

interface NeuralRecommendationService {
    @POST("api/recommend")
    suspend fun getRecommendations(
        @Body request: NeuralRecommendationRequest
    ): NeuralRecommendationResponse

    @PUT("api/user/{user_id}/profile")
    suspend fun updateProfile(
        @Path("user_id") userId: Long,
        @Body request: ProfileUpdateRequest
    ): Response<UserResponse>

    @POST("api/add_interaction")
    suspend fun addInteraction(
        @Body interaction: NeuralInteraction
    ): Response<InteractionResponse>

    @GET("api/books")
    suspend fun getAllBooks(): List<BookData>

    @POST("api/add_interaction_with_book")
    suspend fun addInteractionWithBook(
        @Body request: CombinedInteractionRequest
    ): Response<InteractionResponse>

    @POST("api/train_model/{user_id}")
    suspend fun trainModel(
        @Path("user_id") userId: Long
    ): Response<InteractionResponse>

    @GET("api/user/{user_id}/stats")
    suspend fun getUserStats(
        @Path("user_id") userId: Long
    ): Response<Map<String, Any>>

    @GET("api/system/stats")
    suspend fun getSystemStats(): Response<Map<String, Any>>

    @GET("health")
    suspend fun checkHealth(): HealthResponse

    @POST("api/content_recommend")
    suspend fun getContentRecommendations(
        @Body request: NeuralRecommendationRequest
    ): NeuralRecommendationResponse

    @GET("api/user/{user_id}/recommendation_stats")
    suspend fun getRecommendationStats(
        @Path("user_id") userId: Long
    ): Response<Map<String, Any>>

    @DELETE("api/user/{user_id}/clear_data")
    suspend fun clearUserData(@Path("user_id") userId: Long): Response<InteractionResponse>

    @POST("api/chat_recommend")
    suspend fun chatRecommend(@Body request: ChatRequest): ChatRecommendResponse



    @POST("api/register")
    suspend fun register(@Body request: RegisterRequest): Response<UserResponse>

    @POST("api/login")
    suspend fun login(@Body request: LoginRequest): Response<UserResponse>

    @GET("api/user/{user_id}/books")
    suspend fun getUserBooks(@Path("user_id") userId: Long): List<UserBookDto>

    @POST("api/user/{user_id}/book")
    suspend fun updateUserBook(@Path("user_id") userId: Long, @Body request: UserBookUpdateRequest): Response<Unit>


    @GET("api/book/{book_id}/content")
    suspend fun getBookContent(@Path("book_id") bookId: String): Response<okhttp3.ResponseBody>


    @GET("api/chat_history/{user_id}")
    suspend fun getChatHistory(
        @Path("user_id") userId: Long,
        @Query("session_id") sessionId: String?
    ): Response<ChatHistoryResponse>

    @GET("api/search")
    suspend fun searchBooks(
        @Query("q") query: String,
        @Query("user_id") userId: Long
    ): SearchResponse
}

object NeuralRecommendationApi {
    private const val BASE_URL = "https://bookapp-tu1j.onrender.com"
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    private fun createTrustAllSocketFactory(): SSLSocketFactory {
        val trustAllCerts = arrayOf<TrustManager>(object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        })
        val sslContext = SSLContext.getInstance("TLS")
        sslContext.init(null, trustAllCerts, java.security.SecureRandom())
        return sslContext.socketFactory
    }
    private fun createTrustAllTrustManager(): X509TrustManager {
        return object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        }
    }
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor(loggingInterceptor)
        .hostnameVerifier { _, _ -> true }
        .sslSocketFactory(createTrustAllSocketFactory(), createTrustAllTrustManager())
        .build()

    val retrofit: NeuralRecommendationService by lazy {
        retrofit2.Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(retrofit2.converter.gson.GsonConverterFactory.create())
            .build()
            .create(NeuralRecommendationService::class.java)
    }
}