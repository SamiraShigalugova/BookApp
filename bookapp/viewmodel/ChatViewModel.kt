package com.samira.bookapp.viewmodel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.samira.bookapp.network.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID

sealed class ChatMessage {
    data class UserMessage(val text: String) : ChatMessage()
    data class TextResponse(val text: String) : ChatMessage()
    data class BookRecommendation(val books: List<GoogleBook>) : ChatMessage()
}

class ChatViewModel(private val userId: Long) : ViewModel() {

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val sessionId = UUID.randomUUID().toString()

    init {
        Log.e("CHAT_DEBUG", "Инициализация ChatViewModel, user_id=$userId")
        loadChatHistory()
    }
    private fun loadChatHistory() {
        viewModelScope.launch {
            try {
                val response = NeuralRecommendationApi.retrofit.getChatHistory(userId, null)
                if (response.isSuccessful && response.body() != null) {
                    val history = response.body()!!.history
                    val messages = history.mapNotNull { item ->
                        if (item.is_from_user) {
                            ChatMessage.UserMessage(item.message)
                        } else {
                            val data = item.data
                            if (data?.type == "books" && !data.books.isNullOrEmpty()) {
                                ChatMessage.BookRecommendation(data.books)
                            } else if (data?.type == "text") {
                                ChatMessage.TextResponse(data.text ?: item.message)
                            } else {

                                ChatMessage.TextResponse(item.message)
                            }
                        }
                    }
                    _messages.value = messages
                }
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Ошибка загрузки истории", e)
            }
        }
    }

    fun sendQuery(query: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _messages.value = _messages.value + ChatMessage.UserMessage(query)

            try {
                val request = ChatRequest(
                    query = query,
                    user_id = userId,
                    session_id = sessionId
                )
                val response = NeuralRecommendationApi.retrofit.chatRecommend(request)

                val googleBooks = response.results.map { dto ->
                    GoogleBook(
                        id = dto.id,
                        volumeInfo = VolumeInfo(
                            title = dto.volumeInfo.title,
                            authors = dto.volumeInfo.authors ?: emptyList(),
                            description = dto.volumeInfo.description,
                            categories = dto.volumeInfo.categories ?: emptyList(),
                            averageRating = dto.volumeInfo.averageRating ?: 0f,
                            ratingsCount = dto.volumeInfo.ratingsCount ?: 0,
                            imageLinks = dto.volumeInfo.imageLinks?.let {
                                ImageLinks(thumbnail = it.thumbnail)
                            },
                            language = dto.volumeInfo.language
                        )
                    )
                }

                if (googleBooks.isNotEmpty()) {
                    if (response.is_fallback && response.fallback_message != null) {
                        _messages.value = _messages.value + ChatMessage.TextResponse(response.fallback_message)
                    }
                    _messages.value = _messages.value + ChatMessage.BookRecommendation(googleBooks)
                } else {
                    _messages.value = _messages.value + ChatMessage.TextResponse("Извините, ничего не нашёл. Попробуйте изменить запрос.")
                }
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Error: ${e.message}", e)
                _messages.value = _messages.value + ChatMessage.TextResponse("Ошибка соединения. Попробуйте позже.")
            } finally {
                _isLoading.value = false
            }
        }
    }

    private fun mapToGoogleBook(map: Map<String, Any>): GoogleBook? {
        return try {
            val id = map["id"] as? String ?: return null
            val volumeInfoMap = map["volumeInfo"] as? Map<String, Any> ?: return null
            val title = volumeInfoMap["title"] as? String ?: ""
            val authors = volumeInfoMap["authors"] as? List<String> ?: emptyList()
            val description = volumeInfoMap["description"] as? String
            val categories = volumeInfoMap["categories"] as? List<String> ?: emptyList()
            val averageRating = (volumeInfoMap["averageRating"] as? Number)?.toFloat() ?: 0f
            val ratingsCount = (volumeInfoMap["ratingsCount"] as? Number)?.toInt() ?: 0
            val imageLinksMap = volumeInfoMap["imageLinks"] as? Map<String, String>
            val thumbnail = imageLinksMap?.get("thumbnail")
            val language = volumeInfoMap["language"] as? String

            GoogleBook(
                id = id,
                volumeInfo = VolumeInfo(
                    title = title,
                    authors = authors,
                    publishedDate = null,
                    description = description,
                    categories = categories,
                    averageRating = averageRating,
                    ratingsCount = ratingsCount,
                    imageLinks = if (thumbnail != null) ImageLinks(thumbnail = thumbnail) else null,
                    language = language
                )
            )
        } catch (e: Exception) {
            Log.e("ChatViewModel", "Error parsing book: ${e.message}")
            null
        }
    }

    class Factory(private val userId: Long) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ChatViewModel(userId) as T
        }
    }
}