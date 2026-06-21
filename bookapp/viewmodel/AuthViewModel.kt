package com.samira.bookapp.viewmodel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.samira.bookapp.data.User
import com.samira.bookapp.network.*
import com.samira.bookapp.utils.ValidationHelper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.IOException

class AuthViewModel(private val context: Context) : ViewModel() {

    companion object {
        private const val PREFS_NAME = "auth_prefs"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_USERNAME = "username"
        private const val KEY_IS_LOGGED_IN = "is_logged_in"
        private const val KEY_EMAIL = "email"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _currentUser = MutableStateFlow<User?>(null)
    val currentUser: StateFlow<User?> = _currentUser.asStateFlow()

    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _validationErrors = MutableStateFlow<Map<String, String>>(emptyMap())
    val validationErrors: StateFlow<Map<String, String>> = _validationErrors.asStateFlow()

    init {
        checkCurrentSession()
    }

    private fun checkCurrentSession() {
        val userId = prefs.getLong(KEY_USER_ID, -1L)
        val username = prefs.getString(KEY_USERNAME, null)
        val email = prefs.getString(KEY_EMAIL, null)
        val isLoggedIn = prefs.getBoolean(KEY_IS_LOGGED_IN, false)

        if (isLoggedIn && userId != -1L && username != null) {
            _currentUser.value = User(
                id = userId,
                username = username,
                email = email ?: "",
                createdAt = System.currentTimeMillis()
            )
            _authState.value = AuthState.Authenticated(_currentUser.value!!)
        } else {
            _authState.value = AuthState.NotAuthenticated
        }
    }

    fun register(username: String, email: String, password: String, confirmPassword: String) {
        val usernameValidation = ValidationHelper.isValidUsername(username)
        val emailValidation = ValidationHelper.isValidEmail(email)
        val passwordValidation = ValidationHelper.isValidPassword(password)
        val matchValidation = ValidationHelper.doPasswordsMatch(password, confirmPassword)

        val errors = mutableMapOf<String, String>()

        if (usernameValidation is ValidationHelper.ValidationResult.Error) {
            errors["username"] = usernameValidation.message
        }
        if (emailValidation is ValidationHelper.ValidationResult.Error) {
            errors["email"] = emailValidation.message
        }
        if (passwordValidation is ValidationHelper.ValidationResult.Error) {
            errors["password"] = passwordValidation.message
        }
        if (matchValidation is ValidationHelper.ValidationResult.Error) {
            errors["confirmPassword"] = matchValidation.message
        }

        if (errors.isNotEmpty()) {
            _validationErrors.value = errors
            _authState.value = AuthState.Error(errors.values.joinToString("\n"))
            return
        }

        _validationErrors.value = emptyMap()

        viewModelScope.launch {
            _isLoading.value = true
            _authState.value = AuthState.Loading

            try {
                val response = NeuralRecommendationApi.retrofit.register(
                    RegisterRequest(username, email, password)
                )
                if (response.isSuccessful && response.body() != null) {
                    val userDto = response.body()!!
                    val user = User(
                        id = userDto.user_id.toLong(),
                        username = userDto.username,
                        email = userDto.email,
                        createdAt = userDto.created_at?.toLongOrNull() ?: System.currentTimeMillis()
                    )
                    _currentUser.value = user
                    saveSession(user.id, user.username, user.email ?: "")
                    _authState.value = AuthState.Authenticated(user)
                    Log.d("AuthViewModel", "✅ Успешная регистрация: ${user.username}")
                } else {
                    val errorBody = response.errorBody()?.string()
                    val errorMsg = errorBody?.let {
                        try {
                            val json = org.json.JSONObject(it)
                            json.optString("detail", "Ошибка регистрации")
                        } catch (e: Exception) {
                            it
                        }
                    } ?: "Ошибка регистрации"
                    _authState.value = AuthState.Error(errorMsg)
                    Log.e("AuthViewModel", "❌ Ошибка регистрации: $errorMsg")
                }
            } catch (e: IOException) {
                _authState.value = AuthState.Error("Ошибка сети: ${e.message}")
                Log.e("AuthViewModel", "Network error", e)
            } catch (e: HttpException) {
                _authState.value = AuthState.Error("Ошибка сервера: ${e.message}")
                Log.e("AuthViewModel", "HTTP error", e)
            } catch (e: Exception) {
                _authState.value = AuthState.Error("Ошибка: ${e.message}")
                Log.e("AuthViewModel", "Unexpected error", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun login(username: String, password: String) {
        if (username.isBlank() || password.isBlank()) {
            _authState.value = AuthState.Error("Заполните все поля")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            _authState.value = AuthState.Loading

            try {
                val response = NeuralRecommendationApi.retrofit.login(
                    LoginRequest(username, password)
                )
                if (response.isSuccessful && response.body() != null) {
                    val userDto = response.body()!!
                    val user = User(
                        id = userDto.user_id.toLong(),
                        username = userDto.username,
                        email = userDto.email,
                        createdAt = userDto.created_at?.toLongOrNull() ?: System.currentTimeMillis()
                    )
                    _currentUser.value = user
                    saveSession(user.id, user.username, user.email ?: "")
                    _authState.value = AuthState.Authenticated(user)
                    Log.d("AuthViewModel", "✅ Успешный вход: ${user.username}")
                } else {
                    val errorBody = response.errorBody()?.string()
                    val errorMsg = errorBody?.let {
                        try {
                            val json = org.json.JSONObject(it)
                            json.optString("detail", "Неверные учётные данные")
                        } catch (e: Exception) {
                            it
                        }
                    } ?: "Неверные учётные данные"
                    _authState.value = AuthState.Error(errorMsg)
                    Log.e("AuthViewModel", "❌ Ошибка входа: $errorMsg")
                }
            } catch (e: IOException) {
                _authState.value = AuthState.Error("Ошибка сети: ${e.message}")
                Log.e("AuthViewModel", "Network error", e)
            } catch (e: HttpException) {
                _authState.value = AuthState.Error("Ошибка сервера: ${e.message}")
                Log.e("AuthViewModel", "HTTP error", e)
            } catch (e: Exception) {
                _authState.value = AuthState.Error("Ошибка: ${e.message}")
                Log.e("AuthViewModel", "Unexpected error", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun updateProfile(newUsername: String, newEmail: String, onResult: (Boolean, String) -> Unit) {
        val currentUser = _currentUser.value
        if (currentUser == null) {
            onResult(false, "Пользователь не найден")
            return
        }
        if (newUsername.isBlank()) {
            onResult(false, "Имя пользователя не может быть пустым")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val response = NeuralRecommendationApi.retrofit.updateProfile(
                    currentUser.id,
                    ProfileUpdateRequest(newUsername, newEmail)
                )

                if (response.isSuccessful && response.body() != null) {
                    val userDto = response.body()!!
                    val updatedUser = User(
                        id = userDto.id.toLong(),
                        username = userDto.username,
                        email = userDto.email,
                        createdAt = userDto.created_at?.toLongOrNull() ?: System.currentTimeMillis(),
                        lastLogin = userDto.last_login?.toLongOrNull() ?: 0L
                    )
                    _currentUser.value = updatedUser
                    saveSession(updatedUser.id, updatedUser.username, updatedUser.email ?: "")
                    onResult(true, "Профиль обновлён")
                    Log.d("AuthViewModel", "✅ Профиль обновлён на сервере")
                } else {
                    val errorBody = response.errorBody()?.string()
                    val errorMsg = errorBody?.let {
                        try {
                            val json = org.json.JSONObject(it)
                            json.optString("detail", "Ошибка обновления профиля")
                        } catch (e: Exception) {
                            it
                        }
                    } ?: "Ошибка обновления профиля"
                    onResult(false, errorMsg)
                    Log.e("AuthViewModel", "❌ Ошибка обновления профиля: $errorMsg")
                }
            } catch (e: IOException) {
                onResult(false, "Ошибка сети: ${e.message}")
                Log.e("AuthViewModel", "Network error", e)
            } catch (e: HttpException) {
                onResult(false, "Ошибка сервера: ${e.message}")
                Log.e("AuthViewModel", "HTTP error", e)
            } catch (e: Exception) {
                onResult(false, "Ошибка: ${e.message}")
                Log.e("AuthViewModel", "Unexpected error", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    private fun saveSession(userId: Long, username: String, email: String) {
        prefs.edit().apply {
            putLong(KEY_USER_ID, userId)
            putString(KEY_USERNAME, username)
            putString(KEY_EMAIL, email)
            putBoolean(KEY_IS_LOGGED_IN, true)
            apply()
        }
    }

    fun logout() {
        prefs.edit().clear().apply()
        _currentUser.value = null
        _authState.value = AuthState.NotAuthenticated
        Log.d("AuthViewModel", "✅ Пользователь вышел")
    }

    fun clearValidationErrors() {
        _validationErrors.value = emptyMap()
    }

    sealed class AuthState {
        object Idle : AuthState()
        object Loading : AuthState()
        object NotAuthenticated : AuthState()
        data class Authenticated(val user: User) : AuthState()
        data class Error(val message: String) : AuthState()
    }

    class Factory(private val context: Context) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return AuthViewModel(context) as T
        }
    }
}