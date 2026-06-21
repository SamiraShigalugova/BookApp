package com.samira.bookapp.utils

import java.util.regex.Pattern

object ValidationHelper {

    fun isValidUsername(username: String): ValidationResult {
        return when {
            username.isBlank() -> ValidationResult.Error("Имя пользователя не может быть пустым")
            username.length < 3 -> ValidationResult.Error("Имя пользователя должно содержать минимум 3 символа")
            username.length > 20 -> ValidationResult.Error("Имя пользователя не может превышать 20 символов")
            !username.matches(Regex("^[a-zA-Z0-9_]+$")) -> ValidationResult.Error("Имя пользователя может содержать только буквы, цифры и подчеркивание")
            else -> ValidationResult.Success
        }
    }

    fun isValidEmail(email: String): ValidationResult {

        if (email.isBlank()) {
            return ValidationResult.Success
        }

        val emailPattern = Pattern.compile(
            "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$"
        )
        return if (emailPattern.matcher(email).matches()) {
            ValidationResult.Success
        } else {
            ValidationResult.Error("Введите корректный email (например, user@example.com)")
        }
    }

    fun isValidPassword(password: String): ValidationResult {
        return when {
            password.isBlank() -> ValidationResult.Error("Пароль не может быть пустым")
            password.length < 6 -> ValidationResult.Error("Пароль должен содержать минимум 6 символов")
            else -> ValidationResult.Success
        }
    }
    fun doPasswordsMatch(password: String, confirmPassword: String): ValidationResult {
        return if (password == confirmPassword) {
            ValidationResult.Success
        } else {
            ValidationResult.Error("Пароли не совпадают")
        }
    }

    sealed class ValidationResult {
        object Success : ValidationResult()
        data class Error(val message: String) : ValidationResult()
    }
}