package com.samira.bookapp

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.samira.bookapp.data.AppDatabase
import com.samira.bookapp.ui.theme.AuthScreen
import com.samira.bookapp.viewmodel.AuthViewModel
import com.samira.bookapp.ui.theme.BookAppTheme
import kotlinx.coroutines.MainScope

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val context = LocalContext.current
            val prefs = remember { context.getSharedPreferences("settings", Context.MODE_PRIVATE) }
            var isDarkTheme by remember { mutableStateOf(prefs.getBoolean("dark_theme", false)) }

            BookAppTheme(darkTheme = isDarkTheme) {
                AppContent(
                    isDarkTheme = isDarkTheme,
                    onThemeToggle = { newValue ->
                        isDarkTheme = newValue
                        prefs.edit().putBoolean("dark_theme", newValue).apply()
                    }
                )
            }
        }
    }
}

@Composable
fun AppContent(
    isDarkTheme: Boolean,
    onThemeToggle: (Boolean) -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        val context = LocalContext.current

        val authViewModel: AuthViewModel = viewModel(
            factory = AuthViewModel.Factory(context)
        )

        val authState by authViewModel.authState.collectAsState()
        val currentUser by authViewModel.currentUser.collectAsState()

        LaunchedEffect(currentUser) {
            Log.d("MAIN_ACTIVITY", "👤 Текущий пользователь: ID=${currentUser?.id}, Username=${currentUser?.username}")
            Log.d("MAIN_ACTIVITY", "👤 AuthState: $authState")
        }

        val isAuthenticated = remember(authState) {
            derivedStateOf { authState is AuthViewModel.AuthState.Authenticated }
        }.value

        if (isAuthenticated && currentUser != null) {
            BookApp(
                currentUser = currentUser,
                authViewModel = authViewModel,
                onLogout = { authViewModel.logout() },
                isDarkTheme = isDarkTheme,
                onThemeToggle = onThemeToggle
            )
        } else {
            AuthScreen(
                context = context,
                authViewModel = authViewModel,
                onAuthSuccess = {}
            )
        }
    }
}