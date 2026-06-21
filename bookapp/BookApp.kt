package com.samira.bookapp

import android.util.Log
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.animation.core.*
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.FocusInteraction
import androidx.compose.foundation.interaction.Interaction
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.PressInteraction
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.offset
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.blur
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.samira.bookapp.data.*
import com.samira.bookapp.network.GoogleBook
import com.samira.bookapp.ui.theme.*
import com.samira.bookapp.viewmodel.AuthViewModel
import com.samira.bookapp.viewmodel.BookViewModel
import com.samira.bookapp.viewmodel.ChatViewModel
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.*
import com.samira.bookapp.utils.ValidationHelper
import androidx.compose.animation.core.*
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.material.icons.filled.Schedule

data class BottomNavigationItem(
    val title: String,
    val icon: ImageVector,
    val screen: @Composable () -> Unit
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BookApp(
    database: AppDatabase? = null,
    currentUser: User? = null,
    onLogout: () -> Unit = {},
    onViewRecommendations: () -> Unit = {},
    authViewModel: AuthViewModel? = null,
    isDarkTheme: Boolean,
    onThemeToggle: (Boolean) -> Unit
) {
    var selectedTab by rememberSaveable { mutableStateOf(0) }
    var selectedBook by remember { mutableStateOf<Book?>(null) }
    var selectedOriginalGoogleBook by remember { mutableStateOf<GoogleBook?>(null) }
    val context = LocalContext.current

    LaunchedEffect(currentUser) {
        Log.d("BOOK_APP", "📱 Получен пользователь в BookApp: ID=${currentUser?.id}, Username=${currentUser?.username}")
    }

    val currentUserId = currentUser?.id ?: 1L
    Log.d("BOOK_APP", "🎯 Используется userId: $currentUserId")

    val bookViewModel: BookViewModel = viewModel(
        factory = BookViewModel.Factory(context, currentUserId),
        key = "book_viewmodel_${currentUserId}"
    )

    val navigationItems = listOf(
        BottomNavigationItem(
            title = "Главная",
            icon = Icons.Default.Home,
            screen = {
                HomeScreen(
                    bookViewModel = bookViewModel,
                    onBookClick = { book -> selectedBook = book },
                    onViewRecommendations = { selectedTab = 0 }
                )
            }
        ),
        BottomNavigationItem(
            title = "Поиск",
            icon = Icons.Default.Search,
            screen = {
                SearchScreen(
                    bookViewModel = bookViewModel,
                    onBookClick = { book, originalGoogleBook ->
                        selectedBook = book
                        selectedOriginalGoogleBook = originalGoogleBook
                    },
                    onViewRecommendations = { selectedTab = 0 }
                )
            }
        ),
        BottomNavigationItem(
            title = "Чат",
            icon = Icons.Default.Chat,
            screen = {
                ChatScreen(
                    chatViewModel = viewModel(factory = ChatViewModel.Factory(currentUserId)),
                    bookViewModel = bookViewModel,
                    onBookClick = { book -> selectedBook = book },
                    onAddToLibrary = { googleBook, status ->
                        bookViewModel.addBookToCollectionSync(googleBook.id, status)
                    }
                )
            }
        ),
        BottomNavigationItem(
            title = "Библиотека",
            icon = Icons.Default.Favorite,
            screen = {
                LibraryScreen(
                    bookViewModel = bookViewModel,
                    onBookClick = { book -> selectedBook = book },
                    onViewRecommendations = { selectedTab = 0 }
                )
            }
        ),
        BottomNavigationItem(
            title = "Профиль",
            icon = Icons.Default.Person,
            screen = {
                ProfileScreen(
                    bookViewModel = bookViewModel,
                    authViewModel = authViewModel,
                    onLogout = {
                        authViewModel?.logout()
                        onLogout()
                    },
                    onViewRecommendations = { selectedTab = 0 },
                    currentUser = currentUser,
                    isDarkTheme = isDarkTheme,
                    onThemeToggle = onThemeToggle
                )
            }
        )
    )

    if (selectedBook != null) {
        BookDetailsScreen(
            book = selectedBook!!,
            originalGoogleBook = selectedOriginalGoogleBook,
            bookViewModel = bookViewModel,
            onBackClick = {
                selectedBook = null
                selectedOriginalGoogleBook = null
                bookViewModel.forceUpdate()
            }
        )
    }else {
        Scaffold(
            bottomBar = {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surface,
                    contentColor = MaterialTheme.colorScheme.onSurface
                ) {
                    navigationItems.forEachIndexed { index, item ->
                        NavigationBarItem(
                            icon = {
                                Icon(
                                    item.icon,
                                    contentDescription = item.title,
                                    tint = if (selectedTab == index)
                                        MaterialTheme.colorScheme.primary
                                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                )
                            },
                            label = {
                                Text(
                                    text = item.title,
                                    fontSize = 11.sp,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    color = if (selectedTab == index)
                                        MaterialTheme.colorScheme.primary
                                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                )
                            },
                            selected = selectedTab == index,
                            onClick = { selectedTab = index },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = MaterialTheme.colorScheme.primary,
                                selectedTextColor = MaterialTheme.colorScheme.primary,
                                unselectedIconColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                unselectedTextColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                indicatorColor = Color.Transparent
                            )
                        )
                    }
                }
            }
        ){ paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                navigationItems[selectedTab].screen()
            }
        }
    }
}




@OptIn(ExperimentalFoundationApi::class)
@Composable
fun HomeScreen(
    bookViewModel: BookViewModel?,
    onBookClick: (Book) -> Unit,
    onViewRecommendations: () -> Unit
) {
    if (bookViewModel == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    val aiRecommendedBooks by bookViewModel.aiRecommendedBooks.collectAsState()
    val bestsellerBooks by bookViewModel.bestsellerBooks.collectAsState()
    val unfinishedBooks by bookViewModel.unfinishedBooks.collectAsState()
    val classicBooks by bookViewModel.classicBooks.collectAsState()
    val topRatedBooks by bookViewModel.topRatedBooks.collectAsState()
    val userRatings by bookViewModel.userRatings.collectAsState()
    val bookStatusByGlobalId by bookViewModel.bookStatusByGlobalId.collectAsState()
    val recommendationLoading by bookViewModel.recommendationLoading.collectAsState()
    val isLoadingBestsellers by bookViewModel.isLoadingBestsellers.collectAsState()
    val isLoadingClassic by bookViewModel.isLoadingClassic.collectAsState()
    val isLoadingTopRated by bookViewModel.isLoadingTopRated.collectAsState()


    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp) // ← расстояние между разделами
        ) {

            item {
                SectionHeader(title = "Подобрали для вас", actionText = null, onAction = null)
            }
            item {
                if (recommendationLoading) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                } else if (aiRecommendedBooks.isEmpty()) {
                    EmptySectionMessage("Оцените книги, чтобы получить рекомендации")
                } else {
                    HorizontalBooksRow(
                        books = aiRecommendedBooks,
                        userRatings = userRatings,
                        bookStatusByGlobalId = bookStatusByGlobalId,
                        onBookClick = onBookClick,
                        onAddToCollection = { book, status ->
                            bookViewModel.addBookToCollectionSync(book.globalId, status)
                        },
                        showAddButtons = true,
                        showStatus = true
                    )
                }
            }

            item {
                SectionHeader(title = "Бестселлеры", actionText = null, onAction = null)
            }
            item {
                if (isLoadingBestsellers) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                } else if (bestsellerBooks.isEmpty()) {
                    EmptySectionMessage("Нет данных о бестселлерах")
                } else {
                    HorizontalBooksRow(
                        books = bestsellerBooks,
                        userRatings = userRatings,
                        bookStatusByGlobalId = bookStatusByGlobalId,
                        onBookClick = onBookClick,
                        onAddToCollection = { book, status ->
                            bookViewModel.addBookToCollectionSync(book.globalId, status)
                        },
                        showAddButtons = true,
                        showStatus = true
                    )
                }
            }

            if (unfinishedBooks.isNotEmpty()) {
                item {
                    SectionHeader(title = "Книги ждут вас", actionText = null, onAction = null)
                }
                item {
                    HorizontalBooksRow(
                        books = unfinishedBooks,
                        userRatings = userRatings,
                        bookStatusByGlobalId = bookStatusByGlobalId,
                        onBookClick = onBookClick,
                        onAddToCollection = null,
                        showAddButtons = false,
                        showStatus = true
                    )
                }
            }

            item {
                SectionHeader(title = "Топ рейтинга", actionText = null, onAction = null)
            }
            item {
                if (isLoadingTopRated) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                } else if (topRatedBooks.isEmpty()) {
                    EmptySectionMessage("Нет книг с высоким рейтингом")
                } else {
                    HorizontalBooksRow(
                        books = topRatedBooks,
                        userRatings = userRatings,
                        bookStatusByGlobalId = bookStatusByGlobalId,
                        onBookClick = onBookClick,
                        onAddToCollection = { book, status ->
                            bookViewModel.addBookToCollectionSync(book.globalId, status)
                        },
                        showAddButtons = true,
                        showStatus = true
                    )
                }
            }


            item {
                SectionHeader(title = "Классика", actionText = null, onAction = null)
            }
            item {
                if (isLoadingClassic) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator()
                    }
                } else if (classicBooks.isEmpty()) {
                    EmptySectionMessage("Нет книг в этом разделе")
                } else {
                    HorizontalBooksRow(
                        books = classicBooks,
                        userRatings = userRatings,
                        bookStatusByGlobalId = bookStatusByGlobalId,
                        onBookClick = onBookClick,
                        onAddToCollection = { book, status ->
                            bookViewModel.addBookToCollectionSync(book.globalId, status)
                        },
                        showAddButtons = true,
                        showStatus = true
                    )
                }
            }


        }
    }
}


@Composable
fun SectionHeader(
    title: String,
    actionText: String? = null,
    onAction: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground
        )
        if (actionText != null && onAction != null) {
            TextButton(onClick = onAction) {
                Text(
                    text = actionText,
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
fun HorizontalBooksRow(
    books: List<Book>,
    userRatings: Map<String, Float>,
    bookStatusByGlobalId: Map<String, BookStatus>,
    onBookClick: (Book) -> Unit,
    onAddToCollection: ((Book, BookStatus) -> Unit)? = null,
    showAddButtons: Boolean = true,
    showStatus: Boolean = true
) {
    LazyRow(
        modifier = Modifier.fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(books, key = { it.globalId }) { book ->
            HomeCardCompact(
                book = book,
                status = if (showStatus) bookStatusByGlobalId[book.globalId] else null,
                userRating = userRatings[book.globalId] ?: 0f,
                onBookClick = { onBookClick(book) },
                onAddToCollection = if (showAddButtons && onAddToCollection != null) {
                    { status -> onAddToCollection(book, status) }
                } else null
            )
        }
    }
}

@Composable
fun HomeCardCompact(
    book: Book,
    status: BookStatus?,
    userRating: Float,
    onBookClick: () -> Unit,
    onAddToCollection: ((BookStatus) -> Unit)? = null
) {
    Card(
        modifier = Modifier
            .width(150.dp)
            .height(260.dp)
            .clickable(onClick = onBookClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(2.dp)
    ) {
        Column {

            Box(
                modifier = Modifier
                    .height(180.dp)
                    .fillMaxWidth()
            ) {
                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = book.coverUrl,
                        contentDescription = null,
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(topStart = 12.dp, topEnd = 12.dp)),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.surfaceVariant),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("📚", fontSize = 40.sp)
                    }
                }

                if (status != null) {
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(4.dp)
                            .background(
                                when (status) {
                                    BookStatus.READING -> Color(0xFF34C759)
                                    BookStatus.WANT_TO_READ -> Color(0xFF007AFF)
                                    BookStatus.FINISHED -> Color(0xFFAF52DE)
                                    BookStatus.DROPPED -> Color(0xFFFF3B30)
                                },
                                RoundedCornerShape(4.dp)
                            )
                    ) {
                        Text(
                            text = when (status) {
                                BookStatus.READING -> "📖"
                                BookStatus.WANT_TO_READ -> "🔖"
                                BookStatus.FINISHED -> "✅"
                                BookStatus.DROPPED -> "⏸️"
                            },
                            fontSize = 10.sp,
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
            ) {
                Text(
                    text = book.title,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = book.author,
                    fontSize = 11.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
                if (userRating > 0) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Filled.Star,
                            contentDescription = null,
                            modifier = Modifier.size(10.dp),
                            tint = Color(0xFFFFD700)
                        )
                        Text(
                            text = " ${userRating.toInt()}",
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
            }

            if (onAddToCollection != null && status == null) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Button(
                        onClick = { onAddToCollection(BookStatus.WANT_TO_READ) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color.Transparent,
                            contentColor = MaterialTheme.colorScheme.primary
                        ),
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.primary)
                    ) {
                        Text("Хочу", fontSize = 10.sp)
                    }
                    Button(
                        onClick = { onAddToCollection(BookStatus.READING) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color.Transparent,
                            contentColor = MaterialTheme.colorScheme.secondary
                        ),
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.secondary)
                    ) {
                        Text("Читаю", fontSize = 10.sp)
                    }
                }
            }
        }
    }
}

@Composable
fun EmptySectionMessage(message: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp)
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            fontSize = 14.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun HomeGlassCard(
    book: Book,
    status: BookStatus?,
    userRating: Float,
    onBookClick: () -> Unit,
    onAddToCollection: (BookStatus) -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(380.dp)
            .scale(if (isPressed) 0.98f else 1f)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onBookClick
            ),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF1E1E1E)
        ),
        border = BorderStroke(0.5.dp, Color.White.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {

            Box(
                modifier = Modifier
                    .height(180.dp)
                    .fillMaxWidth()
            ) {
                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = ImageRequest.Builder(LocalContext.current)
                            .data(book.coverUrl)
                            .crossfade(true)
                            .build(),
                        contentDescription = "Обложка ${book.title}",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                        contentScale = ContentScale.Crop,
                        onError = {

                        }
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Color(0xFF2C2C2E),
                                RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("📚", fontSize = 48.sp, color = Color.White.copy(alpha = 0.3f))
                    }
                }

                status?.let {
                    Box(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopStart)
                            .background(
                                when (it) {
                                    BookStatus.READING -> Color(0xFF34C759)
                                    BookStatus.WANT_TO_READ -> Color(0xFF007AFF)
                                    BookStatus.FINISHED -> Color(0xFFAF52DE)
                                    BookStatus.DROPPED -> Color(0xFFFF3B30)
                                },
                                RoundedCornerShape(8.dp)
                            )
                    ) {
                        Text(
                            text = when (it) {
                                BookStatus.READING -> "📖 Читаю"
                                BookStatus.WANT_TO_READ -> "🔖 Хочу"
                                BookStatus.FINISHED -> "✅ Прочитано"
                                BookStatus.DROPPED -> "⏸️ Брошено"
                            },
                            fontSize = 11.sp,
                            color = Color.White,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }

                if (book.averageRating > 0) {
                    Box(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopEnd)
                            .background(
                                Color(0xFF2C2C2E).copy(alpha = 0.8f),
                                RoundedCornerShape(8.dp)
                            )
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = null,
                                modifier = Modifier.size(12.dp),
                                tint = Color(0xFFFFD60A)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "%.1f".format(book.averageRating),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = Color.White
                            )
                        }
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(12.dp)
            ) {
                Text(
                    text = book.genre,
                    fontSize = 11.sp,
                    color = Color(0xFFFFFFFF),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = book.title,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 18.sp
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = book.author,
                    fontSize = 12.sp,
                    color = Color.White.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                if (userRating > 0) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "Ваша:",
                            fontSize = 10.sp,
                            color = Color.White.copy(alpha = 0.5f)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        ReadOnlyRatingBar(
                            rating = userRating,
                            modifier = Modifier
                                .height(8.dp)
                                .width(45.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "${userRating.toInt()}/5",
                            fontSize = 10.sp,
                            color = Color(0xFFFFFFFF)
                        )
                    }
                }

                Spacer(modifier = Modifier.weight(1f))

                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    TransparentButton(
                        text = "Хочу прочитать",
                        onClick = { onAddToCollection(BookStatus.WANT_TO_READ) },
                        textColor = Color(0xFFFFFFFF)
                    )

                    TransparentButton(
                        text = "Читаю",
                        onClick = { onAddToCollection(BookStatus.READING) },
                        textColor = Color(0xFFFFFFFF)
                    )
                }
            }
        }
    }
}
@Composable
fun ThemedTransparentButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .fillMaxWidth()
            .height(36.dp)
            .scale(if (isPressed) 0.97f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = MaterialTheme.colorScheme.onSurface
        ),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
fun TransparentButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    textColor: Color = Color.White
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .fillMaxWidth()
            .height(36.dp)
            .scale(if (isPressed) 0.97f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = textColor
        ),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, textColor.copy(alpha = 0.5f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
fun GlassButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .scale(if (isPressed) 0.95f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = Color.White
        ),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.3f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun EmptyHomeState(
    onViewRecommendations: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                    .border(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            Text(
                text = "У вас пока нет оценок",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )

            Text(
                text = "Оцените несколько книг, чтобы мы могли подобрать рекомендации",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )

            ThemedGlassButton(
                text = "Перейти к поиску",
                onClick = onViewRecommendations,
                modifier = Modifier.width(200.dp)
            )
        }
    }
}

@Composable
fun ThemedGlassButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .scale(if (isPressed) 0.95f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = MaterialTheme.colorScheme.onSurface
        ),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}


@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun SearchScreen(
    bookViewModel: BookViewModel,
    onBookClick: (Book, GoogleBook?) -> Unit,
    onViewRecommendations: () -> Unit = {}
) {
    var searchQuery by remember { mutableStateOf("") }
    var isSearchFocused by remember { mutableStateOf(false) }
    var selectedApiBook by remember { mutableStateOf<GoogleBook?>(null) }
    var showBookDetails by remember { mutableStateOf(false) }

    var isGenreSearchActive by remember { mutableStateOf(false) }

    val apiResults by bookViewModel.searchResults.collectAsState()
    val isSearching by bookViewModel.isSearching.collectAsState()
    val openLibraryGenres by bookViewModel.openLibraryGenres.collectAsState()
    val selectedGenre by bookViewModel.selectedGenre.collectAsState()
    val bookStatusByGlobalId by bookViewModel.bookStatusByGlobalId.collectAsState()

    val currentGenre = selectedGenre

    val filteredResults = remember(apiResults, currentGenre) {
        if (currentGenre != null && !isGenreSearchActive) {
            apiResults.filter { googleBook ->
                val categories = googleBook.volumeInfo.categories ?: emptyList()
                categories.any { category ->
                    category.equals(currentGenre, ignoreCase = true) ||
                            bookViewModel.translateGenreToRussian(category).equals(currentGenre, ignoreCase = true)
                }
            }
        } else {
            apiResults
        }
    }

    LaunchedEffect(searchQuery, currentGenre) {
        // Если выбран жанр и поисковый запрос пуст, ищем по жанру
        if (currentGenre != null && searchQuery.trim().isEmpty()) {
            if (!isGenreSearchActive) {
                isGenreSearchActive = true
                bookViewModel.searchBooksByGenreLocally(currentGenre)
            }
            return@LaunchedEffect
        }

        if (currentGenre != null && searchQuery.trim().isNotEmpty()) {
            isGenreSearchActive = false
            bookViewModel.quickSearchBooks("${currentGenre} ${searchQuery.trim()}")
            return@LaunchedEffect
        }

        if (currentGenre == null && searchQuery.trim().isNotEmpty()) {
            isGenreSearchActive = false
            val trimmedQuery = searchQuery.trim()
            if (trimmedQuery.length >= 2) {
                delay(500)
                if (trimmedQuery == searchQuery.trim() && currentGenre == null) {
                    bookViewModel.quickSearchBooks(trimmedQuery)
                }
            }
            return@LaunchedEffect
        }

        if (currentGenre == null && searchQuery.trim().isEmpty()) {
            isGenreSearchActive = false
            bookViewModel.clearSearchResults()
        }
    }

    if (showBookDetails && selectedApiBook != null) {
        OpenLibraryBookDetailsScreen(
            apiBook = selectedApiBook!!,
            onBackClick = {
                showBookDetails = false
                selectedApiBook = null
            },
            onAddToCollection = { status ->
                bookViewModel.addBookToCollectionSync(selectedApiBook!!.id, status)
                showBookDetails = false
                selectedApiBook = null
            }
        )
    } else {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            Column(
                modifier = Modifier.fillMaxSize()
            ) {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Поиск книг",
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground,
                    modifier = Modifier.padding(horizontal = 20.dp)
                )

                SearchBar(
                    searchQuery = searchQuery,
                    onSearchQueryChange = {
                        searchQuery = it
                        isGenreSearchActive = false
                    },
                    isFocused = isSearchFocused,
                    onFocusChange = { isSearchFocused = it },
                    selectedGenre = currentGenre,
                    onClearGenre = {
                        bookViewModel.selectGenre(null)
                        searchQuery = ""
                        isGenreSearchActive = false
                    }
                )

                if (openLibraryGenres.isNotEmpty()) {
                    GenreSection(
                        genres = openLibraryGenres,
                        selectedGenre = currentGenre,
                        onGenreSelected = { genre ->
                            val newGenre = if (currentGenre == genre) null else genre
                            bookViewModel.selectGenre(newGenre)
                            if (newGenre != null) {
                                searchQuery = ""
                                isGenreSearchActive = true
                            } else {
                                isGenreSearchActive = false
                                if (searchQuery.trim().isNotEmpty()) {
                                    bookViewModel.quickSearchBooks(searchQuery.trim())
                                }
                            }
                        }
                    )
                }

                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(1f)
                ) {
                    when {
                        isSearching -> {
                            ThemedLoadingIndicator(message = "Ищем книги...")
                        }
                        filteredResults.isNotEmpty() -> {
                            LazyVerticalGrid(
                                columns = GridCells.Fixed(2),
                                modifier = Modifier.fillMaxSize(),
                                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                items(filteredResults, key = { it.id }) { googleBook ->
                                    val book = bookViewModel.googleBookToLocalBook(googleBook)
                                    val status = bookStatusByGlobalId[googleBook.id]
                                    ThemedSearchBookCard(
                                        book = book,
                                        status = status,
                                        apiResults = filteredResults,
                                        bookViewModel = bookViewModel,
                                        onBookClick = onBookClick
                                    )
                                }
                            }
                        }
                        searchQuery.trim().isNotEmpty() && filteredResults.isEmpty() && !isSearching && currentGenre == null -> {
                            ThemedEmptyState(
                                query = searchQuery,
                                onViewRecommendations = onViewRecommendations
                            )
                        }
                        currentGenre != null && filteredResults.isEmpty() && !isSearching -> {
                            ThemedEmptyState(
                                query = currentGenre,
                                onViewRecommendations = onViewRecommendations
                            )
                        }
                        else -> {
                            ThemedInitialState(
                                onViewRecommendations = onViewRecommendations
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SearchBar(
    searchQuery: String,
    onSearchQueryChange: (String) -> Unit,
    isFocused: Boolean,
    onFocusChange: (Boolean) -> Unit,
    selectedGenre: String?,
    onClearGenre: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp)
    ) {
        OutlinedTextField(
            value = searchQuery,
            onValueChange = onSearchQueryChange,
            placeholder = {
                Text(
                    text = "Название, автор или жанр...",
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                )
            },
            leadingIcon = {
                Icon(
                    Icons.Default.Search,
                    contentDescription = "Поиск",
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            },
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(24.dp)),
            trailingIcon = {
                if (searchQuery.isNotEmpty()) {
                    IconButton(onClick = { onSearchQueryChange("") }) {
                        Icon(
                            Icons.Default.Clear,
                            contentDescription = "Очистить",
                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
            },
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                focusedTextColor = MaterialTheme.colorScheme.onSurface,
                unfocusedTextColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f),
                cursorColor = MaterialTheme.colorScheme.primary
            ),
            shape = RoundedCornerShape(24.dp),
            singleLine = true,
            interactionSource = remember { MutableInteractionSource() }
                .also { source ->
                    LaunchedEffect(source) {
                        source.interactions.collect { interaction ->
                            if (interaction is FocusInteraction.Focus) {
                                onFocusChange(true)
                            } else if (interaction is FocusInteraction.Unfocus) {
                                onFocusChange(false)
                            }
                        }
                    }
                }
        )
    }
}

@Composable
fun GenreSection(
    genres: List<String>,
    selectedGenre: String?,
    onGenreSelected: (String) -> Unit
) {
    Column(
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
    ) {
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(genres.take(20)) { genre ->
                ThemedGenreChip(
                    genre = genre,
                    isSelected = selectedGenre == genre,
                    onClick = { onGenreSelected(genre) }
                )
            }
        }
    }
}

@Composable
fun ThemedGenreChip(
    genre: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val backgroundColor = if (isSelected)
        MaterialTheme.colorScheme.primary
    else
        MaterialTheme.colorScheme.surface

    val textColor = if (isSelected)
        MaterialTheme.colorScheme.onPrimary
    else
        MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)

    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(20.dp),
        color = backgroundColor,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = if (isSelected) 1f else 0.3f)),
        tonalElevation = 0.dp
    ) {
        Text(
            text = genre,
            fontSize = 13.sp,
            color = textColor,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}

@Composable
fun ThemedSearchBookCard(
    book: Book,
    status: BookStatus?,
    apiResults: List<GoogleBook>,
    bookViewModel: BookViewModel,
    onBookClick: (Book, GoogleBook?) -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(380.dp)
            .scale(if (isPressed) 0.98f else 1f)
            .clickable(
                interactionSource = interactionSource,
                indication = null
            ) {
                val original = apiResults.find { it.id == book.globalId }
                if (original != null) {
                    val tempBook = book.copy(id = 0)
                    onBookClick(tempBook, original)
                } else {
                    onBookClick(book, null)
                }
            },
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .height(180.dp)
                    .fillMaxWidth()
            ) {
                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = book.coverUrl,
                        contentDescription = "Обложка ${book.title}",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant,
                                RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("📚", fontSize = 40.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f))
                    }
                }

                status?.let {
                    Box(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopStart)
                            .background(
                                when (it) {
                                    BookStatus.READING -> Color(0xFF34C759)
                                    BookStatus.WANT_TO_READ -> Color(0xFF007AFF)
                                    BookStatus.FINISHED -> Color(0xFFAF52DE)
                                    BookStatus.DROPPED -> Color(0xFFFF3B30)
                                },
                                RoundedCornerShape(8.dp)
                            )
                    ) {
                        Text(
                            text = when (it) {
                                BookStatus.READING -> "📖 Читаю"
                                BookStatus.WANT_TO_READ -> "🔖 Хочу"
                                BookStatus.FINISHED -> "✅ Прочитано"
                                BookStatus.DROPPED -> "⏸️ Брошено"
                            },
                            fontSize = 11.sp,
                            color = Color.White,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }

                if (book.averageRating > 0) {
                    Box(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopEnd)
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f),
                                RoundedCornerShape(8.dp)
                            )
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = null,
                                modifier = Modifier.size(12.dp),
                                tint = Color(0xFFFFD60A)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "%.1f".format(book.averageRating),
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(12.dp)
            ) {
                Text(
                    text = book.genre,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.primary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = book.title,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 18.sp
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = book.author,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.weight(1f))

                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    ThemedTransparentButton(
                        text = "Хочу прочитать",
                        onClick = {
                            bookViewModel.addBookToCollectionSync(book.globalId, BookStatus.WANT_TO_READ)
                            bookViewModel.forceUpdate()
                        }
                    )

                    ThemedTransparentButton(
                        text = "Читаю",
                        onClick = {
                            bookViewModel.addBookToCollectionSync(book.globalId, BookStatus.READING)
                            bookViewModel.forceUpdate()
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun ThemedLoadingIndicator(message: String) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(48.dp),
                strokeWidth = 3.dp,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = message,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun ThemedEmptyState(
    query: String,
    onViewRecommendations: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                    .border(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.SearchOff,
                    contentDescription = null,
                    modifier = Modifier.size(40.dp),
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            Text(
                text = "Ничего не найдено",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )

            Text(
                text = "По запросу \"$query\" ничего нет. Попробуйте изменить запрос.",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )

            ThemedTransparentButton(
                text = "Посмотреть рекомендации",
                onClick = onViewRecommendations,
                modifier = Modifier.width(200.dp)
            )
        }
    }
}

@Composable
fun ThemedInitialState(
    onViewRecommendations: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                    .border(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Search,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            Text(
                text = "Начните поиск",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )

            Text(
                text = "Введите название книги, автора или выберите жанр",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )

            ThemedTransparentButton(
                text = "Посмотреть рекомендации",
                onClick = onViewRecommendations,
                modifier = Modifier.width(200.dp)
            )
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OpenLibraryBookDetailsScreen(
    apiBook: GoogleBook,
    onBackClick: () -> Unit,
    onAddToCollection: (BookStatus) -> Unit,
    modifier: Modifier = Modifier
) {
    val volumeInfo = apiBook.volumeInfo

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Информация о книге") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Назад")
                    }
                }
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = {
                    onAddToCollection(BookStatus.WANT_TO_READ)
                },
                icon = { Icon(Icons.Default.Add, contentDescription = "Добавить") },
                text = { Text("Добавить в библиотеку") }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
        ) {
            if (volumeInfo.imageLinks?.thumbnail != null) {
                AsyncImage(
                    model = volumeInfo.imageLinks.thumbnail,
                    contentDescription = "Обложка ${volumeInfo.title}",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                        .padding(16.dp),
                    contentScale = ContentScale.Fit
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .padding(16.dp)
                        .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center
                ) {
                    Text("📚", style = MaterialTheme.typography.displayMedium)
                }
            }

            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = volumeInfo.title,
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                if (!volumeInfo.authors.isNullOrEmpty()) {
                    Text(
                        text = "Автор: ${volumeInfo.authors.joinToString(", ")}",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                }

                volumeInfo.publishedDate?.let { publishedDate ->
                    Text(
                        text = "Год издания: ${publishedDate.take(4)}",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                }

                if (!volumeInfo.categories.isNullOrEmpty()) {
                    Text(
                        text = "Жанры: ${volumeInfo.categories.take(5).joinToString(", ")}",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                }

                volumeInfo.description?.let { description ->
                    Text(
                        text = "Описание: $description",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 4.dp),
                        maxLines = 5,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                volumeInfo.averageRating?.let { rating ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(bottom = 4.dp)
                    ) {
                        Text("Рейтинг: ", style = MaterialTheme.typography.bodyMedium)
                        ReadOnlyRatingBar(
                            rating = rating,
                            modifier = Modifier.padding(horizontal = 4.dp)
                        )
                        Text(
                            text = " ${"%.1f".format(rating)}/5",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        volumeInfo.ratingsCount?.let { count ->
                            Text(
                                text = " ($count оценок)",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        "Добавить в:",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    Button(
                        onClick = { onAddToCollection(BookStatus.WANT_TO_READ) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Icon(Icons.Default.Bookmark, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Хочу прочитать")
                    }

                    Button(
                        onClick = { onAddToCollection(BookStatus.READING) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.secondary
                        )
                    ) {
                        Icon(Icons.Default.MenuBook, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Читаю сейчас")
                    }

                    Button(
                        onClick = { onAddToCollection(BookStatus.FINISHED) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.tertiary
                        )
                    ) {
                        Icon(Icons.Default.Done, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Прочитано")
                    }
                }
            }
        }
    }
}

@Composable
fun InitialSearchState(onViewRecommendations: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.Search,
            contentDescription = "Поиск",
            modifier = Modifier.size(120.dp),
            tint = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )

        Text(
            text = "Найдите книги",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 24.dp, bottom = 8.dp),
            textAlign = TextAlign.Center
        )

        Text(
            text = "Введите название книги или автора в поле поиска",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        Button(
            onClick = onViewRecommendations,
            modifier = Modifier.padding(top = 8.dp)
        ) {

            Spacer(modifier = Modifier.width(8.dp))
            Text("Посмотреть рекомендации")
        }
    }
}
@Composable
fun EmptyResultsState(
    query: String,
    onViewRecommendations: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.SearchOff,
            contentDescription = "Нет результатов",
            modifier = Modifier.size(120.dp),
            tint = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )

        Text(
            text = "По запросу \"$query\" ничего не найдено",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 24.dp, bottom = 8.dp),
            textAlign = TextAlign.Center
        )

        Text(
            text = "Попробуйте изменить запрос или выберите другой жанр",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        Button(
            onClick = onViewRecommendations,
            modifier = Modifier.padding(top = 8.dp)
        ) {
            Icon(Icons.Default.Recommend, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Посмотреть рекомендации")
        }
    }
}

@Composable
fun LoadingContent(message: String) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(48.dp),
            strokeWidth = 3.dp,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 16.dp)
        )
    }
}



@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun LibraryScreen(
    bookViewModel: BookViewModel? = null,
    onBookClick: (Book) -> Unit = {},
    onViewRecommendations: () -> Unit = {},
    userId: Long = 1L
) {
    if (bookViewModel == null) {
        ThemedLoadingScreen()
        return
    }

    val readingBooks by bookViewModel.readingBooks.collectAsState()
    val wantToReadBooks by bookViewModel.wantToReadBooks.collectAsState()
    val finishedBooks by bookViewModel.finishedBooks.collectAsState()
    val droppedBooks by bookViewModel.droppedBooks.collectAsState()
    val userRatings by bookViewModel.userRatings.collectAsState()

    var selectedCategory by remember { mutableStateOf(LibraryCategory.READING) }

    val categories = LibraryCategory.entries

    val totalBooks = readingBooks.size + wantToReadBooks.size + finishedBooks.size + droppedBooks.size
    val totalRead = finishedBooks.size

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Моя библиотека",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.padding(horizontal = 20.dp)
            )

            Text(
                text = "$totalBooks книг • $totalRead прочитано",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                modifier = Modifier
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 16.dp)
            )

            ThemedLibraryCategoriesRow(
                categories = categories,
                selectedCategory = selectedCategory,
                onCategorySelected = { selectedCategory = it },
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
            )

            when (selectedCategory) {
                LibraryCategory.READING -> ThemedLibraryBooksContent(
                    books = readingBooks,
                    userRatings = userRatings,
                    category = selectedCategory,
                    onBookClick = onBookClick,
                    emptyStateMessage = "Вы ещё не начали читать",
                    emptyStateSubMessage = "Начните читать первую книгу",
                    onViewRecommendations = onViewRecommendations
                )
                LibraryCategory.WANT_TO_READ -> ThemedLibraryBooksContent(
                    books = wantToReadBooks,
                    userRatings = userRatings,
                    category = selectedCategory,
                    onBookClick = onBookClick,
                    emptyStateMessage = "В планах пока пусто",
                    emptyStateSubMessage = "Добавьте книги, которые хотите прочитать",
                    onViewRecommendations = onViewRecommendations
                )
                LibraryCategory.FINISHED -> ThemedLibraryBooksContent(
                    books = finishedBooks,
                    userRatings = userRatings,
                    category = selectedCategory,
                    onBookClick = onBookClick,
                    emptyStateMessage = "Нет прочитанных книг",
                    emptyStateSubMessage = "Завершённые книги появятся здесь",
                    onViewRecommendations = onViewRecommendations
                )
                LibraryCategory.DROPPED -> ThemedLibraryBooksContent(
                    books = droppedBooks,
                    userRatings = userRatings,
                    category = selectedCategory,
                    onBookClick = onBookClick,
                    emptyStateMessage = "Нет отложенных книг",
                    emptyStateSubMessage = "Книги, которые вы не дочитали, будут здесь",
                    onViewRecommendations = onViewRecommendations
                )
            }
        }
    }
}

@Composable
fun ThemedLibraryCategoriesRow(
    categories: List<LibraryCategory>,
    selectedCategory: LibraryCategory,
    onCategorySelected: (LibraryCategory) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyRow(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(categories) { category ->
            ThemedLibraryCategoryChip(
                category = category,
                isSelected = category == selectedCategory,
                onClick = { onCategorySelected(category) }
            )
        }
    }
}

@Composable
fun ThemedLibraryCategoryChip(
    category: LibraryCategory,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val backgroundColor = if (isSelected)
        MaterialTheme.colorScheme.primary
    else
        MaterialTheme.colorScheme.surface

    val contentColor = if (isSelected)
        MaterialTheme.colorScheme.onPrimary
    else
        MaterialTheme.colorScheme.onSurface

    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(20.dp),
        color = backgroundColor,
        border = BorderStroke(0.5.dp, if (isSelected) Color.Transparent else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.2f)),
        tonalElevation = 0.dp
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Icon(
                imageVector = category.icon,
                contentDescription = null,
                modifier = Modifier.size(16.dp),
                tint = contentColor
            )
            Text(
                text = category.label,
                fontSize = 13.sp,
                color = contentColor,
                fontWeight = if (isSelected) FontWeight.Medium else FontWeight.Normal
            )
        }
    }
}

@Composable
fun ThemedLibraryBooksContent(
    books: List<Book>,
    userRatings: Map<String, Float>,
    category: LibraryCategory,
    onBookClick: (Book) -> Unit,
    emptyStateMessage: String,
    emptyStateSubMessage: String,
    onViewRecommendations: () -> Unit
) {
    if (books.isEmpty()) {
        ThemedLibraryEmptyState(
            category = category,
            message = emptyStateMessage,
            subMessage = emptyStateSubMessage,
            onViewRecommendations = onViewRecommendations
        )
    } else {
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            items(books, key = { it.globalId  }) { book ->
                ThemedLibraryBookCard(
                    book = book,
                    userRating = userRatings[book.globalId] ?: 0f,
                    onBookClick = { onBookClick(book) }
                )
            }
        }
    }
}

@Composable
fun ThemedLibraryBookCard(
    book: Book,
    userRating: Float,
    onBookClick: () -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp)
            .scale(if (isPressed) 0.98f else 1f)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onBookClick
            ),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .height(160.dp)
                    .fillMaxWidth()
                    .background(
                        MaterialTheme.colorScheme.surfaceVariant,
                        RoundedCornerShape(
                            topStart = 16.dp,
                            topEnd = 16.dp,
                            bottomStart = 0.dp,
                            bottomEnd = 0.dp
                        )
                    )
            ) {
                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = book.coverUrl,
                        contentDescription = "Обложка ${book.title}",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = "📚",
                                fontSize = 40.sp,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                            )
                            Text(
                                text = book.title.take(1).uppercase(),
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                            )
                        }
                    }
                }

                if (book.averageRating > 0) {
                    Surface(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopEnd),
                        shape = CircleShape,
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.9f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = "Рейтинг",
                                modifier = Modifier.size(12.dp),
                                tint = MaterialTheme.colorScheme.onPrimary
                            )
                            Spacer(modifier = Modifier.width(2.dp))
                            Text(
                                text = "%.1f".format(book.averageRating),
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                if (userRating > 0) {
                    Surface(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.BottomStart),
                        shape = RoundedCornerShape(8.dp),
                        color = Color(0xFF34C759).copy(alpha = 0.9f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = "Ваша оценка",
                                modifier = Modifier.size(10.dp),
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(2.dp))
                            Text(
                                text = "${userRating.toInt()}",
                                fontSize = 10.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(12.dp)
            ) {
                Text(
                    text = book.genre,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.primary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = book.title,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 16.sp
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = book.author,
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun ThemedLibraryEmptyState(
    category: LibraryCategory,
    message: String,
    subMessage: String,
    onViewRecommendations: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                    .border(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = category.icon,
                    contentDescription = null,
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                )
            }

            Text(
                text = message,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )

            Text(
                text = subMessage,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )

            ThemedLibraryTransparentButton(
                text = "Перейти к поиску",
                onClick = onViewRecommendations
            )
        }
    }
}

@Composable
fun ThemedLibraryTransparentButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .width(200.dp)
            .height(40.dp)
            .scale(if (isPressed) 0.97f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = MaterialTheme.colorScheme.primary
        ),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun ThemedLoadingScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(48.dp),
                strokeWidth = 3.dp,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Загрузка библиотеки...",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }
    }
}

enum class LibraryCategory(val icon: ImageVector, val label: String) {
    READING(Icons.Filled.PlayArrow, "Читаю"),
    WANT_TO_READ(Icons.Filled.Bookmark, "В планах"),
    FINISHED(Icons.Filled.Done, "Прочитано"),
    DROPPED(Icons.Filled.Close, "Брошено")
}

@Composable
fun LibraryBookCard(
    book: Book,
    userRating: Float,
    onBookClick: () -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp)
            .scale(if (isPressed) 0.98f else 1f)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onBookClick
            ),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF1E1E1E)
        ),
        border = BorderStroke(0.5.dp, Color.White.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .height(160.dp)
                    .fillMaxWidth()
                    .background(
                        Color(0xFF2C2C2E),
                        RoundedCornerShape(
                            topStart = 16.dp,
                            topEnd = 16.dp,
                            bottomStart = 0.dp,
                            bottomEnd = 0.dp
                        )
                    )
            ) {
                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = book.coverUrl,
                        contentDescription = "Обложка ${book.title}",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = "📚",
                                fontSize = 40.sp,
                                color = Color.White.copy(alpha = 0.3f)
                            )
                            Text(
                                text = book.title.take(1).uppercase(),
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White.copy(alpha = 0.5f)
                            )
                        }
                    }
                }

                if (book.averageRating > 0) {
                    Surface(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.TopEnd),
                        shape = CircleShape,
                        color = Color(0xFF007AFF).copy(alpha = 0.9f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = "Рейтинг",
                                modifier = Modifier.size(12.dp),
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(2.dp))
                            Text(
                                text = "%.1f".format(book.averageRating),
                                fontSize = 11.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                if (userRating > 0) {
                    Surface(
                        modifier = Modifier
                            .padding(8.dp)
                            .align(Alignment.BottomStart),
                        shape = RoundedCornerShape(8.dp),
                        color = Color(0xFF34C759).copy(alpha = 0.9f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Filled.Star,
                                contentDescription = "Ваша оценка",
                                modifier = Modifier.size(10.dp),
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(2.dp))
                            Text(
                                text = "${userRating.toInt()}",
                                fontSize = 10.sp,
                                color = Color.White,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(12.dp)
            ) {
                Text(
                    text = book.genre,
                    fontSize = 11.sp,
                    color = Color(0xFF60A5FA),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = book.title,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 16.sp
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = book.author,
                    fontSize = 11.sp,
                    color = Color.White.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun LibraryTransparentButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Button(
        onClick = onClick,
        modifier = modifier
            .width(200.dp)
            .height(40.dp)
            .scale(if (isPressed) 0.97f else 1f),
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.Transparent,
            contentColor = Color.Black
        ),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, Color.Black.copy(alpha = 0.3f)),
        elevation = ButtonDefaults.buttonElevation(0.dp),
        interactionSource = interactionSource
    ) {
        Text(
            text = text,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun LoadingScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(48.dp),
                strokeWidth = 3.dp,
                color = Color(0xFF1E1E1E)
            )
            Text(
                text = "Загрузка библиотеки...",
                fontSize = 14.sp,
                color = Color.Black.copy(alpha = 0.6f)
            )
        }
    }
}
@Composable
fun ProfileHeader(
    username: String,
    registrationDate: String,
    isDarkTheme: Boolean,
    onThemeToggle: (Boolean) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .background(Color(0xFF1E1E1E))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Surface(
                    modifier = Modifier.size(90.dp),
                    shape = CircleShape,
                    color = Color(0xFF2C2C2E),
                    border = BorderStroke(2.dp, Color.White.copy(alpha = 0.3f))
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(
                            text = username.firstOrNull()?.uppercase() ?: "👤",
                            fontSize = 36.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = username,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )

                Text(
                    text = "Читатель с $registrationDate",
                    fontSize = 14.sp,
                    color = Color.White.copy(alpha = 0.7f)
                )
            }

            Surface(
                modifier = Modifier
                    .padding(16.dp)
                    .align(Alignment.TopEnd),
                shape = CircleShape,
                color = Color.White.copy(alpha = 0.2f)
            ) {
                IconButton(
                    onClick = { onThemeToggle(!isDarkTheme) }
                ) {
                    Icon(
                        imageVector = if (isDarkTheme) Icons.Default.DarkMode else Icons.Default.LightMode,
                        contentDescription = "Переключить тему",
                        tint = Color.White
                    )
                }
            }
        }
    }
}

@Composable
fun ProfileStats(
    totalBooks: Int,
    totalRead: Int,
    totalReading: Int,
    totalRatings: Int
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "Статистика",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Black,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            ProfileStatCard(
                count = totalBooks,
                label = "Всего",
                icon = Icons.Default.Book,
                color = Color(0xFF007AFF)
            )
            ProfileStatCard(
                count = totalRead,
                label = "Прочитано",
                icon = Icons.Default.Done,
                color = Color(0xFF34C759)
            )
            ProfileStatCard(
                count = totalReading,
                label = "Читаю",
                icon = Icons.Default.MenuBook,
                color = Color(0xFFFF9500)
            )
            ProfileStatCard(
                count = totalRatings,
                label = "Оценок",
                icon = Icons.Default.Star,
                color = Color(0xFFFFD700)
            )
        }
    }
}


@Composable
fun ProfileStatCard(
    count: Int,
    label: String,
    icon: ImageVector,
    color: Color
) {
    Card(
        modifier = Modifier
            .width(70.dp)
            .padding(horizontal = 2.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFF5F5F5)
        ),
        border = BorderStroke(0.5.dp, Color.Black.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = color
            )
            Text(
                text = count.toString(),
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = Color.Black
            )
            Text(
                text = label,
                fontSize = 10.sp,
                color = Color.Black.copy(alpha = 0.6f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
fun ProfileScreen(
    bookViewModel: BookViewModel?,
    authViewModel: AuthViewModel?,
    onLogout: () -> Unit,
    onViewRecommendations: () -> Unit,
    currentUser: User?,
    isDarkTheme: Boolean,
    onThemeToggle: (Boolean) -> Unit
) {
    if (bookViewModel == null) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "Загрузка...",
                fontSize = 16.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }
        return
    }

    var showEditDialog by remember { mutableStateOf(false) }
    var editUsername by remember { mutableStateOf(currentUser?.username ?: "") }
    var editEmail by remember { mutableStateOf(currentUser?.email ?: "") }
    var snackbarMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(currentUser) {
        editUsername = currentUser?.username ?: ""
        editEmail = currentUser?.email ?: ""
    }

    val readingBooks by bookViewModel.readingBooks.collectAsState()
    val wantToReadBooks by bookViewModel.wantToReadBooks.collectAsState()
    val finishedBooks by bookViewModel.finishedBooks.collectAsState()
    val droppedBooks by bookViewModel.droppedBooks.collectAsState()
    val userRatings by bookViewModel.userRatings.collectAsState()

    val totalBooks = readingBooks.size + wantToReadBooks.size + finishedBooks.size + droppedBooks.size
    val totalRead = finishedBooks.size
    val totalReading = readingBooks.size
    val totalRatings = userRatings.size

    val recentBooks = (wantToReadBooks + readingBooks + finishedBooks + droppedBooks)
        .sortedByDescending { it.globalId  }
        .take(5)

    val registrationDate = remember(currentUser?.createdAt) {
        currentUser?.createdAt?.let {
            SimpleDateFormat("dd.MM.yyyy", Locale.getDefault()).format(Date(it))
        } ?: "недавно"
    }

    Scaffold(
        snackbarHost = {
            snackbarMessage?.let { message ->
                Snackbar(
                    modifier = Modifier.padding(16.dp),
                    action = {
                        TextButton(onClick = { snackbarMessage = null }) {
                            Text("OK")
                        }
                    }
                ) {
                    Text(message)
                }
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .verticalScroll(rememberScrollState())
                .padding(paddingValues)
        ) {
            ThemedProfileHeader(
                username = currentUser?.username ?: "Гость",
                registrationDate = registrationDate,
                isDarkTheme = isDarkTheme,
                onThemeToggle = onThemeToggle,
                onEditClick = { showEditDialog = true }
            )

            Spacer(modifier = Modifier.height(24.dp))

            ThemedProfileStats(
                totalBooks = totalBooks,
                totalRead = totalRead,
                totalReading = totalReading,
                totalRatings = totalRatings
            )

            Spacer(modifier = Modifier.height(24.dp))

            if (recentBooks.isNotEmpty()) {
                ThemedRecentBooksSection(
                    books = recentBooks,
                    userRatings = userRatings,
                    readingBooks = readingBooks,
                    finishedBooks = finishedBooks,
                    droppedBooks = droppedBooks,
                    onBookClick = {}
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            ThemedAchievementsSection(
                totalBooks = totalBooks,
                totalRead = totalRead,
                totalRatings = totalRatings
            )

            Spacer(modifier = Modifier.height(24.dp))

            ThemedAccountSettingsSection(
                username = currentUser?.username ?: "Не установлено",
                email = currentUser?.email?.ifEmpty { "Не указан" } ?: "Не указан",
                registrationDate = registrationDate,
                onLogout = onLogout
            )

            Spacer(modifier = Modifier.height(24.dp))
        }
    }

    if (showEditDialog && authViewModel != null) {
        var editUsername by remember { mutableStateOf(currentUser?.username ?: "") }
        var editEmail by remember { mutableStateOf(currentUser?.email ?: "") }
        var usernameError by remember { mutableStateOf<String?>(null) }
        var emailError by remember { mutableStateOf<String?>(null) }
        var isSaving by remember { mutableStateOf(false) }

        LaunchedEffect(showEditDialog) {
            if (showEditDialog) {
                editUsername = currentUser?.username ?: ""
                editEmail = currentUser?.email ?: ""
                usernameError = null
                emailError = null
            }
        }

        AlertDialog(
            onDismissRequest = { if (!isSaving) showEditDialog = false },
            title = { Text("Редактировать профиль") },
            text = {
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedTextField(
                        value = editUsername,
                        onValueChange = {
                            editUsername = it
                            usernameError = null
                        },
                        label = { Text("Имя пользователя") },
                        isError = usernameError != null,
                        supportingText = {
                            val error = usernameError
                            if (error != null) {
                                Text(error, color = MaterialTheme.colorScheme.error)
                            } else {
                                Text("3-20 символов, только буквы, цифры и _", fontSize = 11.sp)
                            }
                        },
                        singleLine = true,
                        enabled = !isSaving,
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = editEmail,
                        onValueChange = {
                            editEmail = it
                            emailError = null
                        },
                        label = { Text("Email (необязательно)") },
                        isError = emailError != null,
                        supportingText = {
                            emailError?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                        },
                        singleLine = true,
                        enabled = !isSaving,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (isSaving) return@TextButton

                        var hasError = false

                        if (editUsername != currentUser?.username) {
                            val usernameValidation = ValidationHelper.isValidUsername(editUsername)
                            if (usernameValidation is ValidationHelper.ValidationResult.Error) {
                                usernameError = usernameValidation.message
                                hasError = true
                            }
                        }

                        if (editEmail != currentUser?.email && editEmail.isNotEmpty()) {
                            val emailValidation = ValidationHelper.isValidEmail(editEmail)
                            if (emailValidation is ValidationHelper.ValidationResult.Error) {
                                emailError = emailValidation.message
                                hasError = true
                            }
                        }

                        if (!hasError) {
                            isSaving = true
                            authViewModel.updateProfile(
                                newUsername = editUsername,
                                newEmail = editEmail
                            ) { success, message ->
                                isSaving = false
                                if (success) {
                                    showEditDialog = false
                                } else {
                                    if (message.contains("имя", ignoreCase = true)) {
                                        usernameError = message
                                    } else if (message.contains("email", ignoreCase = true)) {
                                        emailError = message
                                    }
                                }
                            }
                        }
                    },
                    enabled = !isSaving
                ) {
                    if (isSaving) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp))
                    } else {
                        Text("Сохранить")
                    }
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { if (!isSaving) showEditDialog = false },
                    enabled = !isSaving
                ) {
                    Text("Отмена")
                }
            }
        )
    }
}

// BookApp.kt
@Composable
fun ThemedProfileHeader(
    username: String,
    registrationDate: String,
    isDarkTheme: Boolean,
    onThemeToggle: (Boolean) -> Unit,
    onEditClick: () -> Unit = {}
) {
    Column(
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .background(MaterialTheme.colorScheme.primary)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // Аватар
                Surface(
                    modifier = Modifier.size(90.dp),
                    shape = CircleShape,
                    color = MaterialTheme.colorScheme.surface,
                    border = BorderStroke(2.dp, MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.3f))
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(
                            text = username.firstOrNull()?.uppercase() ?: "👤",
                            fontSize = 36.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = username,
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = onEditClick,
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(
                            Icons.Default.Edit,
                            contentDescription = "Редактировать профиль",
                            tint = MaterialTheme.colorScheme.onPrimary
                        )
                    }
                }

                Text(
                    text = "Читатель с $registrationDate",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.7f)
                )
            }

            Surface(
                modifier = Modifier
                    .padding(16.dp)
                    .align(Alignment.TopEnd),
                shape = CircleShape,
                color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.2f)
            ) {
                IconButton(
                    onClick = { onThemeToggle(!isDarkTheme) }
                ) {
                    Icon(
                        imageVector = if (isDarkTheme) Icons.Default.DarkMode else Icons.Default.LightMode,
                        contentDescription = "Переключить тему",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            }
        }
    }
}

@Composable
fun ThemedProfileStats(
    totalBooks: Int,
    totalRead: Int,
    totalReading: Int,
    totalRatings: Int
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "Статистика",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            ThemedProfileStatCard(
                count = totalBooks,
                label = "Всего",
                icon = Icons.Default.Book,
                color = MaterialTheme.colorScheme.primary
            )
            ThemedProfileStatCard(
                count = totalRead,
                label = "Прочитано",
                icon = Icons.Default.Done,
                color = Color(0xFF34C759)
            )
            ThemedProfileStatCard(
                count = totalReading,
                label = "Читаю",
                icon = Icons.Default.MenuBook,
                color = Color(0xFFFF9500)
            )
            ThemedProfileStatCard(
                count = totalRatings,
                label = "Оценок",
                icon = Icons.Default.Star,
                color = Color(0xFFFFD700)
            )
        }
    }
}

@Composable
fun ThemedProfileStatCard(
    count: Int,
    label: String,
    icon: ImageVector,
    color: Color
) {
    Card(
        modifier = Modifier
            .width(70.dp)
            .padding(horizontal = 2.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = color
            )
            Text(
                text = count.toString(),
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = label,
                fontSize = 10.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
fun ThemedRecentBooksSection(
    books: List<Book>,
    userRatings: Map<String, Float>,
    readingBooks: List<Book>,
    finishedBooks: List<Book>,
    droppedBooks: List<Book>,
    onBookClick: (Book) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "Недавно добавленные",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
            elevation = CardDefaults.cardElevation(0.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                books.forEachIndexed { index, book ->
                    ThemedRecentBookItem(
                        book = book,
                        userRating = userRatings[book.globalId] ?: 0f,
                        readingBooks = readingBooks,
                        finishedBooks = finishedBooks,
                        droppedBooks = droppedBooks,
                        onBookClick = { onBookClick(book) }
                    )
                    if (index < books.size - 1) {
                        Divider(
                            modifier = Modifier.padding(vertical = 8.dp),
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ThemedRecentBookItem(
    book: Book,
    userRating: Float,
    readingBooks: List<Book>,
    finishedBooks: List<Book>,
    droppedBooks: List<Book>,
    onBookClick: () -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is PressInteraction.Press -> isPressed = true
                is PressInteraction.Release -> isPressed = false
                is PressInteraction.Cancel -> isPressed = false
            }
        }
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .scale(if (isPressed) 0.99f else 1f)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onBookClick
            )
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        if (book.coverUrl.isNotEmpty()) {
            AsyncImage(
                model = book.coverUrl,
                contentDescription = null,
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Crop
            )
        } else {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "📚",
                    fontSize = 20.sp,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                )
            }
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 12.dp)
        ) {
            Text(
                text = book.title,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = book.author,
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            if (userRating > 0) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 2.dp)
                ) {
                    Icon(
                        Icons.Filled.Star,
                        contentDescription = null,
                        modifier = Modifier.size(12.dp),
                        tint = Color(0xFFFFD700)
                    )
                    Text(
                        text = " ${userRating.toInt()}/5",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                }
            }
        }

        val statusIcon = when {
            readingBooks.any { it.globalId  == book.globalId  } -> "📖"
            finishedBooks.any { it.globalId  == book.globalId  } -> "✅"
            droppedBooks.any { it.globalId  == book.globalId  } -> "⏸️"
            else -> "🔖"
        }
        Text(
            text = statusIcon,
            fontSize = 16.sp
        )
    }
}

@Composable
fun ThemedAchievementsSection(
    totalBooks: Int,
    totalRead: Int,
    totalRatings: Int
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "Достижения",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
            elevation = CardDefaults.cardElevation(0.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                ThemedAchievementItem(
                    title = "Новичок",
                    description = "Добавьте первую книгу",
                    achieved = totalBooks > 0,
                    icon = "🎯"
                )
                Divider(
                    modifier = Modifier.padding(vertical = 8.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                )
                ThemedAchievementItem(
                    title = "Читатель",
                    description = "Прочитайте 5 книг",
                    achieved = totalRead >= 5,
                    icon = "📖"
                )
                Divider(
                    modifier = Modifier.padding(vertical = 8.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                )
                ThemedAchievementItem(
                    title = "Книжный червь",
                    description = "Добавьте 10 книг",
                    achieved = totalBooks >= 10,
                    icon = "🐛"
                )
                Divider(
                    modifier = Modifier.padding(vertical = 8.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                )
                ThemedAchievementItem(
                    title = "Критик",
                    description = "Оцените 5 книг",
                    achieved = totalRatings >= 5,
                    icon = "⭐"
                )
            }
        }
    }
}

@Composable
fun ThemedAchievementItem(
    title: String,
    description: String,
    achieved: Boolean,
    icon: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .background(
                    color = if (achieved)
                        MaterialTheme.colorScheme.primary.copy(alpha = 0.2f)
                    else
                        MaterialTheme.colorScheme.surface,
                    shape = CircleShape
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = icon,
                fontSize = 18.sp,
                color = if (achieved)
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
            )
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 12.dp)
        ) {
            Text(
                text = title,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = if (achieved)
                    MaterialTheme.colorScheme.onSurface
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
            )
            Text(
                text = description,
                fontSize = 12.sp,
                color = if (achieved)
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
            )
        }

        Text(
            text = if (achieved) "✅" else "🔒",
            fontSize = 16.sp,
            color = if (achieved) Color(0xFF34C759) else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
        )
    }
}

@Composable
fun ThemedAccountSettingsSection(
    username: String,
    email: String,
    registrationDate: String,
    onLogout: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
    ) {
        Text(
            text = "Настройки аккаунта",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)),
            elevation = CardDefaults.cardElevation(0.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                ThemedProfileSettingRow(
                    title = "Имя пользователя",
                    value = username
                )
                Divider(
                    modifier = Modifier.padding(vertical = 8.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                )
                ThemedProfileSettingRow(
                    title = "Email",
                    value = email
                )
                Divider(
                    modifier = Modifier.padding(vertical = 8.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f)
                )
                ThemedProfileSettingRow(
                    title = "Дата регистрации",
                    value = registrationDate
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Кнопка выхода
                var isPressed by remember { mutableStateOf(false) }
                val interactionSource = remember { MutableInteractionSource() }

                LaunchedEffect(interactionSource) {
                    interactionSource.interactions.collect { interaction ->
                        when (interaction) {
                            is PressInteraction.Press -> isPressed = true
                            is PressInteraction.Release -> isPressed = false
                            is PressInteraction.Cancel -> isPressed = false
                        }
                    }
                }

                Button(
                    onClick = onLogout,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(44.dp)
                        .scale(if (isPressed) 0.98f else 1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Transparent,
                        contentColor = Color(0xFFFF3B30)
                    ),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color(0xFFFF3B30).copy(alpha = 0.5f)),
                    elevation = ButtonDefaults.buttonElevation(0.dp),
                    interactionSource = interactionSource
                ) {
                    Icon(
                        Icons.Default.Logout,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Выйти из аккаунта",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color(0xFFFF3B30)
                    )
                }
            }
        }
    }
}

@Composable
fun ThemedProfileSettingRow(
    title: String,
    value: String
) {
    Column(
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = title,
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
        )
        Text(
            text = value,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

