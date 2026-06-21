package com.samira.bookapp.ui.theme
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.samira.bookapp.data.Book
import com.samira.bookapp.data.BookStatus
import com.samira.bookapp.network.GoogleBook
import com.samira.bookapp.viewmodel.BookViewModel
import kotlinx.coroutines.launch
import androidx.compose.material.icons.filled.Bookmark
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.filled.ArrowBack
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.compose.material.icons.filled.Headphones
import androidx.compose.ui.platform.LocalContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BookDetailsScreen(
    book: Book,
    originalGoogleBook: GoogleBook? = null,
    bookViewModel: BookViewModel,
    onBackClick: () -> Unit,
) {
    Log.e("TEST_SCREEN", "ОТКРЫТ BookDetailsScreen")
    var showReadingScreen by remember { mutableStateOf(false) }
    val bookContent by bookViewModel.bookContent.collectAsState()
    val isLoadingContent by bookViewModel.isLoadingContent.collectAsState()
    val contentError by bookViewModel.contentError.collectAsState()
    val coroutineScope = rememberCoroutineScope()
    val bookStatusMap by bookViewModel.bookStatusByGlobalId.collectAsState()
    val userRatingsMap by bookViewModel.userRatings.collectAsState()
    val currentStatus = bookStatusMap[book.globalId]
    val currentUserRating = userRatingsMap[book.globalId] ?: 0f
    var showRatingDialog by remember { mutableStateOf(false) }
    var temporaryRating by remember { mutableStateOf(currentUserRating) }
    val snackbarHostState = remember { SnackbarHostState() }
    val context = LocalContext.current

    if (showReadingScreen) {
        ReadingScreen(
            book = book,
            bookContent = bookContent,
            isLoading = isLoadingContent,
            errorMessage = contentError,
            bookViewModel = bookViewModel,
            onBackClick = {
                showReadingScreen = false
                bookViewModel.clearBookContent()
            }
        )
    } else {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text(
                            text = book.title,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    },
                    navigationIcon = {
                        IconButton(onClick = onBackClick) {
                            Icon(Icons.Default.ArrowBack, contentDescription = "Назад")
                        }
                    }
                )
            },
            snackbarHost = { SnackbarHost(snackbarHostState) }

        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .verticalScroll(rememberScrollState())
            ) {

                if (book.coverUrl.isNotEmpty()) {
                    AsyncImage(
                        model = book.coverUrl,
                        contentDescription = "Обложка ${book.title}",
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
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant,
                                MaterialTheme.shapes.medium
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text("📚", style = MaterialTheme.typography.displayMedium)
                            Text(
                                "Обложка отсутствует",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }


                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = book.title,
                        style = MaterialTheme.typography.headlineMedium,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = "Автор: ${book.author}",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = "Жанр: ${book.genre}",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(bottom = 4.dp)
                    ) {
                        Text(
                            text = "Средний рейтинг: ",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        ReadOnlyRatingBar(
                            rating = book.averageRating,
                            modifier = Modifier.padding(horizontal = 4.dp)
                        )
                        Text(
                            text = " ${"%.1f".format(book.averageRating)}/5",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }

                    if (currentUserRating > 0) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(bottom = 8.dp)
                        ) {
                            Text(
                                text = "Ваша оценка: ",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            ReadOnlyRatingBar(
                                rating = currentUserRating,
                                modifier = Modifier.padding(horizontal = 4.dp)
                            )
                            Text(
                                text = " ${currentUserRating.toInt()}/5",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                            IconButton(
                                onClick = {
                                    temporaryRating = currentUserRating
                                    showRatingDialog = true
                                },
                                modifier = Modifier.size(24.dp)
                            ) {
                                Icon(
                                    Icons.Default.Edit,
                                    contentDescription = "Изменить оценку",
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                        }
                    } else {
                        Button(
                            onClick = {
                                temporaryRating = 3f
                                showRatingDialog = true
                            },
                            modifier = Modifier.padding(bottom = 8.dp)
                        ) {
                            Icon(Icons.Default.Star, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Поставить оценку")
                        }
                    }
                    Text(
                        text = "Описание:",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(bottom = 4.dp, top = 8.dp)
                    )
                    Text(
                        text = book.description,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                    Button(
                        onClick = {
                            coroutineScope.launch {
                                bookViewModel.loadBookContent(book.globalId)
                                showReadingScreen = true
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Icon(Icons.Default.MenuBook, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Читать")
                    }
                    Log.d("BOOK_DETAILS", "🎵 ${book.title} — playlistUrl = '${book.playlistUrl}', длина = ${book.playlistUrl.length}")
                    if (book.playlistUrl.isNotEmpty()) {
                        Button(
                            onClick = {
                                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(book.playlistUrl))
                                context.startActivity(intent)
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF1E1E1E)
                            ),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFFFD700))
                        ) {
                            Icon(
                                Icons.Default.Headphones,
                                contentDescription = null,
                                tint = Color(0xFFFFD700)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Слушать плейлист в Яндекс Музыке",
                                color = Color(0xFFFFD700)
                            )
                        }
                    }
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Статус чтения:",
                            style = MaterialTheme.typography.titleSmall,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        if (currentStatus != null) {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = when (currentStatus) {
                                        BookStatus.READING -> Color(0xFF4CAF50).copy(alpha = 0.1f)
                                        BookStatus.WANT_TO_READ -> Color(0xFF2196F3).copy(alpha = 0.1f)
                                        BookStatus.FINISHED -> Color(0xFF9C27B0).copy(alpha = 0.1f)
                                        BookStatus.DROPPED -> Color(0xFFF44336).copy(alpha = 0.1f)
                                    }
                                )
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = when (currentStatus) {
                                            BookStatus.WANT_TO_READ -> Icons.Default.Bookmark
                                            BookStatus.READING -> Icons.Default.MenuBook
                                            BookStatus.FINISHED -> Icons.Default.Done
                                            BookStatus.DROPPED -> Icons.Default.Clear
                                        },
                                        contentDescription = "Текущий статус",
                                        tint = when (currentStatus) {
                                            BookStatus.READING -> Color(0xFF4CAF50)
                                            BookStatus.WANT_TO_READ -> Color(0xFF2196F3)
                                            BookStatus.FINISHED -> Color(0xFF9C27B0)
                                            BookStatus.DROPPED -> Color(0xFFF44336)
                                        }
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column(
                                        modifier = Modifier.weight(1f)
                                    ) {
                                        Text(
                                            text = "Текущий статус:",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                        Text(
                                            text = when (currentStatus) {
                                                BookStatus.WANT_TO_READ -> "Хочу прочитать"
                                                BookStatus.READING -> "Читаю сейчас"
                                                BookStatus.FINISHED -> "Прочитано"
                                                BookStatus.DROPPED -> "Не дочитал"
                                            },
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                    }
                                }
                            }
                        }
                        BookStatus.values().forEach { status ->
                            Button(
                                onClick = {
                                    coroutineScope.launch {
                                        bookViewModel.addBookToCollection(book.globalId, status)
                                        snackbarHostState.showSnackbar(
                                            "Статус изменен на: ${when (status) {
                                                BookStatus.WANT_TO_READ -> "Хочу прочитать"
                                                BookStatus.READING -> "Читаю сейчас"
                                                BookStatus.FINISHED -> "Прочитано"
                                                BookStatus.DROPPED -> "Не дочитал"
                                            }}",
                                            duration = SnackbarDuration.Short
                                        )
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (currentStatus == status)
                                        MaterialTheme.colorScheme.primary
                                    else MaterialTheme.colorScheme.secondary
                                ),
                                enabled = currentStatus != status
                            ) {
                                val icon = when (status) {
                                    BookStatus.WANT_TO_READ -> Icons.Default.Bookmark
                                    BookStatus.READING -> Icons.Default.MenuBook
                                    BookStatus.FINISHED -> Icons.Default.Done
                                    BookStatus.DROPPED -> Icons.Default.Clear
                                }
                                val text = when (status) {
                                    BookStatus.WANT_TO_READ -> "Хочу прочитать"
                                    BookStatus.READING -> "Читаю сейчас"
                                    BookStatus.FINISHED -> "Прочитано"
                                    BookStatus.DROPPED -> "Не дочитал"
                                }

                                Icon(icon, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(text)
                            }
                        }
                        if (currentStatus != null) {
                            OutlinedButton(
                                onClick = {
                                    coroutineScope.launch {
                                        bookViewModel.removeBookFromCollection(book.globalId)
                                        snackbarHostState.showSnackbar(
                                            "Книга удалена из библиотеки",
                                            duration = SnackbarDuration.Short
                                        )
                                        onBackClick()
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                )
                            ) {
                                Icon(Icons.Default.Delete, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Удалить из библиотеки")
                            }
                        }
                    }
                }
            }
        }
    }
    if (showRatingDialog) {
        AlertDialog(
            onDismissRequest = { showRatingDialog = false },
            title = { Text("Оцените книгу") },
            text = {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text("Выберите оценку от 1 до 5 звезд")

                    var selectedRating by remember { mutableStateOf(temporaryRating) }

                    Row(
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        for (i in 1..5) {
                            IconButton(
                                onClick = { selectedRating = i.toFloat() }
                            ) {
                                Icon(
                                    imageVector = if (i <= selectedRating)
                                        Icons.Filled.Star
                                    else Icons.Outlined.Star,
                                    contentDescription = "$i звезд",
                                    tint = if (i <= selectedRating)
                                        Color(0xFFFFD700)
                                    else Color.Gray,
                                    modifier = Modifier.size(32.dp)
                                )
                            }
                        }
                    }

                    Text(
                        text = "Оценка: ${selectedRating.toInt()}/5",
                        style = MaterialTheme.typography.bodyMedium
                    )

                    temporaryRating = selectedRating
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        coroutineScope.launch {
                            bookViewModel.updateBookRating(book.globalId, temporaryRating)
                            showRatingDialog = false
                            snackbarHostState.showSnackbar(
                                "Оценка сохранена",
                                duration = SnackbarDuration.Short
                            )
                        }
                    }
                ) {
                    Text("Сохранить")
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { showRatingDialog = false }
                ) {
                    Text("Отмена")
                }
            }
        )
    }
}

fun splitTextIntoPages(text: String, charsPerPage: Int = 1500): List<String> {
    if (text.length <= charsPerPage) return listOf(text)

    val pages = mutableListOf<String>()
    var start = 0

    while (start < text.length) {
        var end = start + charsPerPage
        if (end >= text.length) {
            pages.add(text.substring(start))
            break
        }
        val boundaryCandidates = listOf('.', '!', '?', ';', ':', ',', ' ', '\n')
        var bestSplit = end
        for (i in end downTo start) {
            if (boundaryCandidates.contains(text[i])) {
                bestSplit = i + 1
                break
            }
        }
        if (bestSplit == end && end > start) {
            bestSplit = end
        }

        pages.add(text.substring(start, bestSplit).trim())
        start = bestSplit
    }
    return pages.filter { it.isNotBlank() }
}

@OptIn(ExperimentalFoundationApi::class, ExperimentalMaterial3Api::class)
@Composable
fun ReadingScreen(
    book: Book,
    bookContent: String?,
    isLoading: Boolean,
    errorMessage: String?,
    bookViewModel: BookViewModel,
    onBackClick: () -> Unit
) {
    val pages = remember(bookContent) {
        if (!bookContent.isNullOrEmpty()) splitTextIntoPages(bookContent, 1500)
        else emptyList()
    }

    val savedPage = bookViewModel.getReadingPosition(book.globalId)
    val initialPage = if (savedPage in pages.indices) savedPage else 0

    val pagerState = rememberPagerState(
        initialPage = initialPage,
        pageCount = { pages.size }
    )

    LaunchedEffect(pagerState.currentPage) {
        val page = pagerState.currentPage
        if (page > 0) {
            bookViewModel.saveReadingPosition(book.globalId, page)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(book.title) },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Назад")
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when {
                isLoading -> {
                    Box(Modifier.fillMaxSize(), Alignment.Center) {
                        CircularProgressIndicator()
                    }
                }
                errorMessage != null -> {
                    Box(Modifier.fillMaxSize(), Alignment.Center) {
                        Text(errorMessage)
                    }
                }
                pages.isEmpty() -> {
                    Box(Modifier.fillMaxSize(), Alignment.Center) {
                        Text("Текст книги недоступен")
                    }
                }
                else -> {
                    HorizontalPager(
                        state = pagerState,
                        modifier = Modifier.fillMaxSize()
                    ) { page ->
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(24.dp)
                        ) {
                            Text(
                                text = pages[page],
                                style = MaterialTheme.typography.bodyLarge,
                                lineHeight = 26.sp
                            )
                        }
                    }
                }
            }
        }
    }
}