package com.samira.bookapp.viewmodel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.samira.bookapp.data.Book
import com.samira.bookapp.data.BookStatus
import com.samira.bookapp.network.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.samira.bookapp.data.HybridNeuralRepository
import org.json.JSONArray
import java.io.FileNotFoundException
class BookViewModel(
    private val context: Context,
    private val userId: Long
) : ViewModel() {

    private val hybridRepository = HybridNeuralRepository()

    private val _readingBooks = MutableStateFlow<List<Book>>(emptyList())
    val readingBooks: StateFlow<List<Book>> = _readingBooks.asStateFlow()

    private val _wantToReadBooks = MutableStateFlow<List<Book>>(emptyList())
    val wantToReadBooks: StateFlow<List<Book>> = _wantToReadBooks.asStateFlow()

    private val _finishedBooks = MutableStateFlow<List<Book>>(emptyList())
    val finishedBooks: StateFlow<List<Book>> = _finishedBooks.asStateFlow()

    private val _droppedBooks = MutableStateFlow<List<Book>>(emptyList())
    val droppedBooks: StateFlow<List<Book>> = _droppedBooks.asStateFlow()

    private val _searchResults = MutableStateFlow<List<GoogleBook>>(emptyList())
    val searchResults: StateFlow<List<GoogleBook>> = _searchResults.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private val _searchError = MutableStateFlow<String?>(null)
    val searchError: StateFlow<String?> = _searchError.asStateFlow()

    private val _openLibraryGenres = MutableStateFlow<List<String>>(emptyList())
    val openLibraryGenres: StateFlow<List<String>> = _openLibraryGenres.asStateFlow()

    private val _selectedGenre = MutableStateFlow<String?>(null)
    val selectedGenre: StateFlow<String?> = _selectedGenre.asStateFlow()

    private val _userRatings = MutableStateFlow<Map<String, Float>>(emptyMap())
    val userRatings: StateFlow<Map<String, Float>> = _userRatings.asStateFlow()

    private val _aiRecommendedBooks = MutableStateFlow<List<Book>>(emptyList())
    val aiRecommendedBooks: StateFlow<List<Book>> = _aiRecommendedBooks.asStateFlow()

    private val _recommendationLoading = MutableStateFlow(false)
    val recommendationLoading: StateFlow<Boolean> = _recommendationLoading.asStateFlow()

    private val _serverAvailable = MutableStateFlow(false)
    val serverAvailable: StateFlow<Boolean> = _serverAvailable.asStateFlow()

    private val _userStats = MutableStateFlow<Map<String, Any>>(emptyMap())
    val userStats: StateFlow<Map<String, Any>> = _userStats.asStateFlow()

    private val _hybridRecommendationQuality = MutableStateFlow<String>("базовая")
    val hybridRecommendationQuality: StateFlow<String> = _hybridRecommendationQuality.asStateFlow()

    private val _genrePreferences = MutableStateFlow<Map<String, Float>>(emptyMap())
    val genrePreferences: StateFlow<Map<String, Float>> = _genrePreferences.asStateFlow()

    private val _bookStatusByGlobalId = MutableStateFlow<Map<String, BookStatus>>(emptyMap())
    val bookStatusByGlobalId: StateFlow<Map<String, BookStatus>> = _bookStatusByGlobalId.asStateFlow()


    private var allBooksCache: List<Book> = emptyList()

    private val _bestsellerBooks = MutableStateFlow<List<Book>>(emptyList())
    val bestsellerBooks: StateFlow<List<Book>> = _bestsellerBooks.asStateFlow()

    private val _unfinishedBooks = MutableStateFlow<List<Book>>(emptyList())
    val unfinishedBooks: StateFlow<List<Book>> = _unfinishedBooks.asStateFlow()

    private val _isLoadingBestsellers = MutableStateFlow(false)
    val isLoadingBestsellers: StateFlow<Boolean> = _isLoadingBestsellers.asStateFlow()

    private val _classicBooks = MutableStateFlow<List<Book>>(emptyList())
    val classicBooks: StateFlow<List<Book>> = _classicBooks.asStateFlow()

    private val _isLoadingClassic = MutableStateFlow(false)
    val isLoadingClassic: StateFlow<Boolean> = _isLoadingClassic.asStateFlow()

    private val _topRatedBooks = MutableStateFlow<List<Book>>(emptyList())
    val topRatedBooks: StateFlow<List<Book>> = _topRatedBooks.asStateFlow()

    private val _isLoadingTopRated = MutableStateFlow(false)
    val isLoadingTopRated: StateFlow<Boolean> = _isLoadingTopRated.asStateFlow()


    private val _bookContent = MutableStateFlow<String?>(null)
    val bookContent: StateFlow<String?> = _bookContent.asStateFlow()

    private val _isLoadingContent = MutableStateFlow(false)
    val isLoadingContent: StateFlow<Boolean> = _isLoadingContent.asStateFlow()

    private val _contentError = MutableStateFlow<String?>(null)
    private var localBooksCache: List<Book> = emptyList()
    val contentError: StateFlow<String?> = _contentError.asStateFlow()
    init {
        Log.d("BOOK_VIEWMODEL", "🎯 Инициализация для пользователя ID: $userId")
        viewModelScope.launch {


            localBooksCache = loadBooksFromLocalJson()
            allBooksCache = localBooksCache
            Log.d("BOOK_VIEWMODEL", "📚 Загружено ${localBooksCache.size} локальных книг")
            loadUserLibrary()
            loadHybridRecommendations()
            checkHybridServer()
            loadBestsellerBooks()
            loadUnfinishedBooks()
            loadClassicBooks()
            loadTopRatedBooks()

        }
        _openLibraryGenres.value = getAvailableGenres()


    }
    private fun loadBooksFromLocalJson(): List<Book> {
        val booksList = mutableListOf<Book>()
        try {
            val inputStream = context.assets.open("books_with_source.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val jsonArray = JSONArray(jsonString)

            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val title = obj.optString("title", "")
                val author = obj.optString("author", "")
                val genre = obj.optString("genre", "")
                val description = obj.optString("description", "")
                val coverUrl = obj.optString("cover", "")
                val averageRating = obj.optDouble("average_rating", 0.0).toFloat()
                val isBestseller = obj.optBoolean("is_bestseller", false)
                val sourceFile = obj.optString("source_file", "")
                val playlistUrl = obj.optString("playlist_url", "")  // ← вот это добавить
                Log.d("BOOK_VIEWMODEL", "📀 $title — playlistUrl = '$playlistUrl'")
                if (title.contains("Гарри Поттер") || title.contains("P.S.") || title.contains("Анна Каренина")) {
                    Log.d("BOOK_VIEWMODEL", "📀 $title — playlistUrl = '$playlistUrl'")
                }
                if (title.isNotBlank() && author.isNotBlank()) {
                    val book = Book(
                        id = 0,
                        userId = userId,
                        title = title,
                        author = author,
                        genre = genre,
                        description = description,
                        coverUrl = coverUrl,
                        averageRating = averageRating,
                        tags = "",
                        language = "ru",
                        ratingsCount = 0,
                        globalId = generateGlobalId(title, author),
                        isBestseller = isBestseller,
                        playlistUrl = playlistUrl,
                        sourceFile = sourceFile
                    )
                    booksList.add(book)
                }
            }
        } catch (e: Exception) {
            Log.e("BOOK_VIEWMODEL", "Ошибка загрузки локального JSON из assets", e)
        }
        return booksList
    }
    fun saveReadingPosition(bookId: String, position: Int) {
        val prefs = context.getSharedPreferences("reading_positions", Context.MODE_PRIVATE)
        prefs.edit().putInt(bookId, position).apply()
    }

    fun getReadingPosition(bookId: String): Int {
        val prefs = context.getSharedPreferences("reading_positions", Context.MODE_PRIVATE)
        return prefs.getInt(bookId, 0)
    }

    suspend fun loadBookContent(bookId: String) {
        _isLoadingContent.value = true
        _contentError.value = null
        try {
            // Сначала ищем книгу в кэше
            val book = allBooksCache.find { it.globalId == bookId }
            if (book == null) {
                _contentError.value = "Книга не найдена"
                _bookContent.value = null
                return
            }
            val sourcePath = book.sourceFile
            if (sourcePath.isBlank()) {
                _contentError.value = "Путь к тексту не указан"
                _bookContent.value = null
                return
            }

            val content = context.assets.open(sourcePath).bufferedReader(Charsets.UTF_8).use { it.readText() }
            _bookContent.value = content
        } catch (e: FileNotFoundException) {
            _contentError.value = "Текст книги не найден в папке assets"
            _bookContent.value = null
        } catch (e: Exception) {
            _contentError.value = "Ошибка загрузки: ${e.message}"
            _bookContent.value = null
        } finally {
            _isLoadingContent.value = false
        }
    }

    fun clearBookContent() {
        _bookContent.value = null
        _isLoadingContent.value = false
        _contentError.value = null
    }

    fun loadClassicBooks() {
        viewModelScope.launch {
            _isLoadingClassic.value = true
            try {
                if (allBooksCache.isEmpty()) {
                    _isLoadingClassic.value = false
                    return@launch
                }
                val classics = allBooksCache.filter { book ->
                    book.genre.contains("Классика", ignoreCase = true) ||
                            book.genre.contains("Classic", ignoreCase = true)
                }.take(10)
                _classicBooks.value = classics
            } catch (e: Exception) {
                Log.e("BookViewModel", "Ошибка загрузки классики", e)
            } finally {
                _isLoadingClassic.value = false
            }
        }
    }

    fun loadTopRatedBooks() {
        viewModelScope.launch {
            _isLoadingTopRated.value = true
            try {
                if (allBooksCache.isEmpty()) {
                    _isLoadingTopRated.value = false
                    return@launch
                }
                val userLibraryIds = _bookStatusByGlobalId.value.keys
                val topRated = allBooksCache
                    .filter { it.averageRating >= 4.5 && it.globalId !in userLibraryIds }
                    .sortedByDescending { it.averageRating }
                    .take(10)
                _topRatedBooks.value = topRated
            } catch (e: Exception) {
                Log.e("BookViewModel", "Ошибка загрузки топ рейтинга", e)
            } finally {
                _isLoadingTopRated.value = false
            }
        }
    }

    fun loadBestsellerBooks() {
        viewModelScope.launch {
            _isLoadingBestsellers.value = true
            try {
                if (allBooksCache.isEmpty()) {
                    _isLoadingBestsellers.value = false
                    return@launch
                }
                val bestsellers = allBooksCache.filter { it.isBestseller }.shuffled()
                _bestsellerBooks.value = bestsellers
            } catch (e: Exception) {
                Log.e("BookViewModel", "Ошибка загрузки бестселлеров", e)
            } finally {
                _isLoadingBestsellers.value = false
            }
        }
    }
    fun loadUnfinishedBooks() {
        viewModelScope.launch {
            val reading = _readingBooks.value
            _unfinishedBooks.value = reading.take(10)
        }
    }

    fun quickSearchBooks(query: String) {
        if (_isSearching.value) return
        viewModelScope.launch {
            _isSearching.value = true
            _searchError.value = null
            try {
                // Используем обычный поиск, а не чат
                val response = NeuralRecommendationApi.retrofit.searchBooks(query, userId)
                _searchResults.value = response.results
                if (response.results.isEmpty()) {
                    _searchError.value = "Ничего не найдено"
                }
            } catch (e: Exception) {
                _searchError.value = e.message
                _searchResults.value = emptyList()
            } finally {
                _isSearching.value = false
            }
        }
    }
    fun searchBooksByGenreLocally(genre: String) {
        viewModelScope.launch {
            _isSearching.value = true
            _searchError.value = null
            try {
                if (allBooksCache.isEmpty()) {
                    _searchError.value = "Книги ещё загружаются, попробуйте через секунду"
                    _isSearching.value = false
                    return@launch
                }
                val targetGenre = genre.trim().lowercase()

                val results = allBooksCache.filter { book ->
                    val bookGenreRaw = book.genre.trim().lowercase()

                    val directMatch = bookGenreRaw == targetGenre

                    val translatedToRussian = translateGenreToRussian(book.genre).lowercase()
                    val russianMatch = translatedToRussian == targetGenre
                    val translatedToEnglish = translateGenreToEnglish(genre).lowercase()
                    val englishMatch = bookGenreRaw == translatedToEnglish

                    directMatch || russianMatch || englishMatch
                }


                val googleBooks = results.map { book ->
                    GoogleBook(
                        id = book.globalId,
                        volumeInfo = VolumeInfo(
                            title = book.title,
                            authors = listOf(book.author),
                            description = book.description,
                            categories = listOf(book.genre),
                            averageRating = book.averageRating,
                            ratingsCount = book.ratingsCount,
                            imageLinks = if (book.coverUrl.isNotEmpty())
                                ImageLinks(thumbnail = book.coverUrl) else null,
                            language = book.language
                        )
                    )
                }

                _searchResults.value = googleBooks
                if (googleBooks.isEmpty()) {
                    _searchError.value = "Книги по жанру '$genre' не найдены"
                } else {
                    _searchError.value = null
                }
            } catch (e: Exception) {
                _searchError.value = "Ошибка поиска: ${e.message}"
                _searchResults.value = emptyList()
            } finally {
                _isSearching.value = false
            }
        }
    }
    fun searchBooksByGenreAndQueryLocally(genre: String, query: String) {
        viewModelScope.launch {
            _isSearching.value = true
            _searchError.value = null
            try {
                val targetGenre = genre.trim().lowercase()
                val searchQuery = query.trim().lowercase()

                val results = allBooksCache.filter { book ->

                    val bookGenreRaw = book.genre.trim().lowercase()
                    val translatedToRussian = translateGenreToRussian(book.genre).lowercase()
                    val translatedToEnglish = translateGenreToEnglish(genre).lowercase()

                    val genreMatch = bookGenreRaw == targetGenre ||
                            translatedToRussian == targetGenre ||
                            bookGenreRaw == translatedToEnglish

                    val titleMatch = book.title.lowercase().contains(searchQuery)
                    val authorMatch = book.author.lowercase().contains(searchQuery)

                    genreMatch && (titleMatch || authorMatch)
                }

                val googleBooks = results.map { book ->
                    GoogleBook(
                        id = book.globalId,
                        volumeInfo = VolumeInfo(
                            title = book.title,
                            authors = listOf(book.author),
                            description = book.description,
                            categories = listOf(book.genre),
                            averageRating = book.averageRating,
                            ratingsCount = book.ratingsCount,
                            imageLinks = if (book.coverUrl.isNotEmpty())
                                ImageLinks(thumbnail = book.coverUrl) else null,
                            language = book.language
                        )
                    )
                }

                _searchResults.value = googleBooks
                if (googleBooks.isEmpty()) {
                    _searchError.value = "Книг по жанру '$genre' с названием '$query' не найдено"
                } else {
                    _searchError.value = null
                }
            } catch (e: Exception) {
                _searchError.value = "Ошибка поиска: ${e.message}"
                _searchResults.value = emptyList()
            } finally {
                _isSearching.value = false
            }
        }
    }
    fun searchBooksLocallyByQuery(query: String) {
        viewModelScope.launch {
            _isSearching.value = true
            _searchError.value = null
            try {
                val searchQuery = query.trim().lowercase()

                val results = allBooksCache.filter { book ->
                    book.title.lowercase().contains(searchQuery) ||
                            book.author.lowercase().contains(searchQuery) ||
                            book.genre.lowercase().contains(searchQuery)
                }

                val googleBooks = results.map { book ->
                    GoogleBook(
                        id = book.globalId,
                        volumeInfo = VolumeInfo(
                            title = book.title,
                            authors = listOf(book.author),
                            description = book.description,
                            categories = listOf(book.genre),
                            averageRating = book.averageRating,
                            ratingsCount = book.ratingsCount,
                            imageLinks = if (book.coverUrl.isNotEmpty())
                                ImageLinks(thumbnail = book.coverUrl) else null,
                            language = book.language
                        )
                    )
                }

                _searchResults.value = googleBooks
                if (googleBooks.isEmpty()) {
                    _searchError.value = "Ничего не найдено по запросу \"$query\""
                } else {
                    _searchError.value = null
                }
            } catch (e: Exception) {
                _searchError.value = "Ошибка поиска: ${e.message}"
                _searchResults.value = emptyList()
            } finally {
                _isSearching.value = false
            }
        }
    }
    fun selectGenre(genre: String?) {
        _selectedGenre.value = genre
        if (genre != null) {
            searchBooksByGenreLocally(genre)
        } else {
            _searchResults.value = emptyList()
        }
    }

    fun clearSearchResults() {
        _searchResults.value = emptyList()
        _searchError.value = null
    }

    private suspend fun loadUserLibrary() {
        try {
            val books = NeuralRecommendationApi.retrofit.getUserBooks(userId)
            // Все книги уже есть в allBooksCache, просто используем их для дополнения
            val bookMap = allBooksCache.associateBy { it.globalId }
            val bookList = books.map { dto ->
                val cachedBook = bookMap[dto.global_id]
                Book(
                    id = 0,
                    userId = userId,
                    title = dto.title,
                    author = dto.author,
                    genre = dto.genre,
                    description = cachedBook?.description ?: "",
                    coverUrl = dto.cover_url,
                    averageRating = dto.average_rating,
                    tags = "",
                    language = "ru",
                    ratingsCount = 0,
                    globalId = dto.global_id,
                    playlistUrl = dto.playlist_url.ifEmpty { cachedBook?.playlistUrl ?: "" },
                    sourceFile = cachedBook?.sourceFile ?: ""
                )
            }
            val reading = mutableListOf<Book>()
            val want = mutableListOf<Book>()
            val finished = mutableListOf<Book>()
            val dropped = mutableListOf<Book>()
            bookList.forEach { book ->
                val status = books.find { it.global_id == book.globalId }?.status ?: return@forEach
                when (status.lowercase()) {
                    "reading" -> reading.add(book)
                    "want_to_read" -> want.add(book)
                    "finished" -> finished.add(book)
                    "dropped" -> dropped.add(book)
                }
            }
            _readingBooks.value = reading
            loadUnfinishedBooks()
            _wantToReadBooks.value = want
            _finishedBooks.value = finished
            _droppedBooks.value = dropped

            val statusMap = books.associate { it.global_id to stringToBookStatus(it.status) }
            _bookStatusByGlobalId.value = statusMap

            val ratingsMap = books.filter { it.user_rating > 0 }.associate { it.global_id to it.user_rating }
            _userRatings.value = ratingsMap

            updateHybridRecommendationQuality()
            updateGenrePreferences()
        } catch (e: Exception) {
            Log.e("BOOK_VIEWMODEL", "Ошибка загрузки библиотеки", e)
        }
    }

    private fun stringToBookStatus(status: String): BookStatus {
        return when (status.lowercase()) {
            "reading" -> BookStatus.READING
            "want_to_read" -> BookStatus.WANT_TO_READ
            "finished" -> BookStatus.FINISHED
            else -> BookStatus.DROPPED
        }
    }

    suspend fun addBookToCollection(globalId: String, status: BookStatus) {

        val book = allBooksCache.find { it.globalId == globalId }
            ?: _searchResults.value.find { it.id == globalId }?.let { googleBook ->
                Book(
                    id = 0,
                    userId = userId,
                    title = googleBook.volumeInfo.title,
                    author = googleBook.volumeInfo.authors?.joinToString(", ") ?: "",
                    genre = googleBook.volumeInfo.categories?.firstOrNull() ?: "",
                    description = googleBook.volumeInfo.description ?: "",
                    coverUrl = googleBook.volumeInfo.imageLinks?.thumbnail ?: "",
                    averageRating = googleBook.volumeInfo.averageRating ?: 0f,
                    tags = "",
                    language = googleBook.volumeInfo.language ?: "ru",
                    ratingsCount = googleBook.volumeInfo.ratingsCount ?: 0,
                    globalId = globalId
                )
            } ?: return

        val request = UserBookUpdateRequest(
            global_id = globalId,
            title = book.title,
            author = book.author,
            genre = book.genre,
            cover_url = book.coverUrl,
            description = book.description,
            status = status.name.lowercase(),
            rating = _userRatings.value[globalId] ?: 0f
        )
        try {
            val response = NeuralRecommendationApi.retrofit.updateUserBook(userId, request)
            if (response.isSuccessful) {
                loadUserLibrary()
                loadHybridRecommendations()
            } else {
                Log.e("BOOK_VIEWMODEL", "Ошибка добавления книги: ${response.errorBody()?.string()}")
            }
        } catch (e: Exception) {
            Log.e("BOOK_VIEWMODEL", "Ошибка добавления книги", e)
        }
    }

    suspend fun updateBookRating(globalId: String, rating: Float) {

        val book = allBooksCache.find { it.globalId == globalId }
            ?: _readingBooks.value.find { it.globalId == globalId }
            ?: _wantToReadBooks.value.find { it.globalId == globalId }
            ?: _finishedBooks.value.find { it.globalId == globalId }
            ?: _droppedBooks.value.find { it.globalId == globalId }
            ?: return

        val request = UserBookUpdateRequest(
            global_id = globalId,
            title = book.title,
            author = book.author,
            genre = book.genre,
            cover_url = book.coverUrl,
            description = book.description,
            status = "finished",
            rating = rating
        )
        try {
            val response = NeuralRecommendationApi.retrofit.updateUserBook(userId, request)
            if (response.isSuccessful) {
                loadUserLibrary()
                loadHybridRecommendations()
            } else {
                Log.e("BOOK_VIEWMODEL", "Ошибка обновления оценки: ${response.errorBody()?.string()}")
            }
        } catch (e: Exception) {
            Log.e("BOOK_VIEWMODEL", "Ошибка обновления оценки", e)
        }
    }

    suspend fun removeBookFromCollection(globalId: String) {

        val request = UserBookUpdateRequest(
            global_id = globalId,
            title = "",
            author = "",
            genre = "",
            cover_url = "",
            description = "",
            status = "dropped",
            rating = 0f
        )
        try {
            val response = NeuralRecommendationApi.retrofit.updateUserBook(userId, request)
            if (response.isSuccessful) {
                loadUserLibrary()
                loadHybridRecommendations()
            }
        } catch (e: Exception) {
            Log.e("BOOK_VIEWMODEL", "Ошибка удаления книги", e)
        }
    }



    fun loadHybridRecommendations(forceRefresh: Boolean = false) {
        Log.d("VM_RECO", "🔥 loadHybridRecommendations вызван, forceRefresh=$forceRefresh, loading=${_recommendationLoading.value}")

        if (_recommendationLoading.value && !forceRefresh) {
            Log.d("VM_RECO", "⏭️ Пропуск: уже загружается")
            return
        }

        viewModelScope.launch {
            _recommendationLoading.value = true
            Log.d("VM_RECO", "🔄 Начинаем загрузку рекомендаций")

            try {

                val userLibraryGlobalIds = _bookStatusByGlobalId.value.keys.toSet()
                Log.d("VM_RECO", "📚 В библиотеке ${userLibraryGlobalIds.size} книг")

                val allBooks = allBooksCache
                Log.d("VM_RECO", "📖 Всего книг в кэше: ${allBooks.size}")

                val allCandidates = allBooks.filter { it.globalId !in userLibraryGlobalIds }.shuffled()
                Log.d("VM_RECO", "🎯 Кандидатов после фильтрации: ${allCandidates.size}")

                if (allCandidates.isEmpty()) {
                    Log.d("VM_RECO", "⚠️ Нет кандидатов для рекомендаций (все книги уже в библиотеке)")
                    _aiRecommendedBooks.value = emptyList()
                    return@launch
                }

                val candidateBooks = allCandidates.take(100)
                Log.d("VM_RECO", "🎯 Отправляем ${candidateBooks.size} кандидатов на сервер")

                candidateBooks.take(5).forEach { book ->
                    Log.d("VM_RECO", "   Кандидат: ${book.title} (${book.genre})")
                }

                val candidateNeuralBooks = candidateBooks.map { book ->
                    NeuralBookData(
                        id = book.globalId,
                        title = book.title,
                        author = book.author,
                        genre = translateGenreToEnglish(book.genre),
                        tags = if (book.tags.isNotEmpty()) book.tags.split(",") else emptyList(),
                        average_rating = book.averageRating,
                        cover_url = book.coverUrl,
                        description = book.description
                    )
                }
                Log.d("VM_RECO", "📤 Отправляем запрос на сервер...")

                val recommendedNeuralBooks = hybridRepository.getHybridRecommendationsFromServer(
                    userId = userId,
                    candidateBooks = candidateNeuralBooks,
                    limit = 20
                )
                Log.d("VM_RECO", "📥 Получено от сервера: ${recommendedNeuralBooks.size} книг")

                val recommendedBooks = recommendedNeuralBooks
                    .filter { neuralBook ->
                        val globalId = generateGlobalId(neuralBook.title, neuralBook.author)
                        val notInLibrary = globalId !in userLibraryGlobalIds
                        if (!notInLibrary) {
                            Log.d("VM_RECO", "   Пропускаем (уже в библиотеке): ${neuralBook.title}")
                        }
                        notInLibrary
                    }
                    .mapIndexed { index, neuralBook ->

                        val globalId = generateGlobalId(neuralBook.title, neuralBook.author)

                        val localBook = allBooksCache.find {
                            it.globalId == globalId ||
                                    (
                                            it.title.equals(neuralBook.title, ignoreCase = true) &&
                                                    it.author.equals(neuralBook.author, ignoreCase = true)
                                            )
                        }

                        Book(
                            id = -(index + 1).toLong(),
                            userId = userId,
                            title = neuralBook.title,
                            author = neuralBook.author,
                            genre = translateGenreToRussian(neuralBook.genre),
                            description = neuralBook.description,
                            coverUrl = neuralBook.cover_url,
                            averageRating = neuralBook.average_rating,
                            tags = neuralBook.tags.joinToString(","),
                            language = "ru",
                            ratingsCount = 0,
                            globalId = globalId,
                            isBestseller = localBook?.isBestseller ?: false,
                            playlistUrl = localBook?.playlistUrl ?: "",
                            sourceFile = localBook?.sourceFile ?: ""
                        )
                    }

                Log.d("BOOK_VIEWMODEL", "📊 Рекомендации с сервера: ${recommendedNeuralBooks.size}, после фильтрации: ${recommendedBooks.size}")

                if (recommendedBooks.isEmpty()) {
                    Log.d("BOOK_VIEWMODEL", "⚠️ Сервер вернул пустой результат или все книги уже в библиотеке, используем локальные рекомендации")

                    val prefs = _genrePreferences.value
                    val localRecommended = if (prefs.isEmpty()) {
                        Log.d("VM_RECO", "🆕 Используем cold start (нет предпочтений)")
                        getColdStartBooks(userLibraryGlobalIds)
                    } else {
                        Log.d("VM_RECO", "🎨 Используем жанровые предпочтения: $prefs")
                        val scored = allCandidates
                            .map { book ->
                                val prefScore = prefs[book.genre] ?: 0f
                                val finalScore = prefScore * 0.7f + book.averageRating * 0.3f
                                Log.d("VM_RECO", "   ${book.title}: prefScore=$prefScore, avgRating=${book.averageRating}, final=$finalScore")
                                book to finalScore
                            }
                            .sortedByDescending { it.second }
                            .take(20)
                            .map { it.first }
                        scored
                    }

                    val uniqueRecommended = mutableListOf<Book>()
                    val seenGlobalIds = mutableSetOf<String>()
                    for (book in localRecommended) {
                        if (book.globalId !in seenGlobalIds) {
                            seenGlobalIds.add(book.globalId)
                            uniqueRecommended.add(book)
                        }
                    }

                    val shuffledRecommended = if (uniqueRecommended.size > 10) {
                        val top10 = uniqueRecommended.take(10).shuffled()
                        val rest = uniqueRecommended.drop(10)
                        top10 + rest
                    } else {
                        uniqueRecommended.shuffled()
                    }

                    Log.d("VM_RECO", "✅ Локальных рекомендаций: ${shuffledRecommended.size}")
                    _aiRecommendedBooks.value = shuffledRecommended
                } else {

                    val finalRecommendations = if (recommendedBooks.size > 10) {
                        val top10 = recommendedBooks.take(10).shuffled()
                        val rest = recommendedBooks.drop(10)
                        top10 + rest
                    } else {
                        recommendedBooks.shuffled()
                    }
                    Log.d("VM_RECO", "✅ Серверных рекомендаций: ${finalRecommendations.size}")
                    _aiRecommendedBooks.value = finalRecommendations
                }

            } catch (e: Exception) {
                Log.e("VM_RECO", "❌ Ошибка загрузки рекомендаций: ${e.message}", e)
                _aiRecommendedBooks.value = emptyList()
            } finally {
                _recommendationLoading.value = false
                Log.d("VM_RECO", "⏹️ loadHybridRecommendations finished")
            }
        }
    }

    private suspend fun getColdStartBooks(excludeGlobalIds: Set<String>): List<Book> {

        val allBooks = allBooksCache.filter { it.globalId !in excludeGlobalIds }

        // Группируем по жанру
        val byGenre = allBooks.groupBy { it.genre }
        val result = mutableListOf<Book>()

        for ((_, books) in byGenre) {

            val shuffled = books.shuffled().take(5)
            result.addAll(shuffled)
        }

        return result.shuffled().take(20)
    }

    private fun generateGlobalId(title: String, author: String): String {
        return "$title|$author".lowercase().replace(Regex("[^a-zа-я0-9|]"), "")
    }

    private fun updateGenrePreferences() {
        val ratings = _userRatings.value
        val genreScores = mutableMapOf<String, MutableList<Float>>()
        ratings.forEach { (globalId, rating) ->
            val book = allBooksCache.find { it.globalId == globalId }
            book?.let {
                val genre = it.genre
                genreScores.getOrPut(genre) { mutableListOf() }.add(rating)
            }
        }
        val avgScores = genreScores.mapValues { (_, scores) -> scores.average().toFloat() }
        _genrePreferences.value = avgScores
    }

    private fun updateHybridRecommendationQuality() {
        val ratingsCount = _userRatings.value.size
        val quality = when {
            ratingsCount >= 5 -> "высокая"
            ratingsCount >= 2 -> "средняя"
            else -> "базовая"
        }
        _hybridRecommendationQuality.value = quality
        Log.d("BOOK_VIEWMODEL", "📈 Качество гибридных рекомендаций: $quality ($ratingsCount оценок)")
    }

    fun getBookStatus(globalId: String): BookStatus? = _bookStatusByGlobalId.value[globalId]
    fun getCurrentUserId(): Long = userId
    fun forceUpdate() {
        viewModelScope.launch {
            localBooksCache = loadBooksFromLocalJson()
            allBooksCache = localBooksCache
            loadUserLibrary()
            loadHybridRecommendations()
        }
    }

    fun checkHybridServer() {
        viewModelScope.launch {
            try {
                val response = NeuralRecommendationApi.retrofit.checkHealth()
                _serverAvailable.value = response.status == "healthy"
            } catch (e: Exception) {
                _serverAvailable.value = false
            }
        }
    }

    fun translateGenreToRussian(englishGenre: String): String {
        return when (englishGenre.lowercase()) {
            "fantasy" -> "Фэнтези"
            "science fiction" -> "Научная фантастика"
            "romance" -> "Романтика"
            "mystery" -> "Детектив"
            "thriller" -> "Триллер"
            "horror" -> "Ужасы"
            "biography" -> "Биография"
            "history" -> "История"
            "classics" -> "Классика"
            "poetry" -> "Поэзия"
            "drama" -> "Драма"
            "fiction" -> "Художественная литература"
            "adventure" -> "Приключения"
            "children" -> "Детская литература"
            "young adult" -> "Подростковая литература"
            "philosophy" -> "Философия"
            "science" -> "Научная литература"
            "self development" -> "Саморазвитие"
            "business" -> "Бизнес"
            "finance" -> "Финансы"
            "travel" -> "Путешествия"
            else -> englishGenre
        }
    }

    private fun translateGenreToEnglish(russianGenre: String): String {
        return when (russianGenre.lowercase()) {
            "фэнтези" -> "fantasy"
            "научная фантастика" -> "science fiction"
            "романтика" -> "romance"
            "детектив" -> "mystery"
            "триллер" -> "thriller"
            "ужасы" -> "horror"
            "биография" -> "biography"
            "история" -> "history"
            "классика" -> "classics"
            "поэзия" -> "poetry"
            "драма" -> "drama"
            "художественная литература" -> "fiction"
            "приключения" -> "adventure"
            "детская литература" -> "children"
            "подростковая литература" -> "young adult"
            "философия" -> "philosophy"
            "научная литература" -> "science"
            "саморазвитие" -> "self development"
            "бизнес" -> "business"
            "финансы" -> "finance"
            "путешествия" -> "travel"
            else -> "fiction"
        }
    }

    private fun getAvailableGenres(): List<String> {
        return listOf(
            "Фэнтези", "Научная фантастика", "Романтика", "Детектив",
            "Философия",
            "Саморазвитие", "Бизнес", "Финансы"
        )
    }

    companion object {
        fun Factory(context: Context, userId: Long): ViewModelProvider.Factory {
            return object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    return BookViewModel(context, userId) as T
                }
            }
        }
    }

    fun googleBookToLocalBook(googleBook: GoogleBook): Book {

        val cachedBook = allBooksCache.find { it.globalId == googleBook.id }
            ?: allBooksCache.find {
                it.title.equals(googleBook.volumeInfo.title, ignoreCase = true) &&
                        it.author.equals(googleBook.volumeInfo.authors?.firstOrNull(), ignoreCase = true)
            }

        return Book(
            id = 0,
            userId = userId,
            title = googleBook.volumeInfo.title,
            author = googleBook.volumeInfo.authors?.joinToString(", ") ?: "",
            genre = googleBook.volumeInfo.categories?.firstOrNull() ?: "",
            description = googleBook.volumeInfo.description ?: "",
            coverUrl = googleBook.volumeInfo.imageLinks?.thumbnail ?: "",
            averageRating = googleBook.volumeInfo.averageRating ?: 0f,
            tags = "",
            language = googleBook.volumeInfo.language ?: "ru",
            ratingsCount = googleBook.volumeInfo.ratingsCount ?: 0,
            globalId = googleBook.id,

            playlistUrl = cachedBook?.playlistUrl ?: ""
        )
    }

    fun addBookToCollectionSync(globalId: String, status: BookStatus) {
        viewModelScope.launch {
            addBookToCollection(globalId, status)
            loadUserLibrary()
            loadHybridRecommendations()
        }
    }
}

