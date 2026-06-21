package com.samira.bookapp.network

import com.google.gson.annotations.SerializedName

data class GoogleBook(
    @SerializedName("id") val id: String,
    @SerializedName("volumeInfo") val volumeInfo: VolumeInfo
)

data class VolumeInfo(
    @SerializedName("title") val title: String,
    @SerializedName("authors") val authors: List<String>? = emptyList(),
    @SerializedName("publishedDate") val publishedDate: String? = null,
    @SerializedName("description") val description: String? = null,
    @SerializedName("pageCount") val pageCount: Int? = null,
    @SerializedName("categories") val categories: List<String>? = emptyList(),
    @SerializedName("averageRating") val averageRating: Float? = null,
    @SerializedName("ratingsCount") val ratingsCount: Int? = null,
    @SerializedName("imageLinks") val imageLinks: ImageLinks? = null,
    @SerializedName("language") val language: String? = null
)

data class ImageLinks(
    @SerializedName("smallThumbnail") val smallThumbnail: String? = null,
    @SerializedName("thumbnail") val thumbnail: String? = null,
    @SerializedName("small") val small: String? = null,
    @SerializedName("medium") val medium: String? = null,
    @SerializedName("large") val large: String? = null,
    @SerializedName("extraLarge") val extraLarge: String? = null
)