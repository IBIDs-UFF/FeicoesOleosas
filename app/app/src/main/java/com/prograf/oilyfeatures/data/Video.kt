package com.prograf.oilyfeatures.data

import android.net.Uri


data class Video(
    val name: String,
    val duration: Long, // Milliseconds.
    val size: Int, // Bytes.
    val uri: Uri
)