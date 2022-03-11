package com.prograf.oilyfeatures.utils

import android.app.Activity
import android.app.ProgressDialog
import android.net.Uri
import android.webkit.MimeTypeMap
import android.widget.Toast
import com.prograf.oilyfeatures.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.concurrent.TimeUnit


class UploadUtility(
        private var activity: Activity
    ) {

    private var dialog: ProgressDialog? = null
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    fun uploadFile(sourceFileUri: Uri, serverURL: String) {
        val sourceFile = File(URIPathHelper().getPath(activity, sourceFileUri)!!)
        Thread {
            val mimeType = getMimeType(sourceFile)
            if (mimeType == null) {
                showToast(activity.getString(R.string.not_able_to_get_mime_type))
                return@Thread
            }
            toggleProgressDialog(true)
            try {
                val requestBody = MultipartBody.Builder().setType(MultipartBody.FORM)
                    .addFormDataPart("file", sourceFile.name, sourceFile.asRequestBody(mimeType.toMediaTypeOrNull()))
                    .build()
                val request = Request.Builder().url(serverURL).post(requestBody).build()
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    showToast(activity.getString(R.string.file_uploaded_successfully))
                } else {
                    showToast(activity.getString(R.string.file_uploading_error, response.message))
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
                showToast(activity.getString(R.string.file_uploading_error, ex.message))
            }
            finally {
                toggleProgressDialog(false)
            }
        }.start()
    }

    private fun getMimeType(file: File): String? {
        var type: String? = null
        val extension = MimeTypeMap.getFileExtensionFromUrl(file.path)
        if (extension != null) {
            type = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension)
        }
        return type
    }

    private fun showToast(message: String) {
        activity.runOnUiThread {
            Toast.makeText(activity, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun toggleProgressDialog(show: Boolean) {
        activity.runOnUiThread {
            if (show) {
                dialog = ProgressDialog.show(activity, "", activity.getString(R.string.file_uploading_message), true)
            } else {
                dialog?.dismiss()
            }
        }
    }

}
