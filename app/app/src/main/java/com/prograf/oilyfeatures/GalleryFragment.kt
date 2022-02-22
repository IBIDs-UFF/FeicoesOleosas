package com.prograf.oilyfeatures

import android.content.ContentUris
import android.os.Bundle
import android.provider.MediaStore
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.view.isVisible
import com.prograf.oilyfeatures.data.Video
import com.prograf.oilyfeatures.databinding.FragmentGalleryBinding
import com.prograf.oilyfeatures.widget.VideosListAdapter


class GalleryFragment : Fragment() {

    private var _binding: FragmentGalleryBinding? = null

    private val videos: MutableList<Video> = mutableListOf()

    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentGalleryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?
    ) {
        super.onViewCreated(view, savedInstanceState)

        (activity as MainActivity).askPermissionsAgain()
        if ((activity as MainActivity).allPermissionsGranted()) {
            loadVideosList()
            binding.recycler.adapter = VideosListAdapter(requireActivity() as MainActivity, videos)
            binding.recycler.isVisible = true
            binding.permissionNotGrantedText.isVisible = false
        } else {
            binding.recycler.isVisible = false
            binding.permissionNotGrantedText.isVisible = true
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun loadVideosList() {
        videos.clear()
        requireContext().contentResolver.query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            arrayOf(
                MediaStore.MediaColumns._ID,
                MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.DURATION,
                MediaStore.MediaColumns.SIZE,
            ),
            "${MediaStore.Video.Media.RELATIVE_PATH} like ? and ${MediaStore.MediaColumns.MIME_TYPE} like ?",
            arrayOf(Default.MEDIA_RELATIVE_PATH, Default.MEDIA_MIME_TYPE),
            MediaStore.MediaColumns.DISPLAY_NAME
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME)
            val durationColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION)
            val sizeColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.SIZE)
            while (cursor.moveToNext()) {
                videos.add(Video(
                    name = cursor.getString(nameColumn),
                    duration = cursor.getLong(durationColumn),
                    size = cursor.getInt(sizeColumn),
                    uri = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, cursor.getLong(idColumn))
                ))
            }
        }
    }

}