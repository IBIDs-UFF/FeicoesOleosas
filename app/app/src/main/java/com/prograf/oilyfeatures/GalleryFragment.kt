package com.prograf.oilyfeatures

import android.os.Bundle
import android.provider.MediaStore
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.view.isVisible
import com.bumptech.glide.load.model.stream.MediaStoreVideoThumbLoader
import com.prograf.oilyfeatures.data.Video
import com.prograf.oilyfeatures.databinding.FragmentGalleryBinding
import com.prograf.oilyfeatures.widget.VideosListAdapter
import java.nio.file.Paths


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
            binding.recycler.adapter = VideosListAdapter(videos)
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
                MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.MIME_TYPE,
                MediaStore.Video.Media.RELATIVE_PATH
            ),
            null,
            null,
            MediaStore.MediaColumns.DISPLAY_NAME
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                videos.add(Video(cursor.getString(0)))
            }
        }
    }

}