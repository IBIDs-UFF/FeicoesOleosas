package com.prograf.oilyfeatures

import android.content.ContentValues
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.prograf.oilyfeatures.databinding.FragmentRecordBinding
import java.text.SimpleDateFormat
import java.util.*


class RecordFragment : Fragment() {

    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    private var _binding: FragmentRecordBinding? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentRecordBinding.inflate(inflater, container, false)

        binding.recordStartButton.setOnClickListener { startRecord() }
        binding.recordStopButton.setOnClickListener { stopRecord() }

        setHasOptionsMenu(true)

        return binding.root
    }

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?
    ) {
        super.onViewCreated(view, savedInstanceState)

        (activity as MainActivity).askPermissionsAgain()
        if ((activity as MainActivity).allPermissionsGranted()) {
            binding.recordStartButton.isVisible = true
            binding.permissionNotGrantedText.isVisible = false
            startCamera()
        } else {
            binding.recordStartButton.isVisible = false
            binding.permissionNotGrantedText.isVisible = true
        }
        binding.recordStopButton.isVisible = false
        binding.chronometer.isVisible = false

        requireActivity().onBackPressedDispatcher.addCallback(viewLifecycleOwner, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (recording == null) {
                    findNavController().popBackStack()
                } else {
                    Toast.makeText(requireContext(), getString(R.string.record_in_progress), Toast.LENGTH_LONG).show()
                }
            }
        })
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                if (recording != null) {
                    Toast.makeText(requireContext(), getString(R.string.record_in_progress), Toast.LENGTH_LONG).show()
                    true
                } else {
                    super.onOptionsItemSelected(item)
                }
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(
                    Quality.HIGHEST,
                    FallbackStrategy.higherQualityOrLowerThan(Quality.HD)
                ))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(requireActivity(), CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture)
            } catch (error: Exception) {
                Toast.makeText(requireContext(), getString(R.string.binding_failed, error.message), Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun startRecord() {
        stopRecord()

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, Default.MEDIA_MIME_TYPE)
            put(MediaStore.Video.Media.RELATIVE_PATH, Default.MEDIA_RELATIVE_PATH)
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(requireActivity().contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        recording = videoCapture!!.output
            .prepareRecording(requireContext(), mediaStoreOutputOptions)
            .start(ContextCompat.getMainExecutor(requireContext())) { recordEvent ->
                when (recordEvent) {
                    is VideoRecordEvent.Start -> {
                        binding.chronometer.base = SystemClock.elapsedRealtime()
                        binding.chronometer.start()
                        binding.chronometer.isVisible = true
                        binding.recordStopButton.isVisible = true
                        binding.recordStartButton.isVisible = false
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (recordEvent.hasError()) {
                            recording?.close()
                            recording = null
                            Toast.makeText(requireContext(), getString(R.string.record_failed, recordEvent.cause!!.message), Toast.LENGTH_LONG).show()
                        } else {
                            Toast.makeText(requireContext(), getString(R.string.record_success, name), Toast.LENGTH_LONG).show()
                        }
                        requireActivity().actionBar?.setHomeButtonEnabled(true)
                        binding.recordStartButton.isVisible = true
                        binding.recordStopButton.isVisible = false
                        binding.chronometer.isVisible = false
                        binding.chronometer.stop()
                    }
                }
            }
    }

    private fun stopRecord() {
        recording?.stop()
        recording = null
    }

}