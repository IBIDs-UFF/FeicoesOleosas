package com.prograf.oilyfeatures.widget

import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.IntentSenderRequest
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.prograf.oilyfeatures.MainActivity
import com.prograf.oilyfeatures.R
import com.prograf.oilyfeatures.data.Video
import com.prograf.oilyfeatures.utils.UploadUtility
import org.apache.commons.io.FileUtils
import java.util.concurrent.TimeUnit


class VideosListAdapter(
        private val activity: MainActivity,
        val videos: MutableList<Video>
    ) : RecyclerView.Adapter<VideosListAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val nameText: TextView = view.findViewById(R.id.name_text)
        val durationText: TextView = view.findViewById(R.id.duration_text)
        val sizeText: TextView = view.findViewById(R.id.size_text)
        val thumbnailView: ImageView = view.findViewById(R.id.thumbnail_view)
        val sendButton: ImageButton = view.findViewById(R.id.send_button)
        val deleteButton: ImageButton = view.findViewById(R.id.delete_button)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.fragment_galery_item, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val context = holder.itemView.context
        val video = videos[position]
        // Show video info.
        val minutes = TimeUnit.MILLISECONDS.toMinutes(video.duration)
        val seconds = TimeUnit.MILLISECONDS.toSeconds(video.duration) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(video.duration))
        holder.nameText.text = video.name
        holder.durationText.text = context.getString(R.string.duration_format, minutes, seconds)
        holder.sizeText.text = context.getString(R.string.size_format, FileUtils.byteCountToDisplaySize(video.size))
        Glide.with(context)
            .load(video.uri)
            .placeholder(R.drawable.ic_video_place_holder)
            .override(320)
            .into(holder.thumbnailView)
        // Set controls.
        holder.sendButton.setOnClickListener {
            UploadUtility(activity).uploadFile(video.uri, activity.serverURL())
        }
        holder.deleteButton.setOnClickListener {
            activity.galleryDeleteFromAdapter = this
            activity.galleryDeletePosition = videos.indexOf(video)
            val pendingIntent = MediaStore.createDeleteRequest(context.contentResolver, listOf(video.uri))
            activity.galleryDeleteLauncher.launch(
                IntentSenderRequest.Builder(pendingIntent.intentSender).build()
            )
        }
    }

    override fun getItemCount(): Int = videos.size

}