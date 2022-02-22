package com.prograf.oilyfeatures.widget

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.prograf.oilyfeatures.R
import com.prograf.oilyfeatures.data.Video


class VideosListAdapter(
        private val videos: MutableList<Video>
    ) : RecyclerView.Adapter<VideosListAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val displayNameText: TextView = view.findViewById(R.id.display_name_text)
        val thumbnailView: ImageView = view.findViewById(R.id.thumbnail_view)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.fragment_galery_item, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val video = videos[position]
        holder.displayNameText.text = video.displayName
        Glide.with(holder.itemView.context)
            .load(video.displayName)
            .placeholder(R.drawable.ic_video_place_holder)
            .into(holder.thumbnailView)
    }

    override fun getItemCount(): Int = videos.size

}