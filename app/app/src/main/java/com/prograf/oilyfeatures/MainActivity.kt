package com.prograf.oilyfeatures

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import com.prograf.oilyfeatures.data.DefaultSettings
import com.prograf.oilyfeatures.databinding.ActivityMainBinding
import com.prograf.oilyfeatures.widget.VideosListAdapter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10

        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.INTERNET,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
        )
    }

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var settings: SharedPreferences

    lateinit var galleryDeleteLauncher: ActivityResultLauncher<IntentSenderRequest>
    var galleryDeleteFromAdapter: VideosListAdapter? = null
    var galleryDeletePosition: Int? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        settings = getSharedPreferences("Settings", Context.MODE_PRIVATE)
        if (settings.getBoolean("FirstTime", true)) {
            val dialog = SettingsDialogFragment(settings)
            dialog.show(supportFragmentManager, null)
            settings.edit().apply {
                putBoolean("FirstTime", false)
                apply()
            }
        }

        galleryDeleteLauncher = registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) {
            if (it.resultCode == RESULT_OK) {
                Toast.makeText(this, getString(R.string.video_deleted_successfully), Toast.LENGTH_LONG).show()
                galleryDeleteFromAdapter?.let { adapter ->
                    adapter.videos.removeAt(galleryDeletePosition!!)
                    adapter.notifyItemRemoved(galleryDeletePosition!!)
                }
            } else {
                Toast.makeText(this, getString(R.string.action_canceled), Toast.LENGTH_LONG).show()
            }
            galleryDeleteFromAdapter = null
            galleryDeletePosition = null
        }

        setSupportActionBar(binding.toolbar)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)
        setupActionBarWithNavController(navController, appBarConfiguration)

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                val dialog = SettingsDialogFragment(settings)
                dialog.show(supportFragmentManager, null)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (!allPermissionsGranted()) {
                Toast.makeText(this, R.string.permissions_not_granted, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }

    fun askPermissionsAgain() {
        if (!allPermissionsGranted())
        {
            val alertDialogBuilder: AlertDialog.Builder = AlertDialog.Builder(this)
            alertDialogBuilder.setTitle(R.string.permissions_needed_title)
            alertDialogBuilder.setMessage(R.string.permissions_needed_message)
            alertDialogBuilder.setPositiveButton(R.string.open_settings) { _, _ ->
                val intent = Intent()
                intent.action = Settings.ACTION_APPLICATION_DETAILS_SETTINGS
                intent.data = Uri.fromParts("package", this.packageName, null)
                this.startActivity(intent)
            }
            alertDialogBuilder.setNegativeButton(R.string.cancel) { _, _ ->
            }

            val dialog: AlertDialog = alertDialogBuilder.create()
            dialog.show()
        }
    }

    fun allPermissionsGranted(): Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false
            }
        }
        return true
    }

    fun serverURL(): String {
        val protocol = settings.getString(DefaultSettings.PROTOCOL_FIELD, DefaultSettings.PROTOCOL_VALUE)!!.lowercase()
        val address = settings.getString(DefaultSettings.ADDRESS_FIELD, DefaultSettings.ADDRESS_VALUE)!!
        val port = settings.getString(DefaultSettings.PORT_FIELD, DefaultSettings.PORT_VALUE.toString())!!.toIntOrNull()
        return if (port != null) "${protocol}://${address}:${port}" else "${protocol}://${address}"
    }

}