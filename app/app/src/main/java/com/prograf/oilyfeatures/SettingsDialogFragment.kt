package com.prograf.oilyfeatures

import android.app.AlertDialog
import android.app.Dialog
import android.content.SharedPreferences
import android.os.Bundle
import android.view.KeyEvent
import android.widget.Spinner
import androidx.fragment.app.DialogFragment
import com.google.android.material.textfield.TextInputEditText
import com.prograf.oilyfeatures.data.DefaultSettings


class SettingsDialogFragment(
        private val settings: SharedPreferences
    ) : DialogFragment() {

    private lateinit var protocolSpinner: Spinner
    private lateinit var addressEditText: TextInputEditText
    private lateinit var portEditText: TextInputEditText

    override fun onCreateDialog(
        savedInstanceState: Bundle?
    ): Dialog {
        val activity = requireActivity()
        val inflater = activity.layoutInflater
        val view = inflater.inflate(R.layout.fragment_settings_dialog, null)
        val protocolItems = activity.resources.getStringArray(R.array.protocol_items)

        protocolSpinner = view.findViewById(R.id.spinner_protocol)
        protocolSpinner.setSelection(protocolItems.indexOf(settings.getString(DefaultSettings.PROTOCOL_FIELD, DefaultSettings.PROTOCOL_VALUE)))

        addressEditText = view.findViewById(R.id.edit_text_address)
        addressEditText.setText(settings.getString(DefaultSettings.ADDRESS_FIELD, DefaultSettings.ADDRESS_VALUE))
        addressEditText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                if (addressEditText.text.isNullOrBlank())
                    addressEditText.error = getString(R.string.settings_invalid_address)
                updatePositiveButton()
            }
        }

        portEditText = view.findViewById(R.id.edit_text_port)
        portEditText.setText(settings.getString(DefaultSettings.PORT_FIELD, DefaultSettings.PORT_VALUE.toString()))
        portEditText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                val port = portEditText.text.toString().toIntOrNull()
                if (port != null && port <= 0)
                    portEditText.error = getString(R.string.settings_invalid_port)
                updatePositiveButton()
            }
        }

        val builder = AlertDialog.Builder(activity)
            .setView(view)
            .setPositiveButton(R.string.save) { _, _ ->
                settings.edit().apply {
                    putString("Protocol", protocolSpinner.selectedItem.toString())
                    putString("Address", addressEditText.text.toString())
                    putString("Port", portEditText.text.toString())
                    apply()
                }
            }
            .setNegativeButton(R.string.cancel) { _, _ -> }
            .setOnKeyListener { dialog, keyCode, keyEvent ->
                if (keyCode == KeyEvent.KEYCODE_BACK && keyEvent.action == KeyEvent.ACTION_UP && !keyEvent.isCanceled) {
                    dialog.cancel()
                    true
                } else {
                    false
                }
            }

        val dialog = builder.create()
        dialog.setCancelable(false)
        dialog.setCanceledOnTouchOutside(false)
        dialog.setOnShowListener { updatePositiveButton() }
        return dialog
    }

    private fun updatePositiveButton() {
        val positiveButton = (dialog as AlertDialog).getButton(AlertDialog.BUTTON_POSITIVE)
        positiveButton.isEnabled = protocolSpinner.selectedItemPosition != -1 && addressEditText.error.isNullOrBlank() && portEditText.error.isNullOrBlank()
    }

}