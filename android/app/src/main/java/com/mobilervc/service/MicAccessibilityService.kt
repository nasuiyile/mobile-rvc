package com.mobilervc.service

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent

/**
 * Does nothing with the UI. Its only purpose: while an accessibility service of this app is
 * enabled, Android's audio policy treats microphone capture by this app as "accessibility"
 * capture, which may run concurrently with the foreground app's recording instead of being
 * silenced by it (AudioPolicyService::updateUidStates_l). That is what lets the virtual
 * microphone work for WeChat/QQ voice messages.
 */
class MicAccessibilityService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    override fun onServiceConnected() {
        connected = true
    }

    override fun onDestroy() {
        connected = false
        super.onDestroy()
    }

    companion object {
        @Volatile var connected = false
            private set
    }
}
