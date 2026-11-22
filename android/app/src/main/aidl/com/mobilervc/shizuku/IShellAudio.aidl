package com.mobilervc.shizuku;

import com.mobilervc.shizuku.IShellCallback;

// Runs as the shell user (uid 2000) via Shizuku. Redirects other apps' recordings to an
// AudioPolicy loop-back mix fed with converted audio, and captures the real microphone only
// while such a recording is active. Audio travels over pipes as float32 mono PCM.
interface IShellAudio {
    // Shizuku UserService convention: called when the service is unbound/stopped.
    void destroy() = 16777114;

    // captureOut: shell writes microphone audio here (only in shell-capture mode);
    // playbackIn: shell reads converted audio. callback: capture-mode changes.
    // Returns a JSON status string ({"error": ...} on failure).
    String start(in ParcelFileDescriptor captureOut, in ParcelFileDescriptor playbackIn, String configJson,
                 IShellCallback callback) = 1;
    void stop() = 2;
    String status() = 3;
    int uid() = 4;
    // Enables the given accessibility service component (keeps the ones already enabled).
    String ensureAccessibility(String component) = 5;
}
