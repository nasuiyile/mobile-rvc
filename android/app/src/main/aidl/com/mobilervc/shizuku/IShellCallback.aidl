package com.mobilervc.shizuku;

// Who should capture the microphone right now:
//   0 = nobody (no app is recording), 1 = the app (accessibility capture, for voice messages),
//   2 = the shell service (privileged capture, for calls / privacy-sensitive recordings)
oneway interface IShellCallback {
    void onCaptureMode(int mode, String detail);
}
