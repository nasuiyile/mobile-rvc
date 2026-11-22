# JNI entry points
-keep class com.mobilervc.engine.Native { *; }
# Instantiated by Shizuku by class name inside the shell process
-keep class com.mobilervc.shizuku.ShellAudioService { <init>(...); *; }
-keep class com.mobilervc.shizuku.IShellAudio* { *; }
# Reflection on framework audio policy classes needs nothing kept on our side
-dontwarn org.lsposed.hiddenapibypass.**
# ONNX Runtime Java classes are packaged by the AAR but unused
-dontwarn ai.onnxruntime.**
-keep class com.mobilervc.shizuku.ShellSelfTest { public static void main(java.lang.String[]); }
-keep class com.mobilervc.shizuku.MicProbe { public static void main(java.lang.String[]); }
