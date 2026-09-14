#include <jni.h>
#include <cstddef>
#include <string>

extern "C" std::size_t V240CopyPostV240CompatibilityReport(
        char* buffer, std::size_t bufferSize);

extern "C" JNIEXPORT jstring JNICALL
Java_com_unity3d_player_V240CompatibilityReport_nativeGetCompatibilityReport(
        JNIEnv* env, jclass) {
    if (!env) return nullptr;

    const std::size_t length = V240CopyPostV240CompatibilityReport(nullptr, 0);
    std::string report(length + 1, '\0');
    V240CopyPostV240CompatibilityReport(report.data(), report.size());
    return env->NewStringUTF(report.c_str());
}
