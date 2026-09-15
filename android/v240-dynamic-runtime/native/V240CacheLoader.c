#include <jni.h>

/*
 * Cache-channel bootstrap native.
 *
 * This library is intentionally inert. Loading it must not initialize any game
 * metadata bridge, inspect Unity runtime metadata, install hooks, spawn threads,
 * touch Unity objects, or call back into Java. Feature-native code is activated
 * only by later staged cache payloads after the stable Java bootstrap has proved
 * crash-loop rollback on the exact v2.4 APK.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void) reserved;
    JNIEnv* env = NULL;
    if (vm == NULL) return JNI_ERR;
    if ((*vm)->GetEnv(vm, (void**) &env, JNI_VERSION_1_6) != JNI_OK || env == NULL) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}
