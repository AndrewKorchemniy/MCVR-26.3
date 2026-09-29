#if defined(MCVR_ENABLE_PROFILER)

#    include "core/profiler/profiler.hpp"

#    include <jni.h>

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_radiance_client_profiler_Profiler_nativeIsAvailable(JNIEnv *, jclass) {
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_Profiler_nativeEnter(JNIEnv *env, jclass, jstring name) {
    if (name == nullptr) {
        mcvr::profiler::Profiler::enter("java.<null>");
        return;
    }

    const char *chars = env->GetStringUTFChars(name, nullptr);
    mcvr::profiler::Profiler::enter(chars == nullptr ? "java.<unreadable>" : chars);
    if (chars != nullptr) { env->ReleaseStringUTFChars(name, chars); }
}

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_Profiler_nativeExit(JNIEnv *, jclass) {
    mcvr::profiler::Profiler::exit();
}

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_Profiler_nativeMetric(JNIEnv *env,
                                                                               jclass,
                                                                               jstring name,
                                                                               jlong value) {
    if (name == nullptr) {
        mcvr::profiler::Profiler::metric("java.<null>", static_cast<int64_t>(value));
        return;
    }

    const char *chars = env->GetStringUTFChars(name, nullptr);
    mcvr::profiler::Profiler::metric(chars == nullptr ? "java.<unreadable>" : chars, static_cast<int64_t>(value));
    if (chars != nullptr) { env->ReleaseStringUTFChars(name, chars); }
}

JNIEXPORT void JNICALL Java_com_radiance_client_profiler_Profiler_nativeFlush(JNIEnv *, jclass) {
    mcvr::profiler::Profiler::flush();
}
}

#endif
