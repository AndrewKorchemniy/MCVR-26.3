#include <jni.h>

#include "core/middleware/jni_export.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/opengl_ui/opengl_ui_surface.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#endif

namespace {
jlongArray makeLongArray(JNIEnv *env, const std::vector<jlong> &values) {
    jlongArray result = env->NewLongArray(static_cast<jsize>(values.size()));
    if (result == nullptr) { return nullptr; }
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
    return result;
}
}

extern "C" {
JNIEXPORT jlongArray JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_prepareFrame(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.OpenGLUiProxy.prepareFrame");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) { return makeLongArray(env, {}); }
    auto surface = framework->prepareOpenGLUiFrame();
    if (surface == nullptr) { return makeLongArray(env, {}); }
    return makeLongArray(env, {static_cast<jlong>(surface->generation()), static_cast<jlong>(surface->frameIndex()),
                               static_cast<jlong>(surface->width()), static_cast<jlong>(surface->height())});
}

JNIEXPORT jlongArray JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_exportSurface(JNIEnv *env,
                                                                                             jclass,
                                                                                             jint frameIndex) {
    MCVR_PROFILE_SCOPE("jni.OpenGLUiProxy.exportSurface");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) { return makeLongArray(env, {}); }
    auto exportInfo = framework->exportOpenGLUiSurface(static_cast<uint32_t>(frameIndex));
    return makeLongArray(env, {static_cast<jlong>(exportInfo.generation), static_cast<jlong>(exportInfo.frameIndex),
                               static_cast<jlong>(exportInfo.width), static_cast<jlong>(exportInfo.height),
                               static_cast<jlong>(exportInfo.memorySize), static_cast<jlong>(exportInfo.memoryHandle),
                               static_cast<jlong>(exportInfo.vulkanReadySemaphoreHandle),
                               static_cast<jlong>(exportInfo.glCompleteSemaphoreHandle)});
}

JNIEXPORT void JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_releaseWin32Handles(
    JNIEnv *, jclass, jlong memoryHandle, jlong vulkanReadySemaphoreHandle, jlong glCompleteSemaphoreHandle) {
#if defined(_WIN32)
    const jlong handles[] = {memoryHandle, vulkanReadySemaphoreHandle, glCompleteSemaphoreHandle};
    for (jlong handleValue : handles) {
        if (handleValue != 0) {
            CloseHandle(reinterpret_cast<HANDLE>(static_cast<uintptr_t>(handleValue)));
        }
    }
#endif
}

JNIEXPORT void JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_submitFrame(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.OpenGLUiProxy.submitFrame");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) { return; }
    framework->submitOpenGLUiFrame();
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_hasResizePending(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    return framework != nullptr && framework->hasOpenGLUiResizePending() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_hasActiveFrame(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    return framework != nullptr && framework->hasOpenGLUiFrame() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_prepareResize(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    return framework == nullptr || framework->prepareOpenGLUiResize() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_opengl_1ui_OpenGLUiProxy_completeResize(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework != nullptr) { framework->completeOpenGLUiResize(); }
}
}
