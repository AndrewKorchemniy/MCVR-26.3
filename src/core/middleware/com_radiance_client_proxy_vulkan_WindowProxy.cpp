#include "com_radiance_client_proxy_vulkan_WindowProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/all_extern.hpp"
#include "core/profiler/profiler.hpp"
#include "core/vulkan/window.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_WindowProxy_onFramebufferSizeChanged(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.WindowProxy.onFramebufferSizeChanged");
    vk::Window::framebufferResized = true;
}
