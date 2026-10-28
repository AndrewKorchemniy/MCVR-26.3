#include "com_radiance_client_proxy_vulkan_BufferProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateWorldUniformNative(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jlong ptr) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.updateWorldUniform");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::WorldUBO *ubo = reinterpret_cast<vk::Data::WorldUBO *>(ptr);
    buffers->setAndUploadWorldUniformBuffer(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateSkyUniformNative(JNIEnv *,
                                                                                                jclass,
                                                                                                jlong ptr) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.updateSkyUniform");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::SkyUBO *ubo = reinterpret_cast<vk::Data::SkyUBO *>(ptr);
    buffers->setAndUploadSkyUniformBuffer(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateMappingNative(JNIEnv *,
                                                                                             jclass,
                                                                                             jlong ptr) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.updateMapping");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::TextureMapping *mapping = reinterpret_cast<vk::Data::TextureMapping *>(ptr);
    buffers->setAndUploadTextureMappingBuffer(*mapping);
}
