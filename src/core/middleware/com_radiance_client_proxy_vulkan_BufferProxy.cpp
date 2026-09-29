#include "com_radiance_client_proxy_vulkan_BufferProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

JNIEXPORT jlong JNICALL
Java_com_radiance_client_proxy_vulkan_BufferProxy_createAndUploadVertexIndexBufferNative(JNIEnv *,
                                                                                         jclass,
                                                                                         jlong vertexAddress,
                                                                                         jint vertexSize,
                                                                                         jlong sortedIndexAddress,
                                                                                         jint indexSize,
                                                                                         jint indexType,
                                                                                         jint drawMode,
                                                                                         jint vertexCount,
                                                                                         jint indexCount) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.createAndUploadVertexIndexBuffer");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) { return 0; }

    const auto [vertexId, indexId] = buffers->createAndUploadVertexIndexBuffers(
        reinterpret_cast<uint8_t *>(vertexAddress), static_cast<uint32_t>(vertexSize),
        reinterpret_cast<uint8_t *>(sortedIndexAddress), static_cast<uint32_t>(indexSize), indexType, drawMode,
        vertexCount, indexCount);
    return static_cast<jlong>((static_cast<uint64_t>(vertexId) << 32u) | static_cast<uint64_t>(indexId));
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_performQueuedUploadNative(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.performQueuedUpload");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    buffers->performQueuedUpload();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateOverlayPostUniformNative(JNIEnv *,
                                                                                                        jclass,
                                                                                                        jlong ptr) {
    MCVR_PROFILE_SCOPE("jni.BufferProxy.updateOverlayPostUniform");
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::OverlayPostUBO *ubo = reinterpret_cast<vk::Data::OverlayPostUBO *>(ptr);

    auto extent = Renderer::instance().framework()->swapchain()->vkExtent();
    ubo->inSize = {extent.width, extent.height};
    ubo->outSize = {extent.width, extent.height};

    ubo->blurDir = {1.0, 0.0};
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {1.0, 0.0};
    ubo->radiusMultiplier = 0.5;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    ubo->radiusMultiplier = 0.5;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {1.0, 0.0};
    ubo->radiusMultiplier = 0.25;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    ubo->radiusMultiplier = 0.25;
    buffers->appendOverlayPostUniform(*ubo);
}

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
