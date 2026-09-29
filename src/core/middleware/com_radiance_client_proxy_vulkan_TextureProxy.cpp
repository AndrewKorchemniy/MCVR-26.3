#include "com_radiance_client_proxy_vulkan_TextureProxy.h"

#include "core/profiler/profiler.hpp"
#include "core/render/emission.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>

static_assert(sizeof(Textures::UploadRegionInfo) ==
              com_radiance_client_proxy_vulkan_TextureProxy_UPLOAD_REGION_SIZE_BYTES);
static_assert(offsetof(Textures::UploadRegionInfo, sourcePointer) == 0);
static_assert(offsetof(Textures::UploadRegionInfo, sourceSizeBytes) == 8);
static_assert(offsetof(Textures::UploadRegionInfo, sourceRowPixels) == 12);
static_assert(offsetof(Textures::UploadRegionInfo, textureId) == 16);
static_assert(offsetof(Textures::UploadRegionInfo, sourceOffsetX) == 20);
static_assert(offsetof(Textures::UploadRegionInfo, sourceOffsetY) == 24);
static_assert(offsetof(Textures::UploadRegionInfo, destinationOffsetX) == 28);
static_assert(offsetof(Textures::UploadRegionInfo, destinationOffsetY) == 32);
static_assert(offsetof(Textures::UploadRegionInfo, width) == 36);
static_assert(offsetof(Textures::UploadRegionInfo, height) == 40);
static_assert(offsetof(Textures::UploadRegionInfo, level) == 44);
static_assert(offsetof(Textures::UploadRegionInfo, uploadKind) == 48);
static_assert(offsetof(Textures::UploadRegionInfo, sourceBytesPerPixel) == 52);

namespace {

void throwJavaTextureException(JNIEnv *env, const char *message) noexcept {
    if (env == nullptr || env->ExceptionCheck()) { return; }
    jclass exceptionClass = env->FindClass("java/lang/IllegalStateException");
    if (exceptionClass == nullptr) { return; }
    env->ThrowNew(exceptionClass, message);
    env->DeleteLocalRef(exceptionClass);
}

template <typename Function>
void invokeTextureJni(JNIEnv *env, Function &&function) noexcept {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception &error) {
        throwJavaTextureException(env, error.what());
    } catch (...) {
        throwJavaTextureException(env, "Unknown native texture error");
    }
}

std::shared_ptr<Textures> currentTextures() {
    auto *renderer = Renderer::try_instance();
    return renderer == nullptr ? nullptr : renderer->textures();
}

void requireTextureId(jint id) {
    if (id < 0) { throw std::invalid_argument("Texture id must be non-negative"); }
}

}

extern "C" {
JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_generateTextureIdNative(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.generateTextureId");
    jint result = 0;
    invokeTextureJni(env, [&] {
        auto textures = currentTextures();
        if (textures != nullptr) { result = static_cast<jint>(textures->allocateTexture()); }
    });
    return result;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_prepareImageNative(JNIEnv *env,
                                                                                             jclass,
                                                                                             jint id,
                                                                                             jint maxLevel,
                                                                                             jint width,
                                                                                             jint height,
                                                                                             jint format,
                                                                                             jboolean clear,
                                                                                             jfloat clearRed,
                                                                                             jfloat clearGreen,
                                                                                             jfloat clearBlue,
                                                                                             jfloat clearAlpha) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.prepareImage");
    invokeTextureJni(env, [&] {
        requireTextureId(id);
        if (maxLevel <= 0 || width <= 0 || height <= 0) {
            throw std::invalid_argument("Texture dimensions and mip level count must be positive");
        }
        auto textures = currentTextures();
        if (textures == nullptr) return;
        auto vkFormat = static_cast<VkFormat>(format);
        textures->initializeTexture(static_cast<uint32_t>(id), static_cast<uint32_t>(maxLevel),
                                    static_cast<uint32_t>(width), static_cast<uint32_t>(height), vkFormat);
        if (clear) {
            VkClearColorValue clearColor = {{clearRed, clearGreen, clearBlue, clearAlpha}};
            textures->queueTextureClear(static_cast<uint32_t>(id), clearColor);
        }
        if (auto emission = textures->emission(); emission != nullptr) {
            emission->resetTexture(static_cast<uint32_t>(id));
        }
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_releaseTextureNative(JNIEnv *env,
                                                                                               jclass,
                                                                                               jint id) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.releaseTexture");
    invokeTextureJni(env, [&] {
        requireTextureId(id);
        auto textures = currentTextures();
        if (textures != nullptr) { textures->releaseTexture(static_cast<uint32_t>(id)); }
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_queueTextureClearNative(
    JNIEnv *env, jclass, jint id, jfloat clearRed, jfloat clearGreen, jfloat clearBlue, jfloat clearAlpha) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.queueTextureClear");
    invokeTextureJni(env, [&] {
        requireTextureId(id);
        auto textures = currentTextures();
        if (textures == nullptr) return;
        VkClearColorValue clearColor = {{clearRed, clearGreen, clearBlue, clearAlpha}};
        textures->queueTextureClear(static_cast<uint32_t>(id), clearColor);
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_setFilterNative(
    JNIEnv *env, jclass, jint id, jint samplingMode, jint mipmapMode) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.setFilter");
    invokeTextureJni(env, [&] {
        requireTextureId(id);
        auto textures = currentTextures();
        if (textures == nullptr) return;
        auto vkSamplingMode = static_cast<VkFilter>(samplingMode);
        auto vkMipmapMode = static_cast<VkSamplerMipmapMode>(mipmapMode);
        textures->setSamplingMode(static_cast<uint32_t>(id), vkSamplingMode, vkMipmapMode);
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_setClampNative(JNIEnv *env,
                                                                                         jclass,
                                                                                         jint id,
                                                                                         jint addressMode) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.setClamp");
    invokeTextureJni(env, [&] {
        requireTextureId(id);
        auto textures = currentTextures();
        if (textures == nullptr) return;
        auto vkSamplerAddressMode = static_cast<VkSamplerAddressMode>(addressMode);
        textures->setAddressMode(static_cast<uint32_t>(id), vkSamplerAddressMode);
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_queueUploadsNative(JNIEnv *env,
                                                                                             jclass,
                                                                                             jlong regionsPtr,
                                                                                             jint regionCount) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.queueUploads");
    invokeTextureJni(env, [&] {
        if (regionsPtr == 0 || regionCount <= 0) return;
        if (regionCount > 16'384) { throw std::invalid_argument("Too many texture upload regions"); }
        auto textures = currentTextures();
        if (textures == nullptr) return;
        textures->queueUploads(reinterpret_cast<const Textures::UploadRegionInfo *>(regionsPtr),
                               static_cast<uint32_t>(regionCount));
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_submitEmissionUpdateNative(
    JNIEnv *env, jclass, jint textureId, jlong tileKey, jlong cellsPtr, jint cellCount) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.submitEmissionUpdateNative");
    invokeTextureJni(env, [&] {
        requireTextureId(textureId);
        if (cellCount < 0 || (cellCount > 0 && cellsPtr == 0)) {
            throw std::invalid_argument("Invalid emission upload buffer");
        }
        auto textures = currentTextures();
        if (textures == nullptr) return;
        auto emission = textures->emission();
        if (emission == nullptr) return;

        emission->updateTile(static_cast<uint32_t>(textureId), static_cast<uint64_t>(tileKey),
                             reinterpret_cast<const EmissionCellUpload *>(cellsPtr), cellCount);
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_performQueuedUpload(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.TextureProxy.performQueuedUpload");
    invokeTextureJni(env, [&] {
        auto textures = currentTextures();
        if (textures != nullptr) { textures->performQueuedUpload(); }
    });
}
}
