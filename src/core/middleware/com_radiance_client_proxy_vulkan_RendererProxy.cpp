#include "com_radiance_client_proxy_vulkan_RendererProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/all_extern.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/ui_composite.hpp"
#include "core/render/world.hpp"
#include "core/util/crash_handler.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

static void throwJavaException(JNIEnv *env, const char *className, const char *message) {
    jclass exceptionClass = env->FindClass(className);
    if (exceptionClass == nullptr) return;
    env->ThrowNew(exceptionClass, message);
    env->DeleteLocalRef(exceptionClass);
}

static std::u16string JStringToU16(JNIEnv *env, jstring jstr) {
    if (!jstr) return {};
    const jchar *chars = env->GetStringChars(jstr, nullptr);
    jsize len = env->GetStringLength(jstr);
    std::u16string u16(reinterpret_cast<const char16_t *>(chars), reinterpret_cast<const char16_t *>(chars) + len);
    env->ReleaseStringChars(jstr, chars);
    return u16;
}

static std::filesystem::path JStringToPath(JNIEnv *env, jstring jstr) {
    std::u16string u16 = JStringToU16(env, jstr);
    return std::filesystem::path(u16);
}

static std::vector<std::string> JStringArrayToUtf8(JNIEnv *env, jobjectArray array) {
    std::vector<std::string> result;
    if (array == nullptr) return result;
    const jsize count = env->GetArrayLength(array);
    result.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; i++) {
        auto string = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        if (string == nullptr) continue;
        const char *chars = env->GetStringUTFChars(string, nullptr);
        result.emplace_back(chars);
        env->ReleaseStringUTFChars(string, chars);
        env->DeleteLocalRef(string);
    }
    return result;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initFolderPath(JNIEnv *env,
                                                                                          jclass,
                                                                                          jstring folderPath) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.initFolderPath");
    if (folderPath == NULL) { return; }

    Renderer::folderPath = JStringToU16(env, folderPath);
    MCVR_PROFILE_CONFIGURE_OUTPUT(Renderer::folderPath);
}

// Upscaler ids shared with RendererProxy.UPSCALER_* in Java.
namespace {
constexpr jint UPSCALER_DLSS_RR = 0;
constexpr jint UPSCALER_DLSS_SR = 1;
constexpr jint UPSCALER_DLSS_FRAME_GENERATION = 2;
constexpr jint UPSCALER_XESS = 3;

jobjectArray toJavaStringArray(JNIEnv *env, const std::vector<std::string> &strings) {
    jclass stringClass = env->FindClass("java/lang/String");
    if (stringClass == nullptr) return nullptr;
    jobjectArray array = env->NewObjectArray(static_cast<jsize>(strings.size()), stringClass, nullptr);
    env->DeleteLocalRef(stringClass);
    if (array == nullptr) return nullptr;
    for (jsize i = 0; i < static_cast<jsize>(strings.size()); i++) {
        jstring string = env->NewStringUTF(strings[static_cast<size_t>(i)].c_str());
        if (string == nullptr) return nullptr;
        env->SetObjectArrayElement(array, i, string);
        env->DeleteLocalRef(string);
    }
    return array;
}

std::vector<std::string> extensionNames(const std::vector<VkExtensionProperties> &extensions) {
    std::vector<std::string> names;
    names.reserve(extensions.size());
    for (const auto &extension : extensions) { names.emplace_back(extension.extensionName); }
    return names;
}
} // namespace

// The instance extensions an upscaler SDK needs, or null if the SDK could not say (for example, no NVIDIA driver).
// Called before Minecraft creates its instance, so it must not touch the renderer.
JNIEXPORT jobjectArray JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_queryUpscalerInstanceExtensions(JNIEnv *env, jclass, jint upscaler) {
    try {
        if (upscaler == UPSCALER_XESS) {
            std::vector<const char *> extensions;
            uint32_t minApiVersion = 0;
            if (!mcvr::XeSSWrapper::getRequiredInstanceExtensions(extensions, &minApiVersion)) return nullptr;
            return toJavaStringArray(env, std::vector<std::string>(extensions.begin(), extensions.end()));
        }
        std::vector<VkExtensionProperties> extensions;
        NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
        if (upscaler == UPSCALER_DLSS_RR) {
            result = NgxContext::getDlssRRRequiredInstanceExtensions(extensions);
        } else if (upscaler == UPSCALER_DLSS_SR) {
            result = NgxContext::getDlssSRRequiredInstanceExtensions(extensions);
        } else if (upscaler == UPSCALER_DLSS_FRAME_GENERATION) {
            result = NgxContext::getDlssFrameGenerationRequiredInstanceExtensions(extensions);
        }
        return NVSDK_NGX_SUCCEED(result) ? toJavaStringArray(env, extensionNames(extensions)) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

// The device extensions an upscaler SDK needs on this physical device, or null if the SDK could not say. Called while
// Minecraft picks its device features, before the device exists.
JNIEXPORT jobjectArray JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_queryUpscalerDeviceExtensions(
    JNIEnv *env, jclass, jint upscaler, jlong instance, jlong physicalDevice) {
    auto vkInstance = reinterpret_cast<VkInstance>(static_cast<uintptr_t>(instance));
    auto vkPhysicalDevice = reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(physicalDevice));
    if (vkInstance == VK_NULL_HANDLE || vkPhysicalDevice == VK_NULL_HANDLE) return nullptr;
    try {
        // The SDKs call Vulkan through volk's function pointers, which are only loaded once the renderer attaches.
        if (volkInitialize() != VK_SUCCESS) return nullptr;
        if (volkGetLoadedInstance() != vkInstance) volkLoadInstanceOnly(vkInstance);

        if (upscaler == UPSCALER_XESS) {
            std::vector<const char *> extensions;
            if (!mcvr::XeSSWrapper::getRequiredDeviceExtensions(vkInstance, vkPhysicalDevice, extensions)) {
                return nullptr;
            }
            return toJavaStringArray(env, std::vector<std::string>(extensions.begin(), extensions.end()));
        }
        std::vector<VkExtensionProperties> extensions;
        NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
        if (upscaler == UPSCALER_DLSS_RR) {
            result = NgxContext::getDlssRRRequiredDeviceExtensions(vkInstance, vkPhysicalDevice, extensions);
        } else if (upscaler == UPSCALER_DLSS_SR) {
            result = NgxContext::getDlssSRRequiredDeviceExtensions(vkInstance, vkPhysicalDevice, extensions);
        } else if (upscaler == UPSCALER_DLSS_FRAME_GENERATION) {
            result =
                NgxContext::getDlssFrameGenerationRequiredDeviceExtensions(vkInstance, vkPhysicalDevice, extensions);
        }
        return NVSDK_NGX_SUCCEED(result) ? toJavaStringArray(env, extensionNames(extensions)) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRendererWithVulkanDevice(
    JNIEnv *env,
    jclass,
    jlong instance,
    jlong physicalDevice,
    jlong device,
    jint graphicsQueueFamily,
    jlong graphicsQueue,
    jint computeQueueFamily,
    jlong computeQueue,
    jint transferQueueFamily,
    jlong transferQueue,
    jlong surface,
    jobjectArray enabledInstanceExtensions,
    jobjectArray enabledDeviceExtensions) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.initRendererWithVulkanDevice");
    mcvr::installCrashHandler();
    try {
        VulkanHandles handles;
        handles.instance = reinterpret_cast<VkInstance>(static_cast<uintptr_t>(instance));
        handles.physicalDevice = reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(physicalDevice));
        handles.device = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(device));
        handles.queues.graphicsFamily = static_cast<uint32_t>(graphicsQueueFamily);
        handles.queues.graphics = reinterpret_cast<VkQueue>(static_cast<uintptr_t>(graphicsQueue));
        handles.queues.computeFamily = static_cast<uint32_t>(computeQueueFamily);
        handles.queues.compute = reinterpret_cast<VkQueue>(static_cast<uintptr_t>(computeQueue));
        handles.queues.transferFamily = static_cast<uint32_t>(transferQueueFamily);
        handles.queues.transfer = reinterpret_cast<VkQueue>(static_cast<uintptr_t>(transferQueue));
        // VkSurfaceKHR is a 64-bit handle on every platform.
        handles.surface = reinterpret_cast<VkSurfaceKHR>(static_cast<uintptr_t>(surface));
        handles.enabledInstanceExtensions = JStringArrayToUtf8(env, enabledInstanceExtensions);
        handles.enabledDeviceExtensions = JStringArrayToUtf8(env, enabledDeviceExtensions);
        Renderer::init(std::move(handles));
    } catch (const std::exception &error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "Unknown native failure while attaching to the Vulkan device");
    }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_configureSwapchain(
    JNIEnv *env, jclass, jint width, jint height, jint presentMode) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.configureSwapchain");
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) return JNI_FALSE;
    auto framework = renderer->framework();
    if (framework == nullptr) return JNI_FALSE;
    try {
        return framework->configureSwapchain(static_cast<uint32_t>(std::max(width, 0)),
                                             static_cast<uint32_t>(std::max(height, 0)),
                                             static_cast<uint32_t>(std::max(presentMode, 0))) ?
                   JNI_TRUE :
                   JNI_FALSE;
    } catch (const std::exception &error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_isSwapchainSuboptimal(JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) return JNI_FALSE;
    auto framework = renderer->framework();
    return framework != nullptr && framework->isSwapchainSuboptimal() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_acquireContext(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.acquireContext");
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) return JNI_FALSE;
    auto framework = renderer->framework();
    if (framework == nullptr) return JNI_FALSE;
    try {
        return framework->acquireContext() ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception &error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_setPresentSource(
    JNIEnv *, jclass, jlong vkImage, jlong vkImageView, jint width, jint height) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.setPresentSource");
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) return;
    auto framework = renderer->framework();
    if (framework == nullptr) return;
    framework->setPresentSource(reinterpret_cast<VkImage>(static_cast<uintptr_t>(vkImage)),
                                reinterpret_cast<VkImageView>(static_cast<uintptr_t>(vkImageView)),
                                static_cast<uint32_t>(std::max(width, 0)), static_cast<uint32_t>(std::max(height, 0)));
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_setWorldBlurRadius(JNIEnv *,
                                                                                            jclass,
                                                                                            jfloat radius) {
    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    auto composite = framework == nullptr ? nullptr : framework->uiComposite();
    if (composite != nullptr) composite->setWorldBlurRadius(radius);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_requestScreenshot(JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    auto composite = framework == nullptr ? nullptr : framework->uiComposite();
    if (composite != nullptr) composite->requestScreenshot();
}

// Width in the high 32 bits and height in the low 32 bits, or 0 while no screenshot is waiting.
JNIEXPORT jlong JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_screenshotSize(JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    auto composite = framework == nullptr ? nullptr : framework->uiComposite();
    if (composite == nullptr) return 0;
    VkExtent2D extent = composite->screenshotSize();
    return (static_cast<jlong>(extent.width) << 32) | static_cast<jlong>(extent.height);
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_readScreenshot(
    JNIEnv *, jclass, jlong destination, jint width, jint height) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.readScreenshot");
    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    auto composite = framework == nullptr ? nullptr : framework->uiComposite();
    if (composite == nullptr || width <= 0 || height <= 0) return JNI_FALSE;
    return composite->readScreenshot(reinterpret_cast<void *>(static_cast<uintptr_t>(destination)),
                                     static_cast<uint32_t>(width), static_cast<uint32_t>(height)) ?
               JNI_TRUE :
               JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_submitCommand(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.submitCommand");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->submitCommand();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_present(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.present");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->present();
}

// Vanilla submits to the shared graphics queue without locking, while MCVR also submits to it from
// worker threads (texture uploads, BLAS batches) when there is no separate compute queue. Java wraps
// vanilla's submits and waits in these calls so both sides share MCVR's queue mutex. The device locked
// is remembered per thread, so an unlock always releases what its lock acquired even if the renderer
// closes in between.
namespace {
thread_local std::vector<std::shared_ptr<vk::Device>> lockedQueueDevices;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_lockGraphicsQueue(JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    auto device = framework == nullptr ? nullptr : framework->device();
    if (device != nullptr) device->mainQueueMutex().lock();
    lockedQueueDevices.push_back(device);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_unlockGraphicsQueue(JNIEnv *, jclass) {
    if (lockedQueueDevices.empty()) return;
    auto device = lockedQueueDevices.back();
    lockedQueueDevices.pop_back();
    if (device != nullptr) device->mainQueueMutex().unlock();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_getPresentedFrameRate(JNIEnv *, jclass) {
    auto renderer = Renderer::try_instance();
    if (renderer == nullptr) return 0;
    auto framework = renderer->framework();
    return framework == nullptr ? 0 : static_cast<jint>(framework->presentedFrameRate());
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_close(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.close");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    Renderer::instance().close();
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_shouldRenderWorld(JNIEnv *, jclass, jboolean shouldRenderWorld) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.shouldRenderWorld");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->shouldRender() = shouldRenderWorld;
}
