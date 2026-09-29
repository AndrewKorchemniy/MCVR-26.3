#include "com_radiance_client_option_Options.h"
#include "core/middleware/jni_export.hpp"

#include "core/all_extern.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <filesystem>
#include <mutex>

class FrameGenerationCapabilityQuery {
  public:
    static constexpr int FrameGenerationOff = 0;
    static constexpr int FrameGenerationDlss = 1;
    static constexpr int FrameGenerationFsr = 2;
    static constexpr int FrameGenerationModeMaskOff = 1 << FrameGenerationOff;
    static constexpr int FrameGenerationModeMaskDlss = 1 << FrameGenerationDlss;
    static constexpr int FrameGenerationModeMaskFsr = 1 << FrameGenerationFsr;
    static constexpr uint32_t MaxUiFrameGenerationMultiplier = 4;
    static constexpr uint32_t MaxUiGeneratedFrameCount = MaxUiFrameGenerationMultiplier - 1;

    struct Capabilities {
        int modeMask = FrameGenerationModeMaskOff;
        uint32_t dlssMaxMultiplier = 1;
        uint32_t fsrMaxMultiplier = 1;
    };

    static Capabilities query() {
        std::lock_guard<std::mutex> lock(cacheMutex());
        if (isCached()) { return cachedCapabilities(); }

        Capabilities capabilities{};
        auto *renderer = Renderer::try_instance();
        if (renderer == nullptr) { return capabilities; }

        auto framework = renderer->framework();
        if (framework == nullptr || framework->instance() == nullptr || framework->physicalDevice() == nullptr ||
            framework->device() == nullptr) {
            return capabilities;
        }

        isCached() = true;
        if (framework->device()->isDlssFrameGenerationDeviceExtensionsCompatible()) {
            std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
            std::error_code ec;
            std::filesystem::create_directories(dlssPath, ec);

            NgxContext::NgxInitInfo ngxInitInfo{};
            ngxInitInfo.instance = framework->instance();
            ngxInitInfo.physicalDevice = framework->physicalDevice();
            ngxInitInfo.device = framework->device();
            ngxInitInfo.applicationPath = dlssPath.string();

            auto ngxContext = NgxContext::acquireShared(ngxInitInfo);
            if (ngxContext != nullptr && NVSDK_NGX_SUCCEED(ngxContext->queryDlssFrameGenerationAvailable())) {
                const uint32_t maxGeneratedFrameCount = std::min<uint32_t>(
                    MaxUiGeneratedFrameCount, std::max<uint32_t>(1, ngxContext->queryDlssFrameGenerationMaxFrames()));
                capabilities.modeMask |= FrameGenerationModeMaskDlss;
                capabilities.dlssMaxMultiplier =
                    std::min<uint32_t>(MaxUiFrameGenerationMultiplier, maxGeneratedFrameCount + 1);
            }
        }

#ifdef MCVR_ENABLE_FFX_UPSCALER
        if (framework->device()->isFsrFrameGenerationDeviceExtensionsCompatible()) {
            capabilities.modeMask |= FrameGenerationModeMaskFsr;
            capabilities.fsrMaxMultiplier = 2;
        }
#endif

        cachedCapabilities() = capabilities;
        return cachedCapabilities();
    }

  private:
    static std::mutex &cacheMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static bool &isCached() {
        static bool cached = false;
        return cached;
    }

    static Capabilities &cachedCapabilities() {
        static Capabilities capabilities{};
        return capabilities;
    }
};

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxFps(JNIEnv *,
                                                                               jclass,
                                                                               jint maxFps,
                                                                               jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetMaxFps");
    Renderer::options.maxFps = maxFps;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetInactivityFpsLimit(JNIEnv *,
                                                                                           jclass,
                                                                                           jint inactivityFpsLimit,
                                                                                           jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetInactivityFpsLimit");
    Renderer::options.inactivityFpsLimit = inactivityFpsLimit;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVsync(JNIEnv *,
                                                                              jclass,
                                                                              jboolean vsync,
                                                                              jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetVsync");
    Renderer::options.vsync = vsync;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetFrameGenerationMode(JNIEnv *,
                                                                                            jclass,
                                                                                            jint frameGenerationMode,
                                                                                            jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetFrameGenerationMode");
    Renderer::options.frameGenerationMode = static_cast<uint32_t>(frameGenerationMode);
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetFrameGenerationMultiplier(
    JNIEnv *, jclass, jint frameGenerationMultiplier, jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetFrameGenerationMultiplier");
    const auto clampedMultiplier = std::clamp(frameGenerationMultiplier, static_cast<jint>(2), static_cast<jint>(4));
    Renderer::options.frameGenerationMultiplier = static_cast<uint32_t>(clampedMultiplier);
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetAvailableFrameGenerationModeMask(JNIEnv *,
                                                                                                         jclass) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeGetAvailableFrameGenerationModeMask");
    return FrameGenerationCapabilityQuery::query().modeMask;
}

JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetMaxFrameGenerationMultiplier(
    JNIEnv *, jclass, jint frameGenerationMode) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeGetMaxFrameGenerationMultiplier");
    const auto capabilities = FrameGenerationCapabilityQuery::query();
    switch (frameGenerationMode) {
        case FrameGenerationCapabilityQuery::FrameGenerationDlss:
            return (capabilities.modeMask & FrameGenerationCapabilityQuery::FrameGenerationModeMaskDlss) != 0 ?
                       static_cast<jint>(capabilities.dlssMaxMultiplier) :
                       1;
        case FrameGenerationCapabilityQuery::FrameGenerationFsr:
            return (capabilities.modeMask & FrameGenerationCapabilityQuery::FrameGenerationModeMaskFsr) != 0 ?
                       static_cast<jint>(capabilities.fsrMaxMultiplier) :
                       1;
        case FrameGenerationCapabilityQuery::FrameGenerationOff:
        default: return 1;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingBatchSize(
    JNIEnv *, jclass, jint chunkBuildingBatchSize, jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetChunkBuildingBatchSize");
    Renderer::options.chunkBuildingBatchSize = chunkBuildingBatchSize;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingTotalBatches(
    JNIEnv *, jclass, jint chunkBuildingTotalBatches, jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetChunkBuildingTotalBatches");
    Renderer::options.chunkBuildingTotalBatches = chunkBuildingTotalBatches;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingThreads(JNIEnv *,
                                                                                             jclass,
                                                                                             jint chunkBuildingThreads,
                                                                                             jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetChunkBuildingThreads");
    Renderer::options.chunkBuildingThreads = chunkBuildingThreads;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCollectChunkEmission(
    JNIEnv *, jclass, jboolean collectChunkEmission, jboolean write) {
    MCVR_PROFILE_SCOPE("jni.Options.nativeSetCollectChunkEmission");
    (void)write;
    bool collect = static_cast<bool>(collectChunkEmission);
    if (Renderer::options.collectChunkEmission == collect) { return; }

    Renderer::options.collectChunkEmission = collect;
    if (!Renderer::is_initialized()) { return; }

    auto world = Renderer::instance().world();
    if (world != nullptr && world->chunks() != nullptr) { world->chunks()->setCollectChunkEmission(collect); }
    if (!collect) {
        auto textures = Renderer::instance().textures();
        if (textures != nullptr) { textures->releaseEmission(); }
    }
}
