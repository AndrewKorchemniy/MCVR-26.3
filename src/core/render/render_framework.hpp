#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/frame_generation.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#ifdef MCVR_UI_BACKEND_OPENGL
#    include "core/render/opengl_ui/opengl_ui_surface.hpp"
#endif
#include "core/render/pipeline.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <optional>

class Framework;
#ifdef MCVR_UI_BACKEND_VULKAN
class UIModule;
struct UIModuleContext;
#endif
namespace mcvr {
class FrameGenerationManager;
#ifdef MCVR_UI_BACKEND_OPENGL
class OpenGLUiSurface;
struct OpenGLUiSurfaceExport;
#endif
}

class FrameResourceRetainer : public SharedObject<FrameResourceRetainer> {
  public:
    FrameResourceRetainer(std::shared_ptr<Framework> framework);

    template <typename T>
    void retain(std::shared_ptr<T> resource);
    template <typename... T>
    void retainMany(const std::shared_ptr<T> &...resources);

    void beginFrame(uint32_t frameIndex);
    void reset(uint32_t frameCount);

  private:
    struct FrameResources {
        std::vector<std::shared_ptr<void>> pending;
        std::vector<std::shared_ptr<void>> submitted;
    };

    template <typename T>
    void retainLocked(const std::shared_ptr<T> &resource);

    std::vector<FrameResources> retainedResourcesByFrame_;
    std::recursive_mutex mtx_;
};

struct FrameworkContext : public SharedObject<FrameworkContext> {
    struct AsyncWorldBranch {
        std::shared_ptr<vk::CommandBuffer> commandBuffer;
        std::shared_ptr<vk::Semaphore> readySemaphore;
        std::shared_ptr<vk::Semaphore> completeSemaphore;
        size_t launchSegment = 0;
        size_t joinSegment = size_t(-1);
    };
    std::weak_ptr<Framework> framework;

    uint32_t frameIndex;

    std::shared_ptr<vk::Instance> instance;
    std::shared_ptr<vk::Window> window;
    std::shared_ptr<vk::PhysicalDevice> physicalDevice;
    std::shared_ptr<vk::Device> device;
    std::shared_ptr<vk::VMA> vma;
    std::shared_ptr<vk::Swapchain> swapchain;
    std::shared_ptr<vk::SwapchainImage> swapchainImage;
    std::shared_ptr<vk::CommandPool> commandPool;
    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore = nullptr;
    std::shared_ptr<vk::Semaphore> commandProcessedSemaphore;
    std::shared_ptr<vk::Fence> commandFinishedFence;

    std::shared_ptr<vk::CommandBuffer> uploadCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> overlayCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> fuseCommandBuffer;
    std::vector<AsyncWorldBranch> asyncWorldBranches;

    size_t scheduleWorldAsyncBranch(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                    std::shared_ptr<vk::Semaphore> readySemaphore,
                                    std::shared_ptr<vk::Semaphore> completeSemaphore);
    void joinWorldAsyncBranch(size_t branchIndex);

    FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frame_index);
    ~FrameworkContext();

#ifdef MCVR_UI_BACKEND_VULKAN
    void fuseFinal();
#endif
};

class Framework : public SharedObject<Framework> {
    friend FrameworkContext;
    friend FrameResourceRetainer;

  public:
    Framework();
    ~Framework();

    void init(GLFWwindow *window);
    void init(std::shared_ptr<vk::PresentationTarget> presentationTarget,
              vk::OpenGLDeviceIdentity openGLDeviceIdentity);
    void acquireContext();
    void submitCommand();
    void present();
#ifdef MCVR_UI_BACKEND_OPENGL
    std::shared_ptr<mcvr::OpenGLUiSurface> prepareOpenGLUiFrame();
    mcvr::OpenGLUiSurfaceExport exportOpenGLUiSurface(uint32_t frameIndex);
    std::shared_ptr<mcvr::OpenGLUiSurface> openGLUiSurface(uint32_t frameIndex);
    void submitOpenGLUiFrame();
    bool hasOpenGLUiResizePending() const;
    bool hasOpenGLUiFrame() const;
    bool prepareOpenGLUiResize();
    void completeOpenGLUiResize();
#endif
    void recreate();
    void waitDeviceIdle();
    void waitRenderQueueIdle();
    void waitBackendQueueIdle();
    void close();
    bool isRunning();
    uint32_t presentedFrameRate() const;

    void takeScreenshot(bool withUI, int width, int height, int channel, void *dstPointer);

    std::recursive_mutex &recreateMtx();

    std::shared_ptr<vk::Instance> instance();
    std::shared_ptr<vk::Window> window();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice();
    std::shared_ptr<vk::Device> device();
    std::shared_ptr<vk::VMA> vma();
    std::shared_ptr<vk::Swapchain> swapchain();
    std::shared_ptr<vk::CommandPool> mainCommandPool();
    std::shared_ptr<vk::CommandPool> asyncCommandPool();

    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer();

    std::vector<std::shared_ptr<vk::Semaphore>> &commandProcessedSemaphores();
    std::vector<std::shared_ptr<vk::Fence>> &commandFinishedFences();
    std::vector<std::shared_ptr<FrameworkContext>> &contexts();
    std::shared_ptr<FrameworkContext> safeAcquireCurrentContext();

    std::shared_ptr<Pipeline> pipeline();

    FrameResourceRetainer &frameResourceRetainer();

  private:
    void init(std::shared_ptr<vk::PresentationTarget> presentationTarget,
              std::optional<vk::OpenGLDeviceIdentity> openGLDeviceIdentity);

    struct AdditionalPresentAcquire {
        uint32_t acquireSlot = 0;
        uint32_t imageIndex = 0;
        std::shared_ptr<vk::Semaphore> semaphore;
    };

    std::shared_ptr<vk::Semaphore> acquireSemaphore();
    void recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore);
    void waitPresentCopyFences(uint32_t imageIndex);
    void retireSemaphore(std::shared_ptr<vk::Semaphore> semaphore);
    void retireSemaphores(std::vector<std::shared_ptr<vk::Semaphore>> &semaphores);
    void retireQueuedSemaphores(std::queue<std::shared_ptr<vk::Semaphore>> &semaphores);
    void retireContextImageAcquiredSemaphores();
    bool presentSwapchainImage(uint32_t imageIndex, VkSemaphore waitSemaphore, bool overlapNextFrame = false);
    void finishPendingPresent();
    void recordPresentedFrame();
    bool acquireAdditionalPresentImage(uint32_t presentSlot, AdditionalPresentAcquire &acquire);
    void discardAdditionalPresentAcquire(uint32_t acquireSlot);
    bool submitPresentCopy(std::shared_ptr<vk::Image> sourceImage,
                           uint32_t imageIndex,
                           const std::vector<VkSemaphore> &waitSemaphores,
                           const std::vector<uint32_t> &waitedAdditionalAcquireSlots,
                           uint32_t presentSlot,
                           bool restoreAsFinalColor);
    bool presentFrameGenerationEvaluation();
    uint32_t effectiveFrameRateLimit() const;
    void limitFrameRate();
    void submitFrameCommandBuffers(VkSemaphore signalSemaphore);
#ifdef MCVR_UI_BACKEND_OPENGL
    void initOpenGLUiSurfaces();
    void clearOpenGLUiSurfaces();
    void releaseRetiredOpenGLUiSurfaces();
    void requestOpenGLUiResize();
    bool submitOpenGLUiPostFrameCommands(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                         std::shared_ptr<vk::Fence> fence,
                                         VkSemaphore waitSemaphore,
                                         VkSemaphore signalSemaphore);
#endif

  private:
    std::shared_ptr<vk::Instance> instance_;
    std::shared_ptr<vk::Window> window_;
    std::shared_ptr<vk::PhysicalDevice> physicalDevice_;
    std::shared_ptr<vk::Device> device_;
    std::shared_ptr<vk::VMA> vma_;
    std::shared_ptr<vk::Swapchain> swapchain_;
    std::shared_ptr<vk::CommandPool> mainCommandPool_;
    std::shared_ptr<vk::CommandPool> asyncCommandPool_;

    std::vector<std::shared_ptr<vk::CommandBuffer>> uploadCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> overlayCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> worldCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> fuseCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> presentCopyCommandBuffers_;
#ifdef MCVR_UI_BACKEND_OPENGL
    std::vector<std::shared_ptr<vk::CommandBuffer>> openGLUiPostFrameCommandBuffers_;
#endif
    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer_;

    std::shared_ptr<Pipeline> pipeline_;

    std::vector<std::shared_ptr<vk::Semaphore>> commandProcessedSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> presentCopyProcessedSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> additionalPresentImageAcquiredSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> retiredSemaphores_;
    std::vector<std::shared_ptr<vk::Fence>> commandFinishedFences_;
    std::vector<std::shared_ptr<vk::Fence>> presentCopyFinishedFences_;
#ifdef MCVR_UI_BACKEND_OPENGL
    std::vector<std::shared_ptr<vk::Fence>> openGLUiPostFrameFinishedFences_;
#endif
    std::vector<std::shared_ptr<vk::Fence>> additionalPresentImageAcquireWaitFences_;

    std::vector<std::shared_ptr<FrameworkContext>> contexts_;

    std::shared_ptr<FrameworkContext> currentContext_ = nullptr;
    uint32_t currentContextIndex_ = 0;
    std::queue<uint32_t> indexHistory_;

    std::queue<std::shared_ptr<vk::Semaphore>> recycledImageAcquiredSemaphores_;
    std::future<VkResult> pendingPresent_;
    std::shared_ptr<vk::Semaphore> prefetchedImageSemaphore_;
    uint32_t prefetchedImageIndex_ = 0;
    std::recursive_mutex recreateMtx_;

    bool running_ = true;
    std::chrono::steady_clock::time_point presentedFrameSampleStart_{};
    uint32_t presentedFrameSampleCount_ = 0;
    std::atomic<uint32_t> presentedFrameRate_{0};
    std::chrono::steady_clock::time_point frameLimitAnchor_{};
    uint32_t frameLimitFps_ = 0;

    std::shared_ptr<FrameResourceRetainer> frameResourceRetainer_;
    std::shared_ptr<mcvr::FrameGenerationManager> frameGenerationManager_;
#ifdef MCVR_UI_BACKEND_OPENGL
    std::vector<std::shared_ptr<mcvr::OpenGLUiSurface>> openGLUiSurfaces_;
    std::vector<std::shared_ptr<mcvr::OpenGLUiSurface>> retiredOpenGLUiSurfaces_;
    uint32_t openGLUiSurfaceGeneration_ = 1;
    bool openGLUiResizePending_ = false;
#endif
};

template <typename T>
void FrameResourceRetainer::retainLocked(const std::shared_ptr<T> &resource) {
    if (resource != nullptr) {
        for (auto &frameResources : retainedResourcesByFrame_) { frameResources.pending.push_back(resource); }
    }
}

template <typename T>
void FrameResourceRetainer::retain(std::shared_ptr<T> resource) {
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    retainLocked(resource);
}

template <typename... T>
void FrameResourceRetainer::retainMany(const std::shared_ptr<T> &...resources) {
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    (retainLocked(resources), ...);
}
