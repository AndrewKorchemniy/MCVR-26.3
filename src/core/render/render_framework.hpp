#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/frame_generation.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/pipeline.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

class Framework;
class UiComposite;
namespace mcvr {
class FrameGenerationManager;
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

// The Vulkan handles Minecraft created and Radiance adopts instead of creating its own.
struct VulkanHandles {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    vk::AdoptedQueues queues;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::vector<std::string> enabledInstanceExtensions;
    std::vector<std::string> enabledDeviceExtensions;
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
    // Composites the UI surface over the world.
    std::shared_ptr<vk::CommandBuffer> overlayCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer;
    // Frame generation and the final copy to the swapchain image.
    std::shared_ptr<vk::CommandBuffer> fuseCommandBuffer;
    std::vector<AsyncWorldBranch> asyncWorldBranches;

    size_t scheduleWorldAsyncBranch(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                    std::shared_ptr<vk::Semaphore> readySemaphore,
                                    std::shared_ptr<vk::Semaphore> completeSemaphore);
    void joinWorldAsyncBranch(size_t branchIndex);

    FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frame_index);
    ~FrameworkContext();

    void fuseFinal();
};

class Framework : public SharedObject<Framework> {
    friend FrameworkContext;
    friend FrameResourceRetainer;

  public:
    Framework();
    ~Framework();

    // Adopts Minecraft's instance, device, queues and surface and creates the swapchain on that surface.
    void init(VulkanHandles handles);
    // presentMode is the ordinal of IMMEDIATE, MAILBOX, FIFO, FIFO_RELAXED. Returns false if the swapchain could not
    // be created.
    bool configureSwapchain(uint32_t width, uint32_t height, uint32_t presentMode);
    bool isSwapchainSuboptimal();
    // Returns false if no swapchain image could be acquired.
    bool acquireContext();
    // The image Minecraft drew its UI into this frame. Not owned and consumed by the next submitCommand().
    void setPresentSource(VkImage image, VkImageView imageView, uint32_t width, uint32_t height);
    void submitCommand();
    void present();
    void recreate();
    void waitDeviceIdle();
    void waitRenderQueueIdle();
    void waitBackendQueueIdle();
    void close();
    bool isRunning();
    uint32_t presentedFrameRate() const;

    std::recursive_mutex &recreateMtx();

    std::shared_ptr<vk::Instance> instance();
    std::shared_ptr<vk::Window> window();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice();
    std::shared_ptr<vk::Device> device();
    std::shared_ptr<vk::VMA> vma();
    std::shared_ptr<vk::Swapchain> swapchain();
    std::shared_ptr<vk::CommandPool> mainCommandPool();
    std::shared_ptr<vk::CommandPool> asyncCommandPool();
    std::shared_ptr<UiComposite> uiComposite();

    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer();

    std::vector<std::shared_ptr<vk::Semaphore>> &commandProcessedSemaphores();
    std::vector<std::shared_ptr<vk::Fence>> &commandFinishedFences();
    std::vector<std::shared_ptr<FrameworkContext>> &contexts();
    std::shared_ptr<FrameworkContext> safeAcquireCurrentContext();

    std::shared_ptr<Pipeline> pipeline();

    FrameResourceRetainer &frameResourceRetainer();

  private:
    struct AdditionalPresentAcquire {
        uint32_t acquireSlot = 0;
        uint32_t imageIndex = 0;
        std::shared_ptr<vk::Semaphore> semaphore;
    };

    bool acquireContextLocked(bool recreateWhenOutOfDate);
    void createFrameResources();
    void clearFrameResources();
    std::shared_ptr<vk::Semaphore> acquireSemaphore();
    void recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore);
    void waitPresentCopyFences(uint32_t imageIndex);
    void retireSemaphore(std::shared_ptr<vk::Semaphore> semaphore);
    void retireSemaphores(std::vector<std::shared_ptr<vk::Semaphore>> &semaphores);
    void retireQueuedSemaphores(std::queue<std::shared_ptr<vk::Semaphore>> &semaphores);
    void retireContextImageAcquiredSemaphores();
    bool presentSwapchainImage(uint32_t imageIndex, VkSemaphore waitSemaphore);
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
    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer_;

    std::shared_ptr<Pipeline> pipeline_;
    std::shared_ptr<UiComposite> uiComposite_;

    std::vector<std::shared_ptr<vk::Semaphore>> commandProcessedSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> presentCopyProcessedSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> additionalPresentImageAcquiredSemaphores_;
    std::vector<std::shared_ptr<vk::Semaphore>> retiredSemaphores_;
    std::vector<std::shared_ptr<vk::Fence>> commandFinishedFences_;
    std::vector<std::shared_ptr<vk::Fence>> presentCopyFinishedFences_;
    std::vector<std::shared_ptr<vk::Fence>> additionalPresentImageAcquireWaitFences_;

    std::vector<std::shared_ptr<FrameworkContext>> contexts_;

    std::shared_ptr<FrameworkContext> currentContext_ = nullptr;
    uint32_t currentContextIndex_ = 0;

    std::queue<std::shared_ptr<vk::Semaphore>> recycledImageAcquiredSemaphores_;
    std::recursive_mutex recreateMtx_;

    bool running_ = true;
    // Set when vkAcquireNextImageKHR or vkQueuePresentKHR reports that the swapchain no longer matches the surface.
    std::atomic<bool> swapchainSuboptimal_{false};
    std::chrono::steady_clock::time_point presentedFrameSampleStart_{};
    uint32_t presentedFrameSampleCount_ = 0;
    std::atomic<uint32_t> presentedFrameRate_{0};
    std::chrono::steady_clock::time_point frameLimitAnchor_{};
    uint32_t frameLimitFps_ = 0;

    std::shared_ptr<FrameResourceRetainer> frameResourceRetainer_;
    std::shared_ptr<mcvr::FrameGenerationManager> frameGenerationManager_;
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
