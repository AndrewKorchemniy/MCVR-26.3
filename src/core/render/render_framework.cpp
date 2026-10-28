#include "core/render/render_framework.hpp"

#include "common/shared.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/ui_composite.hpp"
#include "core/render/world.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"
#include "core/vulkan/result.hpp"

#include <cstdio>
#include <exception>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>

#if defined(MCVR_ENABLE_PROFILER)
#    include <fstream>

class RenderFrameworkMetrics {
  public:
    static int64_t readNativeResidentSetBytes() {
#    if defined(__linux__)
        std::ifstream statm("/proc/self/statm");
        uint64_t totalPages = 0;
        uint64_t residentPages = 0;
        if (!(statm >> totalPages >> residentPages)) { return -1; }
        return static_cast<int64_t>(residentPages * 4096ull);
#    else
        return -1;
#    endif
    }

    static void recordFrameMemoryMetrics(const char *phase) {
        vk::recordResourceMemoryProfilerMetrics(phase);

        int64_t residentSetBytes = readNativeResidentSetBytes();
        if (residentSetBytes >= 0) {
            MCVR_PROFILE_METRIC(std::string("memory.process.rss_bytes.native.") + phase, residentSetBytes);
        }
    }
};

#    define MCVR_RECORD_FRAME_MEMORY_METRICS(phase) RenderFrameworkMetrics::recordFrameMemoryMetrics(phase)
#else
#    define MCVR_RECORD_FRAME_MEMORY_METRICS(phase) ((void)0)
#endif

FrameworkContext::FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frameIndex)
    : framework(framework),
      frameIndex(frameIndex),
      instance(framework->instance_),
      window(framework->window_),
      physicalDevice(framework->physicalDevice_),
      device(framework->device_),
      vma(framework->vma_),
      swapchain(framework->swapchain_),
      swapchainImage(frameIndex < framework->swapchain_->swapchainImages().size() ?
                         framework->swapchain_->swapchainImages()[frameIndex] :
                         nullptr),
      commandPool(framework->mainCommandPool_),
      commandProcessedSemaphore(framework->commandProcessedSemaphores_[frameIndex]),
      commandFinishedFence(framework->commandFinishedFences_[frameIndex]),
      uploadCommandBuffer(framework->uploadCommandBuffers_[frameIndex]),
      overlayCommandBuffer(framework->overlayCommandBuffers_[frameIndex]),
      worldCommandBuffer(framework->worldCommandBuffers_[frameIndex]),
      fuseCommandBuffer(framework->fuseCommandBuffers_[frameIndex]) {}

FrameworkContext::~FrameworkContext() {
}

size_t FrameworkContext::scheduleWorldAsyncBranch(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                                  std::shared_ptr<vk::Semaphore> readySemaphore,
                                                  std::shared_ptr<vk::Semaphore> completeSemaphore) {
    if (!device->hasIndependentSecondaryQueue() || commandBuffer == nullptr || readySemaphore == nullptr ||
        completeSemaphore == nullptr || commandBuffer == worldCommandBuffer) {
        throw std::runtime_error("invalid asynchronous world branch");
    }
    asyncWorldBranches.push_back({
        .commandBuffer = std::move(commandBuffer),
        .readySemaphore = std::move(readySemaphore),
        .completeSemaphore = std::move(completeSemaphore),
        .launchSegment = worldCommandBuffer->split(),
    });
    return asyncWorldBranches.size() - 1;
}

void FrameworkContext::joinWorldAsyncBranch(size_t branchIndex) {
    auto &branch = asyncWorldBranches.at(branchIndex);
    if (branch.joinSegment != size_t(-1)) { return; }
    branch.joinSegment = worldCommandBuffer->split() + 1;
}

void FrameworkContext::fuseFinal() {
    MCVR_PROFILE_SCOPE("native.FrameworkContext.fuseFinal");
    auto f = framework.lock();

    if (!f->isRunning()) return;

    vk::CommandBuffer::ScopedDebugLabel debugLabel(*fuseCommandBuffer, "present/final_blit", 0.90f, 0.45f, 0.20f);

    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto finalImage = f->uiComposite()->finalColor(frameIndex);

    fuseCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = finalImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = finalImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = swapchainImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = swapchainImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });

    finalImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(finalImage->width()), static_cast<int>(finalImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(swapchainImage->width()), static_cast<int>(swapchainImage->height()),
                               1};

    vkCmdBlitImage(fuseCommandBuffer->vkCommandBuffer(), finalImage->vkImage(), finalImage->imageLayout(),
                   swapchainImage->vkImage(), swapchainImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

    fuseCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = finalImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = finalImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = swapchainImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = swapchainImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});

    finalImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
}

Framework::Framework() {}

void Framework::init(VulkanHandles handles) {
    MCVR_PROFILE_SCOPE("native.Framework.init");
    vk::ResourceMemoryTagScope memoryTag("framework");
    auto step = [](const char *name) { std::fprintf(stderr, "[MCVR] init: %s\n", name); };
    step("instance");
    instance_ = vk::Instance::create(handles.instance, handles.enabledInstanceExtensions);
    step("window");
    window_ = vk::Window::create(instance_, handles.surface);
    step("physical device");
    physicalDevice_ = vk::PhysicalDevice::create(instance_, handles.physicalDevice, handles.queues);
    step("device");
    device_ = vk::Device::create(instance_, physicalDevice_, handles.device, handles.enabledDeviceExtensions);
    // The device was created with bufferDeviceAddress (part of Radiance's required feature set), which VMA needs.
    step("vma");
    vma_ = vk::VMA::create(instance_, physicalDevice_, device_);
    step("swapchain");
    swapchain_ = vk::Swapchain::create(physicalDevice_, device_, window_);
    step("command pools");
    mainCommandPool_ = vk::CommandPool::create(physicalDevice_, device_);
    asyncCommandPool_ = vk::CommandPool::create(physicalDevice_, device_, physicalDevice_->secondaryQueueIndex());
    frameResourceRetainer_ = FrameResourceRetainer::create(shared_from_this());
    worldAsyncCommandBuffer_ = vk::CommandBuffer::create(device_, asyncCommandPool_);

    step("frame resources");
    createFrameResources();

    step("ui composite");
    uiComposite_ = UiComposite::create(shared_from_this());
    step("pipeline");
    pipeline_ = Pipeline::create(shared_from_this());
    step("frame generation");
    frameGenerationManager_ = mcvr::FrameGenerationManager::create();
    frameGenerationManager_->init(shared_from_this());
    step("done");
}

Framework::~Framework() {
}

void Framework::createFrameResources() {
    const uint32_t imageCount = swapchain_->imageCount();

    for (uint32_t i = 0; i < imageCount; i++) {
        uploadCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        overlayCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        worldCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        fuseCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }
    const uint32_t presentSlotCount = imageCount * mcvr::MaxFrameGenerationMultiplier;
    for (uint32_t i = 0; i < presentSlotCount; i++) {
        presentCopyCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }

    for (uint32_t i = 0; i < imageCount; i++) { commandFinishedFences_.push_back(vk::Fence::create(device_, true)); }
    for (uint32_t i = 0; i < presentSlotCount; i++) {
        presentCopyFinishedFences_.push_back(vk::Fence::create(device_, true));
    }

    for (uint32_t i = 0; i < imageCount; i++) { commandProcessedSemaphores_.push_back(vk::Semaphore::create(device_)); }
    for (uint32_t i = 0; i < imageCount; i++) {
        presentCopyProcessedSemaphores_.push_back(vk::Semaphore::create(device_));
    }
    for (uint32_t i = 0; i < presentSlotCount; i++) {
        additionalPresentImageAcquiredSemaphores_.push_back(vk::Semaphore::create(device_));
        additionalPresentImageAcquireWaitFences_.push_back(nullptr);
    }

    for (uint32_t i = 0; i < imageCount; i++) { contexts_.push_back(FrameworkContext::create(shared_from_this(), i)); }
}

void Framework::clearFrameResources() {
    currentContextIndex_ = 0;
    currentContext_ = nullptr;
    contexts_.clear();

    uploadCommandBuffers_.clear();
    overlayCommandBuffers_.clear();
    worldCommandBuffers_.clear();
    fuseCommandBuffers_.clear();
    presentCopyCommandBuffers_.clear();
    commandFinishedFences_.clear();
    presentCopyFinishedFences_.clear();
    additionalPresentImageAcquireWaitFences_.clear();
}

bool Framework::configureSwapchain(uint32_t width, uint32_t height, uint32_t presentMode) {
    MCVR_PROFILE_SCOPE("native.Framework.configureSwapchain");
    if (!running_ || width == 0 || height == 0) return false;

    VkPresentModeKHR vkPresentMode;
    switch (presentMode) {
        case 0: vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR; break;
        case 1: vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR; break;
        case 3: vkPresentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR; break;
        case 2:
        default: vkPresentMode = VK_PRESENT_MODE_FIFO_KHR; break;
    }

    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    swapchain_->setDesiredConfiguration(width, height, vkPresentMode);
    if (!swapchainSuboptimal_.load() && swapchain_->matchesDesiredConfiguration()) return true;

    try {
        recreate();
    } catch (const std::exception &error) {
        std::cerr << "[Framework] failed to configure the swapchain: " << error.what() << std::endl;
        return false;
    }
    return !swapchainSuboptimal_.load();
}

bool Framework::isSwapchainSuboptimal() {
    if (swapchainSuboptimal_.load()) return true;
    if (!running_ || swapchain_ == nullptr) return false;

    VkSurfaceCapabilitiesKHR capabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &capabilities) != VK_SUCCESS) {
        return false;
    }
    const VkExtent2D extent = swapchain_->vkExtent();
    return capabilities.currentExtent.width != 0xFFFFFFFFu && capabilities.currentExtent.width != 0 &&
           capabilities.currentExtent.height != 0 &&
           (capabilities.currentExtent.width != extent.width || capabilities.currentExtent.height != extent.height);
}

bool Framework::acquireContext() {
    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    return acquireContextLocked(false);
}

bool Framework::acquireContextLocked(bool recreateWhenOutOfDate) {
    if (!running_) return false;
    if (currentContext_ != nullptr) return true;

#if defined(MCVR_ENABLE_PROFILER)
    const auto acquireStart = std::chrono::steady_clock::now();
#endif
    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore;
    uint32_t imageIndex = 0;
    for (;;) {
        imageAcquiredSemaphore = acquireSemaphore();
        VkResult result = vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(), UINT64_MAX,
                                                imageAcquiredSemaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // A semaphore passed to a failed acquire is unsignaled and stays reusable.
            recycleSemaphore(imageAcquiredSemaphore);
            swapchainSuboptimal_.store(true);
            if (!recreateWhenOutOfDate) return false;
            recreate();
            if (!running_) { return false; }
            continue;
        }
        if (result == VK_ERROR_SURFACE_LOST_KHR && !recreateWhenOutOfDate) {
            recycleSemaphore(imageAcquiredSemaphore);
            swapchainSuboptimal_.store(true);
            return false;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            recycleSemaphore(imageAcquiredSemaphore);
            vk::requireSuccess(result, "vkAcquireNextImageKHR(main frame)");
        }
        if (result == VK_SUBOPTIMAL_KHR) { swapchainSuboptimal_.store(true); }
        break;
    }

#if defined(MCVR_ENABLE_PROFILER)
    const auto imageReady = std::chrono::steady_clock::now();
#endif
    std::shared_ptr<vk::Fence> fence = contexts_[imageIndex]->commandFinishedFence;
    vk::waitForFence(device_->vkDevice(), fence->vkFence(), "vkWaitForFences(main frame)");
#if defined(MCVR_ENABLE_PROFILER)
    const auto commandReady = std::chrono::steady_clock::now();
#endif
    waitPresentCopyFences(imageIndex);
#if defined(MCVR_ENABLE_PROFILER)
    const auto copiesReady = std::chrono::steady_clock::now();
#endif
    currentContextIndex_ = imageIndex;
    currentContext_ = contexts_[imageIndex];
    MCVR_PROFILE_BEGIN_FRAME(imageIndex);
    MCVR_RECORD_FRAME_MEMORY_METRICS("begin");
    MCVR_PROFILE_SCOPE("native.Framework.acquireContext");
#if defined(MCVR_ENABLE_PROFILER)
    MCVR_PROFILE_METRIC("frame_acquire.swapchain_ns",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(imageReady - acquireStart).count());
    MCVR_PROFILE_METRIC("frame_acquire.command_fence_ns",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(commandReady - imageReady).count());
    MCVR_PROFILE_METRIC("frame_acquire.present_copy_fences_ns",
                       std::chrono::duration_cast<std::chrono::nanoseconds>(copiesReady - commandReady).count());
#endif
    frameResourceRetainer_->beginFrame(imageIndex);

    if (currentContext_->imageAcquiredSemaphore != nullptr) {
        recycleSemaphore(currentContext_->imageAcquiredSemaphore);
        currentContext_->imageAcquiredSemaphore = nullptr;
    }
    currentContext_->imageAcquiredSemaphore = imageAcquiredSemaphore;

    currentContext_->asyncWorldBranches.clear();
    currentContext_->uploadCommandBuffer->begin();
    currentContext_->worldCommandBuffer->begin();
    currentContext_->overlayCommandBuffer->begin();
    currentContext_->fuseCommandBuffer->begin();
    currentContext_->uploadCommandBuffer->beginDebugLabel("upload", 0.25f, 0.80f, 0.95f);
    currentContext_->overlayCommandBuffer->beginDebugLabel("ui", 0.95f, 0.55f, 0.25f);
    currentContext_->fuseCommandBuffer->beginDebugLabel("present", 0.85f, 0.35f, 0.65f);

    {
        MCVR_PROFILE_SCOPE("native.frame.reset");
        Renderer::instance().buffers()->resetFrame();
        Renderer::instance().textures()->resetFrame();
        Renderer::instance().world()->resetFrame();
        Renderer::instance().world()->chunks()->resetFrame();
        Renderer::instance().world()->entities()->resetFrame();
    }
    return true;
}

void Framework::setPresentSource(VkImage image, VkImageView imageView, uint32_t width, uint32_t height) {
    if (uiComposite_ == nullptr) return;
    uiComposite_->uiSurface()->setTarget(image, imageView, width, height);
}

void Framework::submitFrameCommandBuffers(VkSemaphore signalSemaphore) {
    MCVR_PROFILE_SCOPE("native.Framework.submitFrameCommandBuffers");
    const auto segments = currentContext_->worldCommandBuffer->recordedSegments();
    const auto &branches = currentContext_->asyncWorldBranches;
    for (const auto &branch : branches) {
        if (branch.launchSegment >= branch.joinSegment || branch.joinSegment >= segments.size()) {
            throw std::runtime_error("asynchronous world branch must join before frame completion");
        }
    }
    auto fence = currentContext_->commandFinishedFence;
    vk::requireSuccess(vkResetFences(device_->vkDevice(), 1, &fence->vkFence()), "vkResetFences(main frame)");
    std::unique_lock mainQueueLock(device_->mainQueueMutex(), std::defer_lock);
    std::unique_lock asyncQueueLock(device_->asyncComputeQueueMutex(), std::defer_lock);
    {
        MCVR_PROFILE_SCOPE("native.Framework.waitSubmitQueues");
        std::lock(mainQueueLock, asyncQueueLock);
    }
    for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
        const bool isLast = segmentIndex + 1 == segments.size();
        std::vector<VkCommandBuffer> commands;
        std::vector<VkSemaphore> waits;
        std::vector<VkSemaphore> signals;
        if (segmentIndex == 0) {
            commands.push_back(currentContext_->uploadCommandBuffer->vkCommandBuffer());
            waits.push_back(currentContext_->imageAcquiredSemaphore->vkSemaphore());
        }
        commands.push_back(segments[segmentIndex]);
        for (const auto &branch : branches) {
            if (branch.launchSegment == segmentIndex) { signals.push_back(branch.readySemaphore->vkSemaphore()); }
            if (branch.joinSegment == segmentIndex) { waits.push_back(branch.completeSemaphore->vkSemaphore()); }
        }
        if (isLast) {
            commands.push_back(currentContext_->overlayCommandBuffer->vkCommandBuffer());
            commands.push_back(currentContext_->fuseCommandBuffer->vkCommandBuffer());
            signals.push_back(signalSemaphore);
        }
        const std::vector<VkPipelineStageFlags> waitStages(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waits.size());
        submitInfo.pWaitSemaphores = waits.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = static_cast<uint32_t>(commands.size());
        submitInfo.pCommandBuffers = commands.data();
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
        submitInfo.pSignalSemaphores = signals.data();
        {
            MCVR_PROFILE_SCOPE("native.Framework.submitMainSegment");
            vk::requireSuccess(
                vkQueueSubmit(device_->mainVkQueue(), 1, &submitInfo, isLast ? fence->vkFence() : VK_NULL_HANDLE),
                "vkQueueSubmit(main frame segment)");
        }
        for (const auto &branch : branches) {
            if (branch.launchSegment != segmentIndex) { continue; }
            const VkSemaphore ready = branch.readySemaphore->vkSemaphore();
            const VkSemaphore complete = branch.completeSemaphore->vkSemaphore();
            const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            const auto branchSegments = branch.commandBuffer->recordedSegments();
            VkSubmitInfo asyncSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            asyncSubmit.waitSemaphoreCount = 1;
            asyncSubmit.pWaitSemaphores = &ready;
            asyncSubmit.pWaitDstStageMask = &waitStage;
            asyncSubmit.commandBufferCount = static_cast<uint32_t>(branchSegments.size());
            asyncSubmit.pCommandBuffers = branchSegments.data();
            asyncSubmit.signalSemaphoreCount = 1;
            asyncSubmit.pSignalSemaphores = &complete;
            {
                MCVR_PROFILE_SCOPE("native.Framework.submitAsyncBranch");
                vk::requireSuccess(vkQueueSubmit(device_->asyncComputeQueue(), 1, &asyncSubmit, VK_NULL_HANDLE),
                                   "vkQueueSubmit(asynchronous world branch)");
            }
        }
    }
}

void Framework::submitCommand() {
    MCVR_PROFILE_SCOPE("native.Framework.submitCommand");
    if (!running_) return;

    safeAcquireCurrentContext();

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*currentContext_->uploadCommandBuffer, "upload/textures", 0.25f,
                                                       0.80f, 0.95f);
        MCVR_PROFILE_SCOPE("native.Textures.performQueuedUpload");
        Renderer::instance().textures()->performQueuedUpload();
    }
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*currentContext_->uploadCommandBuffer, "upload/chunks", 0.30f,
                                                       0.75f, 0.95f);
        MCVR_PROFILE_SCOPE("native.Chunks.queueChunkPackedDataUpload");
        auto world = Renderer::instance().world();
        auto chunks = world != nullptr ? world->chunks() : nullptr;
        if (chunks != nullptr) { chunks->queueChunkPackedDataUpload(); }
    }
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*currentContext_->uploadCommandBuffer, "upload/buffers", 0.35f,
                                                       0.70f, 0.95f);
        MCVR_PROFILE_SCOPE("native.Buffers.performQueuedUpload");
        Renderer::instance().buffers()->performQueuedUpload();
    }

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    const bool renderWorld = Renderer::instance().world()->shouldRender() && pipelineContext->worldPipelineContext != nullptr;
    if (renderWorld) { pipelineContext->worldPipelineContext->render(); }
    {
        MCVR_PROFILE_SCOPE("native.ui.composite");
        uiComposite_->record(currentContext_, renderWorld);
    }

    bool frameGenerationEvaluated = false;
    if (frameGenerationManager_ != nullptr) {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*currentContext_->fuseCommandBuffer, "frame_generation", 0.80f,
                                                       0.30f, 0.85f);
        frameGenerationEvaluated = frameGenerationManager_->recordEvaluation(currentContext_);
    }
    if (!frameGenerationEvaluated) { currentContext_->fuseFinal(); }

    {
        MCVR_PROFILE_SCOPE("native.commandBuffers.end");
        currentContext_->uploadCommandBuffer->endDebugLabel()->end();
        currentContext_->worldCommandBuffer->end();
        currentContext_->overlayCommandBuffer->endDebugLabel()->end();
        currentContext_->fuseCommandBuffer->endDebugLabel()->end();
    }

    submitFrameCommandBuffers(currentContext_->commandProcessedSemaphore->vkSemaphore());
    // Minecraft recreates its main target on resize, so the surface is only valid for the frame it was set for.
    uiComposite_->uiSurface()->clearTarget();
}

void Framework::present() {
    MCVR_PROFILE_SCOPE("native.Framework.present");
    if (!running_) return;
    if (currentContext_ == nullptr) return;

    if (frameGenerationManager_ != nullptr && frameGenerationManager_->hasPendingEvaluation()) {
        if (presentFrameGenerationEvaluation()) {
            MCVR_RECORD_FRAME_MEMORY_METRICS("end");
            MCVR_PROFILE_END_FRAME();
            limitFrameRate();
        }
        frameGenerationManager_->clearPendingEvaluation();
        currentContext_ = nullptr;
        return;
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentContext_->commandProcessedSemaphore->vkSemaphore();

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_->vkSwapchain();
    presentInfo.pImageIndices = &currentContext_->frameIndex;

    VkResult result;
    {
        std::unique_lock<std::recursive_mutex> queueLock(device_->presentQueueMutex());
        result = vkQueuePresentKHR(device_->presentQueue(), &presentInfo);
    }
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) { recordPresentedFrame(); }
    currentContext_ = nullptr;

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { swapchainSuboptimal_.store(true); }
    if (Renderer::options.needRecreate || pipeline_->isRecreationNeeded()) {
        MCVR_RECORD_FRAME_MEMORY_METRICS("end");
        MCVR_PROFILE_END_FRAME();
        recreate();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR) {
        vk::requireSuccess(result, "vkQueuePresentKHR(main frame)");
    }

    MCVR_RECORD_FRAME_MEMORY_METRICS("end");
    MCVR_PROFILE_END_FRAME();
    limitFrameRate();
}

bool Framework::presentSwapchainImage(uint32_t imageIndex, VkSemaphore waitSemaphore) {
    MCVR_PROFILE_SCOPE("native.Framework.presentSwapchainImage");
    MCVR_PROFILE_METRIC("present.independent_queue", device_->presentQueue() != device_->mainVkQueue());
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = waitSemaphore == VK_NULL_HANDLE ? 0u : 1u;
    presentInfo.pWaitSemaphores = waitSemaphore == VK_NULL_HANDLE ? nullptr : &waitSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_->vkSwapchain();
    presentInfo.pImageIndices = &imageIndex;

    VkResult result;
    {
        MCVR_PROFILE_SCOPE("native.Framework.queuePresent");
        std::unique_lock<std::recursive_mutex> queueLock(device_->presentQueueMutex());
        result = vkQueuePresentKHR(device_->presentQueue(), &presentInfo);
    }
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) { recordPresentedFrame(); }

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { swapchainSuboptimal_.store(true); }
    if (Renderer::options.needRecreate || pipeline_->isRecreationNeeded()) {
        MCVR_RECORD_FRAME_MEMORY_METRICS("end");
        MCVR_PROFILE_END_FRAME();
        recreate();
        return false;
    } else if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        vk::requireSuccess(result, "vkQueuePresentKHR(frame-generation image)");
    }

    return true;
}

void Framework::recordPresentedFrame() {
    const auto now = std::chrono::steady_clock::now();
    if (presentedFrameSampleStart_ == std::chrono::steady_clock::time_point{}) { presentedFrameSampleStart_ = now; }

    presentedFrameSampleCount_++;
    const std::chrono::duration<double> elapsed = now - presentedFrameSampleStart_;
    if (elapsed.count() < 1.0) return;

    const auto framesPerSecond = static_cast<uint32_t>(presentedFrameSampleCount_ / elapsed.count() + 0.5);
    presentedFrameRate_.store(framesPerSecond, std::memory_order_relaxed);
    presentedFrameSampleCount_ = 0;
    presentedFrameSampleStart_ = now;
}

uint32_t Framework::presentedFrameRate() const {
    return presentedFrameRate_.load(std::memory_order_relaxed);
}

bool Framework::acquireAdditionalPresentImage(uint32_t presentSlot, AdditionalPresentAcquire &acquire) {
    if (presentSlot >= additionalPresentImageAcquiredSemaphores_.size() ||
        presentSlot >= additionalPresentImageAcquireWaitFences_.size()) {
        return false;
    }

    auto waitFence = additionalPresentImageAcquireWaitFences_[presentSlot];
    if (waitFence != nullptr) {
        vk::waitForFence(device_->vkDevice(), waitFence->vkFence(),
                         "vkWaitForFences(frame-generation acquire semaphore)");
        additionalPresentImageAcquireWaitFences_[presentSlot] = nullptr;
    }

    auto semaphore = additionalPresentImageAcquiredSemaphores_[presentSlot];
    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(), UINT64_MAX,
                                            semaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainSuboptimal_.store(true);
        MCVR_RECORD_FRAME_MEMORY_METRICS("end");
        MCVR_PROFILE_END_FRAME();
        return false;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        vk::requireSuccess(result, "vkAcquireNextImageKHR(frame-generation present)");
    }

    if (result == VK_SUBOPTIMAL_KHR) { swapchainSuboptimal_.store(true); }

    acquire.acquireSlot = presentSlot;
    acquire.imageIndex = imageIndex;
    acquire.semaphore = semaphore;
    return true;
}

void Framework::discardAdditionalPresentAcquire(uint32_t acquireSlot) {
    if (acquireSlot >= additionalPresentImageAcquiredSemaphores_.size() ||
        acquireSlot >= additionalPresentImageAcquireWaitFences_.size()) {
        return;
    }

    frameResourceRetainer_->retain(additionalPresentImageAcquiredSemaphores_[acquireSlot]);
    additionalPresentImageAcquiredSemaphores_[acquireSlot] = vk::Semaphore::create(device_);
    additionalPresentImageAcquireWaitFences_[acquireSlot] = nullptr;
}

bool Framework::submitPresentCopy(std::shared_ptr<vk::Image> sourceImage,
                                  uint32_t imageIndex,
                                  const std::vector<VkSemaphore> &waitSemaphores,
                                  const std::vector<uint32_t> &waitedAdditionalAcquireSlots,
                                  uint32_t presentSlot,
                                  bool restoreAsFinalColor) {
    if (sourceImage == nullptr || imageIndex >= swapchain_->swapchainImages().size() ||
        imageIndex >= presentCopyProcessedSemaphores_.size() || presentSlot >= presentCopyCommandBuffers_.size()) {
        for (uint32_t acquireSlot : waitedAdditionalAcquireSlots) { discardAdditionalPresentAcquire(acquireSlot); }
        return false;
    }

    auto commandBuffer = presentCopyCommandBuffers_[presentSlot];
    auto fence = presentCopyFinishedFences_[presentSlot];
    vk::waitForFence(device_->vkDevice(), fence->vkFence(), "vkWaitForFences(frame-generation present copy)");

    vk::requireSuccess(vkResetFences(device_->vkDevice(), 1, &fence->vkFence()),
                       "vkResetFences(frame-generation present copy)");
    commandBuffer->reset()->begin();

    auto swapchainImage = swapchain_->swapchainImages()[imageIndex];
    auto mainQueueIndex = physicalDevice_->mainQueueIndex();
    commandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                    .oldLayout = sourceImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = sourceImage,
                    .subresourceRange = sourceImage->fullSubresourceRange(),
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .oldLayout = swapchainImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = swapchainImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });
    sourceImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(sourceImage->width()), static_cast<int>(sourceImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(swapchainImage->width()), static_cast<int>(swapchainImage->height()),
                               1};

    vkCmdBlitImage(commandBuffer->vkCommandBuffer(), sourceImage->vkImage(), sourceImage->imageLayout(),
                   swapchainImage->vkImage(), swapchainImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

    VkImageLayout restoredSourceLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (restoreAsFinalColor) {
#ifdef USE_AMD
        restoredSourceLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        restoredSourceLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    commandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = sourceImage->imageLayout(),
                    .newLayout = restoredSourceLayout,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = sourceImage,
                    .subresourceRange = sourceImage->fullSubresourceRange(),
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                    .dstAccessMask = 0,
                    .oldLayout = swapchainImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = swapchainImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });
    sourceImage->imageLayout() = restoredSourceLayout;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    commandBuffer->end();

    std::vector<VkSemaphore> activeWaitSemaphores;
    for (auto semaphore : waitSemaphores) {
        if (semaphore != VK_NULL_HANDLE) { activeWaitSemaphores.push_back(semaphore); }
    }
    std::vector<VkPipelineStageFlags> waitStageMasks(activeWaitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSemaphore signalSemaphore = presentCopyProcessedSemaphores_[imageIndex]->vkSemaphore();
    VkCommandBuffer vkCommandBuffer = commandBuffer->vkCommandBuffer();

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(activeWaitSemaphores.size());
    submitInfo.pWaitSemaphores = activeWaitSemaphores.empty() ? nullptr : activeWaitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStageMasks.empty() ? nullptr : waitStageMasks.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vkCommandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSemaphore;

    {
        std::unique_lock<std::recursive_mutex> queueLock(device_->mainQueueMutex());
        vk::requireSuccess(vkQueueSubmit(device_->mainVkQueue(), 1, &submitInfo, fence->vkFence()),
                           "vkQueueSubmit(frame-generation present copy)");
    }

    for (uint32_t acquireSlot : waitedAdditionalAcquireSlots) {
        if (acquireSlot < additionalPresentImageAcquireWaitFences_.size()) {
            additionalPresentImageAcquireWaitFences_[acquireSlot] = fence;
        }
    }

    return true;
}

bool Framework::presentFrameGenerationEvaluation() {
    MCVR_PROFILE_SCOPE("native.Framework.presentFrameGenerationEvaluation");
    const auto evaluation = frameGenerationManager_->lastEvaluation();
    const uint32_t firstImageIndex = currentContext_->frameIndex;
    const uint32_t presentSlotStride = mcvr::MaxFrameGenerationMultiplier;
    const uint32_t firstPresentSlot = firstImageIndex * presentSlotStride;
    const VkSemaphore renderedFrameReady = currentContext_->commandProcessedSemaphore->vkSemaphore();

    auto submitAndPresent = [&](std::shared_ptr<vk::Image> sourceImage, uint32_t imageIndex,
                                const std::vector<VkSemaphore> &waitSemaphores,
                                const std::vector<uint32_t> &waitedAdditionalAcquireSlots, uint32_t presentOrder,
                                bool restoreAsFinalColor) -> bool {
        const uint32_t presentSlot = imageIndex * presentSlotStride + presentOrder;
        if (!submitPresentCopy(sourceImage, imageIndex, waitSemaphores, waitedAdditionalAcquireSlots, presentSlot,
                               restoreAsFinalColor)) {
            return false;
        }
        return presentSwapchainImage(imageIndex, presentCopyProcessedSemaphores_[imageIndex]->vkSemaphore());
    };

    if (evaluation.presentGenerated) {
        if (evaluation.generatedColors.empty() ||
            evaluation.generatedColors.size() > mcvr::MaxFrameGenerationGeneratedFrames) {
            return false;
        }

        for (uint32_t generatedFrameIndex = 0; generatedFrameIndex < evaluation.generatedColors.size();
             generatedFrameIndex++) {
            uint32_t imageIndex = firstImageIndex;
            std::vector<VkSemaphore> waitSemaphores;
            std::vector<uint32_t> waitedAdditionalAcquireSlots;
            if (generatedFrameIndex == 0) {
                waitSemaphores.push_back(renderedFrameReady);
            } else {
                const uint32_t acquireSlot = firstImageIndex * presentSlotStride + generatedFrameIndex;
                AdditionalPresentAcquire acquire{};
                if (!acquireAdditionalPresentImage(acquireSlot, acquire)) { return false; }
                imageIndex = acquire.imageIndex;
                waitSemaphores.push_back(acquire.semaphore->vkSemaphore());
                waitedAdditionalAcquireSlots.push_back(acquire.acquireSlot);
            }

            if (!submitAndPresent(evaluation.generatedColors[generatedFrameIndex], imageIndex, waitSemaphores,
                                  waitedAdditionalAcquireSlots, generatedFrameIndex, false)) {
                return false;
            }
        }

        const uint32_t realFramePresentOrder = static_cast<uint32_t>(evaluation.generatedColors.size());
        const uint32_t realFrameAcquireSlot = firstImageIndex * presentSlotStride + realFramePresentOrder;
        AdditionalPresentAcquire realAcquire{};
        if (!acquireAdditionalPresentImage(realFrameAcquireSlot, realAcquire)) { return false; }
        return submitAndPresent(evaluation.realColor, realAcquire.imageIndex, {realAcquire.semaphore->vkSemaphore()},
                                {realAcquire.acquireSlot}, realFramePresentOrder, true);
    }

    if (!submitPresentCopy(evaluation.realColor, firstImageIndex, {renderedFrameReady}, {}, firstPresentSlot, true)) {
        return false;
    }
    return presentSwapchainImage(firstImageIndex, presentCopyProcessedSemaphores_[firstImageIndex]->vkSemaphore());
}

uint32_t Framework::effectiveFrameRateLimit() const {
    const uint32_t maxFps = Renderer::options.maxFps;

    if (maxFps == 0 || maxFps >= 260) { return 0; }

    return maxFps;
}

void Framework::limitFrameRate() {
    const uint32_t fpsLimit = effectiveFrameRateLimit();
    if (fpsLimit == 0) {
        frameLimitAnchor_ = {};
        frameLimitFps_ = 0;
        return;
    }

    const auto frameDuration =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / fpsLimit));
    if (frameDuration <= std::chrono::steady_clock::duration::zero()) { return; }

    const auto now = std::chrono::steady_clock::now();
    if (frameLimitFps_ != fpsLimit || frameLimitAnchor_ == std::chrono::steady_clock::time_point{}) {
        frameLimitAnchor_ = now;
        frameLimitFps_ = fpsLimit;
    }

    const auto target = frameLimitAnchor_ + frameDuration;
    if (now < target) {
        std::this_thread::sleep_until(target);
        frameLimitAnchor_ = std::chrono::steady_clock::now();
    } else {
        frameLimitAnchor_ = now;
    }
}

void Framework::recreate() {
    MCVR_PROFILE_SCOPE("native.Framework.recreate");
    if (!running_) return;

    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    const bool reportNativeProgress = Pipeline::nativeRebuildActive();

    try {
        // A minimized window has a zero-sized surface, which no swapchain can be created for. Minecraft reconfigures
        // the swapchain once the window has a size again.
        VkSurfaceCapabilitiesKHR capabilities;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                      &capabilities) == VK_SUCCESS &&
            capabilities.currentExtent.width != 0xFFFFFFFFu &&
            (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)) {
            swapchainSuboptimal_.store(true);
            if (reportNativeProgress) { Pipeline::endNativeRebuild(); }
            return;
        }

        Renderer::options.needRecreate = false;
        waitDeviceIdle();

        retireContextImageAcquiredSemaphores();
        retireSemaphores(commandProcessedSemaphores_);
        retireSemaphores(presentCopyProcessedSemaphores_);
        retireSemaphores(additionalPresentImageAcquiredSemaphores_);
        retireQueuedSemaphores(recycledImageAcquiredSemaphores_);
        clearFrameResources();

        swapchain_->reconstruct();
        swapchainSuboptimal_.store(false);

        uint32_t size = swapchain_->imageCount();
        frameResourceRetainer_->reset(size);

        createFrameResources();
        uiComposite_ = UiComposite::create(shared_from_this());

        if (frameGenerationManager_ != nullptr) { frameGenerationManager_->recreate(); }
        pipeline_->recreate(shared_from_this());

        Renderer::instance().textures()->bindAllTextures();

        if (reportNativeProgress) { Pipeline::endNativeRebuild(); }
    } catch (...) {
        if (reportNativeProgress) { Pipeline::endNativeRebuild(); }
        throw;
    }
}

void Framework::waitDeviceIdle() {
    std::scoped_lock queueLock(device_->mainQueueMutex(), device_->secondaryQueueMutex(),
                               device_->asyncComputeQueueMutex(), device_->presentQueueMutex());
    vk::requireSuccess(vkDeviceWaitIdle(device_->vkDevice()), "vkDeviceWaitIdle");
}

void Framework::waitRenderQueueIdle() {
    std::scoped_lock queueLock(device_->mainQueueMutex(), device_->presentQueueMutex());
    vk::requireSuccess(vkQueueWaitIdle(device_->mainVkQueue()), "vkQueueWaitIdle(main queue)");
    if (device_->presentQueue() != device_->mainVkQueue()) {
        vk::requireSuccess(vkQueueWaitIdle(device_->presentQueue()), "vkQueueWaitIdle(present queue)");
    }
}

void Framework::waitBackendQueueIdle() {
    std::unique_lock<std::recursive_mutex> queueLock(device_->secondaryQueueMutex());
    vk::requireSuccess(vkQueueWaitIdle(device_->secondaryQueue()), "vkQueueWaitIdle(backend queue)");
}

void Framework::close() {
    if (running_) {
        pipeline_->close();
        if (frameGenerationManager_ != nullptr) { frameGenerationManager_->close(); }
        NgxContext::shutdownShared();
    }
    running_ = false;
}

bool Framework::isRunning() {
    return running_;
}

std::recursive_mutex &Framework::recreateMtx() {
    return recreateMtx_;
}

std::shared_ptr<vk::Instance> Framework::instance() {
    return instance_;
}

std::shared_ptr<vk::Window> Framework::window() {
    return window_;
}

std::shared_ptr<vk::PhysicalDevice> Framework::physicalDevice() {
    return physicalDevice_;
}

std::shared_ptr<vk::Device> Framework::device() {
    return device_;
}

std::shared_ptr<vk::VMA> Framework::vma() {
    return vma_;
}

std::shared_ptr<vk::Swapchain> Framework::swapchain() {
    return swapchain_;
}

std::shared_ptr<vk::CommandPool> Framework::mainCommandPool() {
    return mainCommandPool_;
}

std::shared_ptr<vk::CommandPool> Framework::asyncCommandPool() {
    return asyncCommandPool_;
}

std::shared_ptr<UiComposite> Framework::uiComposite() {
    return uiComposite_;
}

std::shared_ptr<vk::CommandBuffer> Framework::worldAsyncCommandBuffer() {
    return worldAsyncCommandBuffer_;
}

std::vector<std::shared_ptr<vk::Semaphore>> &Framework::commandProcessedSemaphores() {
    return commandProcessedSemaphores_;
}

std::vector<std::shared_ptr<vk::Fence>> &Framework::commandFinishedFences() {
    return commandFinishedFences_;
}

std::vector<std::shared_ptr<FrameworkContext>> &Framework::contexts() {
    return contexts_;
}

std::shared_ptr<FrameworkContext> Framework::safeAcquireCurrentContext() {
    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    while (currentContext_ == nullptr) { acquireContextLocked(true); }
    return currentContext_;
}

std::shared_ptr<Pipeline> Framework::pipeline() {
    return pipeline_;
}

FrameResourceRetainer &Framework::frameResourceRetainer() {
    return *frameResourceRetainer_;
}

std::shared_ptr<vk::Semaphore> Framework::acquireSemaphore() {
    std::shared_ptr<vk::Semaphore> semaphore;
    if (recycledImageAcquiredSemaphores_.empty()) {
        semaphore = vk::Semaphore::create(device_);
    } else {
        semaphore = recycledImageAcquiredSemaphores_.front();
        recycledImageAcquiredSemaphores_.pop();
    }
    return semaphore;
}

void Framework::recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore) {
    recycledImageAcquiredSemaphores_.push(semaphore);
}

void Framework::waitPresentCopyFences(uint32_t imageIndex) {
    const uint32_t presentSlotStride = mcvr::MaxFrameGenerationMultiplier;
    const uint32_t beginSlot = imageIndex * presentSlotStride;
    const uint32_t endSlot = std::min<uint32_t>(beginSlot + presentSlotStride, presentCopyFinishedFences_.size());

    for (uint32_t presentSlot = beginSlot; presentSlot < endSlot; presentSlot++) {
        auto fence = presentCopyFinishedFences_[presentSlot];
        if (fence == nullptr) { continue; }
        vk::waitForFence(device_->vkDevice(), fence->vkFence(), "vkWaitForFences(frame-generation present copy slot)");
    }
}

void Framework::retireSemaphore(std::shared_ptr<vk::Semaphore> semaphore) {
    if (semaphore != nullptr) { retiredSemaphores_.push_back(semaphore); }
}

void Framework::retireSemaphores(std::vector<std::shared_ptr<vk::Semaphore>> &semaphores) {
    for (auto &semaphore : semaphores) { retireSemaphore(semaphore); }
    semaphores.clear();
}

void Framework::retireQueuedSemaphores(std::queue<std::shared_ptr<vk::Semaphore>> &semaphores) {
    while (!semaphores.empty()) {
        retireSemaphore(semaphores.front());
        semaphores.pop();
    }
}

void Framework::retireContextImageAcquiredSemaphores() {
    for (auto &context : contexts_) {
        if (context == nullptr) { continue; }
        retireSemaphore(context->imageAcquiredSemaphore);
        context->imageAcquiredSemaphore = nullptr;
    }
}

FrameResourceRetainer::FrameResourceRetainer(std::shared_ptr<Framework> framework) {
    reset(framework->swapchain_->imageCount());
}

void FrameResourceRetainer::beginFrame(uint32_t frameIndex) {
    std::unique_lock<std::recursive_mutex> lck(mtx_);

    if (frameIndex >= retainedResourcesByFrame_.size()) {
        throw std::runtime_error("Frame resource retention index exceeds the swapchain image count");
    }

    auto &frameResources = retainedResourcesByFrame_[frameIndex];
    frameResources.submitted.clear();
    frameResources.submitted.swap(frameResources.pending);
}

void FrameResourceRetainer::reset(uint32_t frameCount) {
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    retainedResourcesByFrame_.clear();
    retainedResourcesByFrame_.resize(frameCount);
}
