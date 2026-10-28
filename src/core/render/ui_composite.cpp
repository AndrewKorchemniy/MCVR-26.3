#include "core/render/ui_composite.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

void UiSurfaceImage::setTarget(VkImage image, VkImageView imageView, uint32_t width, uint32_t height) {
    image_ = image;
    imageView_ = imageView;
    width_ = width;
    height_ = height;
    layout_ = VK_IMAGE_LAYOUT_GENERAL;
}

void UiSurfaceImage::clearTarget() {
    image_ = VK_NULL_HANDLE;
    imageView_ = VK_NULL_HANDLE;
    width_ = 0;
    height_ = 0;
}

bool UiSurfaceImage::valid() const {
    return image_ != VK_NULL_HANDLE && width_ > 0 && height_ > 0;
}

uint32_t UiSurfaceImage::width() {
    return width_;
}

uint32_t UiSurfaceImage::height() {
    return height_;
}

uint32_t UiSurfaceImage::depth() {
    return 1;
}

uint32_t UiSurfaceImage::layer() {
    return 1;
}

VkFormat &UiSurfaceImage::vkFormat() {
    return format_;
}

VkImage &UiSurfaceImage::vkImage() {
    return image_;
}

VkImageView &UiSurfaceImage::vkImageView(int) {
    return imageView_;
}

VkImageLayout &UiSurfaceImage::imageLayout() {
    layout_ = VK_IMAGE_LAYOUT_GENERAL;
    return layout_;
}

VkImageSubresourceRange UiSurfaceImage::fullSubresourceRange() const {
    return {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

UiComposite::UiComposite() {}

UiComposite::~UiComposite() {}

void UiComposite::init(std::shared_ptr<Framework> framework) {
    vk::ResourceMemoryTagScope memoryTag("ui");
    framework_ = framework;
    uiSurface_ = std::make_shared<UiSurfaceImage>();

    auto device = framework->device();
    const uint32_t frameCount = framework->swapchain()->imageCount();
    const VkExtent2D extent = framework->swapchain()->vkExtent();

    sampler_ = vk::Sampler::create(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                   VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    linearSampler_ = vk::Sampler::create(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    finalColorImages_.resize(frameCount);
    uiCopyImages_.resize(frameCount);
    worldCopyImages_.resize(frameCount);
    descriptorTables_.resize(frameCount);
    for (uint32_t i = 0; i < frameCount; i++) {
        finalColorImages_[i] = vk::DeviceLocalImage::create(
            device, framework->vma(), false, extent.width, extent.height, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        uiCopyImages_[i] = vk::DeviceLocalImage::create(device, framework->vma(), false, extent.width, extent.height, 1,
                                                        VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
        worldCopyImages_[i] = vk::DeviceLocalImage::create(device, framework->vma(), false, extent.width,
                                                           extent.height, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_IMAGE_USAGE_SAMPLED_BIT);

        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet()
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant({
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                       .offset = 0,
                                       .size = sizeof(PushConstants),
                                   })
                                   .build(device);
        descriptorTables_[i]->bindImage(finalColorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        descriptorTables_[i]->bindSamplerImage(sampler_, uiCopyImages_[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
                                               1, 0);
        descriptorTables_[i]->bindSamplerImage(linearSampler_, worldCopyImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 2, 0);
    }

    auto shader =
        vk::Shader::create(device, (Renderer::folderPath / "shaders/ui/composite_comp.spv").string());
    pipeline_ = vk::ComputePipelineBuilder{}.defineShader(shader).definePipelineLayout(descriptorTables_[0]).build(device);
    if (pipeline_ == nullptr) { throw std::runtime_error("failed to create the UI composite pipeline"); }
}

std::shared_ptr<UiSurfaceImage> UiComposite::uiSurface() {
    return uiSurface_;
}

std::shared_ptr<vk::DeviceLocalImage> UiComposite::finalColor(uint32_t frameIndex) {
    return frameIndex < finalColorImages_.size() ? finalColorImages_[frameIndex] : nullptr;
}

void UiComposite::record(std::shared_ptr<FrameworkContext> context, bool worldRendered) {
    MCVR_PROFILE_SCOPE("native.UiComposite.record");
    auto framework = framework_.lock();
    auto commandBuffer = context->overlayCommandBuffer;
    auto finalImage = finalColorImages_.at(context->frameIndex);
    auto uiCopy = uiCopyImages_.at(context->frameIndex);
    auto mainQueueIndex = context->physicalDevice->mainQueueIndex();
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*commandBuffer, "ui/composite", 0.95f, 0.55f, 0.25f);

    constexpr VkAccessFlags2 allAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    auto fullBarrier = [&] {
        commandBuffer->barriersMemory({{
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = allAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = allAccess,
        }});
    };
    auto imageBarrier = [&](std::shared_ptr<vk::Image> image, VkImageLayout oldLayout, VkImageLayout newLayout) {
        return vk::CommandBuffer::ImageMemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = allAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = allAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = image,
            .subresourceRange = vk::wholeColorSubresourceRange,
        };
    };

    // Minecraft submitted its frame to this queue before MCVR's, so everything it wrote is ordered before this point;
    // the barrier makes its writes visible. Its images stay in GENERAL and are never transitioned here.
    fullBarrier();

    auto worldImage = context->framework.lock()->pipeline()->acquirePipelineContext(context)->worldPipelineContext;
    const bool copyWorld = worldRendered && worldImage != nullptr && worldImage->outputImage != nullptr;
    // A blur request lasts one frame: Minecraft asks again every frame its blurred menu is drawn.
    const float blurRadius = worldBlurRadius_.exchange(0.0f);
    const bool blurWorld = copyWorld && blurRadius > 0.0f;
    auto worldCopy = worldCopyImages_.at(context->frameIndex);

    if (copyWorld) {
        auto world = worldImage->outputImage;
        const VkImageLayout worldRestoreLayout =
#ifdef USE_AMD
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(world, world->imageLayout(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                 imageBarrier(finalImage, finalImage->imageLayout(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)});
        world->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        finalImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageBlit imageBlit{};
        imageBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.srcOffsets[1] = {static_cast<int>(world->width()), static_cast<int>(world->height()), 1};
        imageBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.dstOffsets[1] = {static_cast<int>(finalImage->width()), static_cast<int>(finalImage->height()), 1};
        vkCmdBlitImage(commandBuffer->vkCommandBuffer(), world->vkImage(), world->imageLayout(),
                       finalImage->vkImage(), finalImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

        if (blurWorld) {
            commandBuffer->barriersBufferImage(
                {}, {imageBarrier(worldCopy, worldCopy->imageLayout(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)});
            worldCopy->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            vkCmdBlitImage(commandBuffer->vkCommandBuffer(), world->vkImage(), world->imageLayout(),
                           worldCopy->vkImage(), worldCopy->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
        }

        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(world, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, worldRestoreLayout),
                 imageBarrier(finalImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL)});
        world->imageLayout() = worldRestoreLayout;
        finalImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    } else {
        // Nothing rendered by MCVR (menus, loading screens): Minecraft's own frame is the whole picture.
        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(finalImage, finalImage->imageLayout(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)});
        VkClearColorValue clearColor{};
        clearColor.float32[3] = 1.0f;
        auto range = vk::wholeColorSubresourceRange;
        vkCmdClearColorImage(commandBuffer->vkCommandBuffer(), finalImage->vkImage(),
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);
        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(finalImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL)});
        finalImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    }

    // The composite always binds the world copy, so keep it in the layout its descriptor names even when unused.
    if (worldCopy->imageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(worldCopy, worldCopy->imageLayout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)});
        worldCopy->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (uiSurface_->valid()) {
        // Minecraft stores its targets bottom-up like a GL framebuffer, so the copy flips Y while scaling to the
        // swapchain size.
        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(uiCopy, uiCopy->imageLayout(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)});
        uiCopy->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageBlit imageBlit{};
        imageBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.srcOffsets[0] = {0, static_cast<int>(uiSurface_->height()), 0};
        imageBlit.srcOffsets[1] = {static_cast<int>(uiSurface_->width()), 0, 1};
        imageBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.dstOffsets[1] = {static_cast<int>(uiCopy->width()), static_cast<int>(uiCopy->height()), 1};
        vkCmdBlitImage(commandBuffer->vkCommandBuffer(), uiSurface_->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                       uiCopy->vkImage(), uiCopy->imageLayout(), 1, &imageBlit,
                       (uiSurface_->width() == uiCopy->width() && uiSurface_->height() == uiCopy->height()) ?
                           VK_FILTER_NEAREST :
                           VK_FILTER_LINEAR);

        commandBuffer->barriersBufferImage(
            {}, {imageBarrier(uiCopy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                 imageBarrier(finalImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL)});
        uiCopy->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        auto descriptorTable = descriptorTables_[context->frameIndex];
        commandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(pipeline_);
        PushConstants pushConstants{.blurRadius = blurWorld ? blurRadius : 0.0f};
        vkCmdPushConstants(commandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer->vkCommandBuffer(), (finalImage->width() + 15) / 16, (finalImage->height() + 15) / 16,
                      1);

        commandBuffer->barriersBufferImage({}, {imageBarrier(finalImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL)});
        // Minecraft's next frame writes the surface again; order those writes after the copy above.
        fullBarrier();
    }

    std::scoped_lock screenshotLock(screenshotMutex_);
    if (screenshotRequested_) {
        screenshotRequested_ = false;
        const uint32_t width = finalImage->width();
        const uint32_t height = finalImage->height();
        const size_t size = static_cast<size_t>(width) * height * 4;
        if (screenshotBuffer_ == nullptr || screenshotBuffer_->size() != size) {
            framework->frameResourceRetainer().retain(screenshotBuffer_);
            screenshotBuffer_ =std::make_shared<vk::HostVisibleBuffer>(framework->vma(), framework->device(), size,
                                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        fullBarrier();
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(commandBuffer->vkCommandBuffer(), finalImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                               screenshotBuffer_->vkBuffer(), 1, &region);
        commandBuffer->barriersMemory({{
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
            .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        }});
        screenshotCaptured_ = true;
        screenshotExtent_ = {width, height};
    }
}

void UiComposite::setWorldBlurRadius(float radius) {
    worldBlurRadius_.store(std::max(radius, 0.0f));
}

void UiComposite::requestScreenshot() {
    std::scoped_lock lock(screenshotMutex_);
    screenshotRequested_ = true;
}

VkExtent2D UiComposite::screenshotSize() {
    std::scoped_lock lock(screenshotMutex_);
    return screenshotCaptured_ ? screenshotExtent_ : VkExtent2D{0, 0};
}

bool UiComposite::readScreenshot(void *dst, uint32_t width, uint32_t height) {
    auto framework = framework_.lock();
    if (framework == nullptr || dst == nullptr) return false;
    std::scoped_lock lock(screenshotMutex_);
    if (!screenshotCaptured_ || screenshotExtent_.width != width || screenshotExtent_.height != height) {
        return false;
    }
    screenshotCaptured_ = false;
    {
        // The copy was recorded into a frame that has been submitted by now; screenshots are rare enough to wait for
        // the whole queue rather than that frame's fence.
        auto device = framework->device();
        std::scoped_lock queueLock(device->mainQueueMutex());
        vkQueueWaitIdle(device->mainVkQueue());
    }
    const size_t size = static_cast<size_t>(width) * height * 4;
    screenshotBuffer_->downloadFromBuffer(size, 0);
    std::memcpy(dst, screenshotBuffer_->mappedPtr(), size);
    return true;
}
