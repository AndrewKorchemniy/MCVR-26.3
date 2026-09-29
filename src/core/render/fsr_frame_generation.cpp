#include "core/render/fsr_frame_generation.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr_setup.hpp"
#include "core/render/render_framework.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/swapchain.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef MCVR_ENABLE_FFX_UPSCALER
#    include <ffx_api/ffx_framegeneration.hpp>
#    include <ffx_api/vk/ffx_api_vk.hpp>
#endif

class FsrFrameGenerationHelper {
  public:
#ifdef MCVR_ENABLE_FFX_UPSCALER
    static const char *returnCodeString(ffx::ReturnCode code) {
        switch (code) {
            case ffx::ReturnCode::Ok: return "Ok";
            case ffx::ReturnCode::Error: return "Error";
            case ffx::ReturnCode::ErrorUnknownDesctype: return "ErrorUnknownDesctype";
            case ffx::ReturnCode::ErrorRuntimeError: return "ErrorRuntimeError";
            case ffx::ReturnCode::ErrorNoProvider: return "ErrorNoProvider";
            case ffx::ReturnCode::ErrorMemory: return "ErrorMemory";
            case ffx::ReturnCode::ErrorParameter: return "ErrorParameter";
            default: return "Unknown";
        }
    }

    static FfxApiResource
    makeTextureResource(const std::shared_ptr<vk::Image> &image, FfxApiResourceState state, FfxApiResourceUsage usage) {
        FfxApiResource resource{};
        resource.resource = reinterpret_cast<void *>(image->vkImage());
        resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
        resource.description.format = mcvr::fsr::vkToFfxFormat(image->vkFormat());
        resource.description.width = image->width();
        resource.description.height = image->height();
        resource.description.depth = 1;
        resource.description.mipCount = 1;
        resource.description.usage = usage;
        resource.state = state;
        return resource;
    }

    static bool isHdrFormat(VkFormat format) {
        switch (format) {
            case VK_FORMAT_R16G16B16A16_SFLOAT:
            case VK_FORMAT_R32G32B32A32_SFLOAT:
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return true;
            default: return false;
        }
    }
#endif

    static bool validImage(const std::shared_ptr<vk::Image> &image) {
        return image != nullptr && image->vkImage() != VK_NULL_HANDLE && image->vkImageView() != VK_NULL_HANDLE;
    }
};

namespace mcvr {

FsrFrameGenerationProvider::~FsrFrameGenerationProvider() {
    close();
}

bool FsrFrameGenerationProvider::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
#ifndef MCVR_ENABLE_FFX_UPSCALER
    return false;
#else
    if (framework == nullptr || framework->device() == nullptr || framework->physicalDevice() == nullptr ||
        framework->swapchain() == nullptr || !framework->device()->isFsrFrameGenerationDeviceExtensionsCompatible()) {
        return false;
    }

    available_ = true;
    capabilities_.supported = true;
    capabilities_.maxGeneratedFrameCount = 1;
    return true;
#endif
}

void FsrFrameGenerationProvider::releaseContext() {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    auto framework = framework_.lock();
    if (frameGenerationContext_ != nullptr) {
        if (framework != nullptr) { framework->waitRenderQueueIdle(); }
        auto result = ffx::DestroyContext(frameGenerationContext_);
        if (result != ffx::ReturnCode::Ok) {
            std::cerr << "[FSR FG] DestroyContext failed: " << FsrFrameGenerationHelper::returnCodeString(result)
                      << " (" << static_cast<uint32_t>(result) << ")" << std::endl;
        }
        frameGenerationContext_ = nullptr;
    }
#endif
    displayExtent_ = {};
    renderExtent_ = {};
    backbufferFormat_ = VK_FORMAT_UNDEFINED;
    hudlessBackbufferFormat_ = VK_FORMAT_UNDEFINED;
    frameId_ = 0;
}

void FsrFrameGenerationProvider::close() {
    releaseContext();
    available_ = false;
    capabilities_ = {};
}

bool FsrFrameGenerationProvider::isAvailable() const {
    return available_;
}

FrameGenerationCapabilities FsrFrameGenerationProvider::capabilities() const {
    return capabilities_;
}

bool FsrFrameGenerationProvider::ensureContext(const FrameGenerationInputs &inputs) {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    (void)inputs;
    return false;
#else
    auto framework = framework_.lock();
    if (framework == nullptr || framework->device() == nullptr || framework->physicalDevice() == nullptr) {
        return false;
    }

    const VkExtent2D displayExtent{inputs.displayWidth, inputs.displayHeight};
    const VkExtent2D renderExtent{inputs.renderWidth, inputs.renderHeight};
    const VkFormat format = inputs.finalColor->vkFormat();
    const bool hasHudlessColor = FsrFrameGenerationHelper::validImage(inputs.hudlessColor) &&
                                 inputs.hudlessColor->width() == inputs.displayWidth &&
                                 inputs.hudlessColor->height() == inputs.displayHeight;
    const VkFormat hudlessFormat = hasHudlessColor ? inputs.hudlessColor->vkFormat() : VK_FORMAT_UNDEFINED;
    if (frameGenerationContext_ != nullptr && displayExtent_.width == displayExtent.width &&
        displayExtent_.height == displayExtent.height && renderExtent_.width == renderExtent.width &&
        renderExtent_.height == renderExtent.height && backbufferFormat_ == format &&
        hudlessBackbufferFormat_ == hudlessFormat) {
        return true;
    }

    releaseContext();

    ffx::CreateBackendVKDesc backendDesc{};
    backendDesc.vkDevice = framework->device()->vkDevice();
    backendDesc.vkPhysicalDevice = framework->physicalDevice()->vkPhysicalDevice();
    backendDesc.vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(mcvr::fsr::customVkGetDeviceProcAddr);

    ffx::CreateContextDescFrameGeneration createDesc{};
    createDesc.displaySize = {displayExtent.width, displayExtent.height};
    createDesc.maxRenderSize = {std::max(renderExtent.width, displayExtent.width),
                                std::max(renderExtent.height, displayExtent.height)};
    createDesc.backBufferFormat = mcvr::fsr::vkToFfxFormat(format);
    createDesc.flags = 0;
    if (FsrFrameGenerationHelper::isHdrFormat(format) || inputs.colorBuffersHDR) {
        createDesc.flags |= FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if (inputs.depthInverted) { createDesc.flags |= FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED; }

    ffx::CreateContextDescFrameGenerationHudless hudlessDesc{};
    if (hasHudlessColor) { hudlessDesc.hudlessBackBufferFormat = mcvr::fsr::vkToFfxFormat(hudlessFormat); }
    const auto result = hasHudlessColor ?
                            ffx::CreateContext(frameGenerationContext_, nullptr, createDesc, hudlessDesc, backendDesc) :
                            ffx::CreateContext(frameGenerationContext_, nullptr, createDesc, backendDesc);
    if (result != ffx::ReturnCode::Ok) {
        std::cerr << "[FSR FG] CreateContext failed: " << FsrFrameGenerationHelper::returnCodeString(result) << " ("
                  << static_cast<uint32_t>(result) << ")" << std::endl;
        frameGenerationContext_ = nullptr;
        available_ = false;
        return false;
    }

    displayExtent_ = displayExtent;
    renderExtent_ = renderExtent;
    backbufferFormat_ = format;
    hudlessBackbufferFormat_ = hudlessFormat;
    return true;
#endif
}

bool FsrFrameGenerationProvider::evaluate(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                          const FrameGenerationInputs &inputs,
                                          std::shared_ptr<vk::DeviceLocalImage> outputInterpolated,
                                          uint32_t multiFrameCount,
                                          uint32_t multiFrameIndex) {
    MCVR_PROFILE_SCOPE("native.FsrFrameGenerationProvider.evaluate");
    if (!available_ || commandBuffer == nullptr || !FsrFrameGenerationHelper::validImage(inputs.finalColor) ||
        !FsrFrameGenerationHelper::validImage(inputs.depth) ||
        !FsrFrameGenerationHelper::validImage(inputs.motionVector) ||
        !FsrFrameGenerationHelper::validImage(outputInterpolated) || multiFrameCount != 1 || multiFrameIndex != 1) {
        return false;
    }

#ifndef MCVR_ENABLE_FFX_UPSCALER
    return false;
#else
    if (!ensureContext(inputs)) { return false; }

    auto framework = framework_.lock();
    if (framework == nullptr || framework->physicalDevice() == nullptr) { return false; }

    const bool hasHudlessColor = FsrFrameGenerationHelper::validImage(inputs.hudlessColor) &&
                                 inputs.hudlessColor->width() == inputs.displayWidth &&
                                 inputs.hudlessColor->height() == inputs.displayHeight;
    const uint32_t mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto addReadBarrier = [&](const std::shared_ptr<vk::Image> &image) {
        if (!FsrFrameGenerationHelper::validImage(image)) { return; }
        barriers.push_back({.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                            .oldLayout = image->imageLayout(),
                            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            .srcQueueFamilyIndex = mainQueueIndex,
                            .dstQueueFamilyIndex = mainQueueIndex,
                            .image = image,
                            .subresourceRange = vk::wholeColorSubresourceRange});
    };

    addReadBarrier(inputs.finalColor);
    if (hasHudlessColor && inputs.hudlessColor != inputs.finalColor) { addReadBarrier(inputs.hudlessColor); }
    addReadBarrier(inputs.depth);
    addReadBarrier(inputs.motionVector);
    barriers.push_back({.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                        .oldLayout = outputInterpolated->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = outputInterpolated,
                        .subresourceRange = vk::wholeColorSubresourceRange});
    commandBuffer->barriersBufferImage({}, barriers);

    inputs.finalColor->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (hasHudlessColor) { inputs.hudlessColor->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }
    inputs.depth->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    inputs.motionVector->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    outputInterpolated->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    const uint64_t frameId = ++frameId_;

    ffx::ConfigureDescFrameGeneration configureDesc{};
    configureDesc.frameGenerationEnabled = true;
    configureDesc.allowAsyncWorkloads = false;
    configureDesc.flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
    if (hasHudlessColor) {
        configureDesc.HUDLessColor = FsrFrameGenerationHelper::makeTextureResource(
            inputs.hudlessColor, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
    }
    configureDesc.generationRect = {0, 0, static_cast<int32_t>(inputs.displayWidth),
                                    static_cast<int32_t>(inputs.displayHeight)};
    configureDesc.frameID = frameId;
    const auto configureResult = ffx::Configure(frameGenerationContext_, configureDesc);
    if (configureResult != ffx::ReturnCode::Ok) {
        std::cerr << "[FSR FG] Configure failed: " << FsrFrameGenerationHelper::returnCodeString(configureResult)
                  << " (" << static_cast<uint32_t>(configureResult) << ")" << std::endl;
        return false;
    }

    ffx::DispatchDescFrameGenerationPrepare prepareDesc{};
    prepareDesc.frameID = frameId;
    prepareDesc.flags = 0;
    prepareDesc.commandList = commandBuffer->vkCommandBuffer();
    prepareDesc.renderSize = {inputs.renderWidth, inputs.renderHeight};
    prepareDesc.jitterOffset = {-inputs.jitter.x, -inputs.jitter.y};
    prepareDesc.motionVectorScale = {inputs.motionVectorScale.x, inputs.motionVectorScale.y};
    prepareDesc.frameTimeDelta = inputs.frameTimeDelta;
    prepareDesc.unused_reset = inputs.reset;
    prepareDesc.cameraNear = inputs.cameraNear;
    prepareDesc.cameraFar = inputs.cameraFar;
    prepareDesc.cameraFovAngleVertical = inputs.cameraFov;
    prepareDesc.viewSpaceToMetersFactor = 1.0f;
    prepareDesc.depth = FsrFrameGenerationHelper::makeTextureResource(
        inputs.depth, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
    prepareDesc.motionVectors = FsrFrameGenerationHelper::makeTextureResource(
        inputs.motionVector, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);

    ffx::DispatchDescFrameGenerationPrepareCameraInfo cameraInfo{};
    std::memcpy(cameraInfo.cameraPosition, &inputs.cameraPos, sizeof(cameraInfo.cameraPosition));
    std::memcpy(cameraInfo.cameraUp, &inputs.cameraUp, sizeof(cameraInfo.cameraUp));
    std::memcpy(cameraInfo.cameraRight, &inputs.cameraRight, sizeof(cameraInfo.cameraRight));
    std::memcpy(cameraInfo.cameraForward, &inputs.cameraForward, sizeof(cameraInfo.cameraForward));

    const auto prepareResult = ffx::Dispatch(frameGenerationContext_, prepareDesc, cameraInfo);
    if (prepareResult != ffx::ReturnCode::Ok) {
        std::cerr << "[FSR FG] Prepare dispatch failed: " << FsrFrameGenerationHelper::returnCodeString(prepareResult)
                  << " (" << static_cast<uint32_t>(prepareResult) << ")" << std::endl;
        return false;
    }

    ffx::DispatchDescFrameGeneration dispatchDesc{};
    dispatchDesc.commandList = commandBuffer->vkCommandBuffer();
    dispatchDesc.presentColor = FsrFrameGenerationHelper::makeTextureResource(
        inputs.finalColor, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
    dispatchDesc.outputs[0] = FsrFrameGenerationHelper::makeTextureResource(
        outputInterpolated, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
    dispatchDesc.numGeneratedFrames = 1;
    dispatchDesc.reset = inputs.reset;
    dispatchDesc.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    dispatchDesc.minMaxLuminance[0] = 0.0f;
    dispatchDesc.minMaxLuminance[1] = 1000.0f;
    dispatchDesc.generationRect = {0, 0, static_cast<int32_t>(inputs.displayWidth),
                                   static_cast<int32_t>(inputs.displayHeight)};
    dispatchDesc.frameID = frameId;

    const auto dispatchResult = ffx::Dispatch(frameGenerationContext_, dispatchDesc);
    if (dispatchResult != ffx::ReturnCode::Ok) {
        std::cerr << "[FSR FG] Dispatch failed: " << FsrFrameGenerationHelper::returnCodeString(dispatchResult) << " ("
                  << static_cast<uint32_t>(dispatchResult) << ")" << std::endl;
        return false;
    }

    return true;
#endif
}

}
