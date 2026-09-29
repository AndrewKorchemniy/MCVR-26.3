#include "core/render/dlss_frame_generation.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"

#include <nvsdk_ngx_helpers_dlssg_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

class DlssFrameGenerationHelper {
  public:
    static NVSDK_NGX_Result check(NVSDK_NGX_Result result, const char *functionName, int line) {
        return checkNgxResult(result, functionName, line);
    }

    static NVSDK_NGX_Resource_VK makeImageResource(const std::shared_ptr<vk::Image> &image, bool readWrite) {
        return NVSDK_NGX_Create_ImageView_Resource_VK(image->vkImageView(), image->vkImage(),
                                                      vk::wholeColorSubresourceRange, image->vkFormat(), image->width(),
                                                      image->height(), readWrite);
    }

    static void copyMat(float dst[4][4], const glm::mat4 &src) {
        std::memcpy(dst, glm::value_ptr(src), sizeof(float) * 16);
    }
};

namespace mcvr {

DlssFrameGenerationProvider::~DlssFrameGenerationProvider() {
    close();
}

bool DlssFrameGenerationProvider::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
    if (framework == nullptr || framework->device() == nullptr ||
        !framework->device()->isDlssFrameGenerationDeviceExtensionsCompatible()) {
        return false;
    }

    std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
    std::error_code ec;
    std::filesystem::create_directories(dlssPath, ec);

    NgxContext::NgxInitInfo ngxInitInfo{};
    ngxInitInfo.instance = framework->instance();
    ngxInitInfo.physicalDevice = framework->physicalDevice();
    ngxInitInfo.device = framework->device();
    ngxInitInfo.applicationPath = dlssPath.string();

    ngxContext_ = NgxContext::acquireShared(ngxInitInfo);
    if (ngxContext_ == nullptr) { return false; }

    if (NVSDK_NGX_FAILED(ngxContext_->queryDlssFrameGenerationAvailable())) { return false; }

    available_ = true;
    capabilities_.supported = true;
    capabilities_.maxGeneratedFrameCount = std::max<uint32_t>(1, ngxContext_->queryDlssFrameGenerationMaxFrames());
    return true;
}

void DlssFrameGenerationProvider::releaseFeature() {
    if (featureHandle_ != nullptr) {
        auto framework = framework_.lock();
        if (framework != nullptr) { framework->waitRenderQueueIdle(); }
        NVSDK_NGX_VULKAN_ReleaseFeature(featureHandle_);
        featureHandle_ = nullptr;
    }
    displayExtent_ = {};
    renderExtent_ = {};
    backbufferFormat_ = VK_FORMAT_UNDEFINED;
}

void DlssFrameGenerationProvider::close() {
    releaseFeature();
    ngxContext_ = nullptr;
    available_ = false;
    capabilities_ = {};
}

bool DlssFrameGenerationProvider::isAvailable() const {
    return available_;
}

FrameGenerationCapabilities DlssFrameGenerationProvider::capabilities() const {
    return capabilities_;
}

bool DlssFrameGenerationProvider::ensureFeature(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                                const FrameGenerationInputs &inputs) {
    if (ngxContext_ == nullptr || ngxContext_->parameters() == nullptr) { return false; }

    const VkExtent2D displayExtent{inputs.displayWidth, inputs.displayHeight};
    const VkExtent2D renderExtent{inputs.renderWidth, inputs.renderHeight};
    const VkFormat format = inputs.finalColor->vkFormat();
    if (featureHandle_ != nullptr && displayExtent_.width == displayExtent.width &&
        displayExtent_.height == displayExtent.height && renderExtent_.width == renderExtent.width &&
        renderExtent_.height == renderExtent.height && backbufferFormat_ == format) {
        return true;
    }

    releaseFeature();

    NVSDK_NGX_DLSSG_Create_Params createParams{};
    createParams.Width = displayExtent.width;
    createParams.Height = displayExtent.height;
    createParams.NativeBackbufferFormat = static_cast<unsigned int>(format);
    createParams.RenderWidth = renderExtent.width;
    createParams.RenderHeight = renderExtent.height;
    createParams.DynamicResolutionScaling = false;

    auto *params = ngxContext_->parameters();
    params->Set(NVSDK_NGX_DLSSG_Parameter_Width, createParams.Width);
    params->Set(NVSDK_NGX_DLSSG_Parameter_Height, createParams.Height);
    params->Set(NVSDK_NGX_DLSSG_Parameter_InternalWidth, createParams.RenderWidth);
    params->Set(NVSDK_NGX_DLSSG_Parameter_InternalHeight, createParams.RenderHeight);
    params->Set(NVSDK_NGX_DLSSG_Parameter_DynamicResolution, createParams.DynamicResolutionScaling ? 1u : 0u);

    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;
    NVSDK_NGX_Result result = DlssFrameGenerationHelper::check(
        NGX_VK_CREATE_DLSSG(commandBuffer->vkCommandBuffer(), creationNodeMask, visibilityNodeMask, &featureHandle_,
                            params, &createParams),
        __func__, __LINE__);
    if (NVSDK_NGX_FAILED(result)) {
        featureHandle_ = nullptr;
        return false;
    }

    displayExtent_ = displayExtent;
    renderExtent_ = renderExtent;
    backbufferFormat_ = format;
    return true;
}

bool DlssFrameGenerationProvider::evaluate(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                           const FrameGenerationInputs &inputs,
                                           std::shared_ptr<vk::DeviceLocalImage> outputInterpolated,
                                           uint32_t multiFrameCount,
                                           uint32_t multiFrameIndex) {
    MCVR_PROFILE_SCOPE("native.DlssFrameGenerationProvider.evaluate");
    if (!available_ || commandBuffer == nullptr || inputs.finalColor == nullptr || inputs.depth == nullptr ||
        inputs.motionVector == nullptr || outputInterpolated == nullptr) {
        return false;
    }

    if (!ensureFeature(commandBuffer, inputs)) { return false; }

    auto mainQueueIndex = framework_.lock()->physicalDevice()->mainQueueIndex();
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto addReadBarrier = [&](const std::shared_ptr<vk::Image> &image) {
        if (image == nullptr) { return; }
        barriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask =
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = image->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = image,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
    };

    addReadBarrier(inputs.finalColor);
    addReadBarrier(inputs.depth);
    addReadBarrier(inputs.motionVector);
    if (inputs.hudlessColor != inputs.finalColor) { addReadBarrier(inputs.hudlessColor); }
    addReadBarrier(inputs.uiColorAlpha);

    barriers.push_back({
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout = outputInterpolated->imageLayout(),
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = outputInterpolated,
        .subresourceRange = vk::wholeColorSubresourceRange,
    });

    commandBuffer->barriersBufferImage({}, barriers);

    inputs.finalColor->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    inputs.depth->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    inputs.motionVector->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (inputs.hudlessColor != nullptr) {
        inputs.hudlessColor->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (inputs.uiColorAlpha != nullptr) {
        inputs.uiColorAlpha->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    outputInterpolated->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    NVSDK_NGX_Resource_VK backbuffer = DlssFrameGenerationHelper::makeImageResource(inputs.finalColor, false);
    NVSDK_NGX_Resource_VK depth = DlssFrameGenerationHelper::makeImageResource(inputs.depth, false);
    NVSDK_NGX_Resource_VK motionVector = DlssFrameGenerationHelper::makeImageResource(inputs.motionVector, false);
    NVSDK_NGX_Resource_VK outputInterp = DlssFrameGenerationHelper::makeImageResource(outputInterpolated, true);
    NVSDK_NGX_Resource_VK hudless{};
    NVSDK_NGX_Resource_VK ui{};

    NVSDK_NGX_VK_DLSSG_Eval_Params evalParams{};
    evalParams.pBackbuffer = &backbuffer;
    evalParams.pDepth = &depth;
    evalParams.pMVecs = &motionVector;
    evalParams.pOutputInterpFrame = &outputInterp;

    if (inputs.hudlessColor != nullptr && inputs.hudlessColor->width() == inputs.displayWidth &&
        inputs.hudlessColor->height() == inputs.displayHeight) {
        hudless = DlssFrameGenerationHelper::makeImageResource(inputs.hudlessColor, false);
        evalParams.pHudless = &hudless;
    }
    if (inputs.uiColorAlpha != nullptr && inputs.uiColorAlpha->width() == inputs.displayWidth &&
        inputs.uiColorAlpha->height() == inputs.displayHeight) {
        ui = DlssFrameGenerationHelper::makeImageResource(inputs.uiColorAlpha, false);
        evalParams.pUI = &ui;
    }

    NVSDK_NGX_DLSSG_Opt_Eval_Params optParams{};
    optParams.multiFrameCount = multiFrameCount;
    optParams.multiFrameIndex = multiFrameIndex;
    DlssFrameGenerationHelper::copyMat(optParams.cameraViewToClip, inputs.cameraViewToClip);
    DlssFrameGenerationHelper::copyMat(optParams.clipToCameraView, inputs.clipToCameraView);
    DlssFrameGenerationHelper::copyMat(optParams.clipToLensClip, inputs.clipToLensClip);
    DlssFrameGenerationHelper::copyMat(optParams.clipToPrevClip, inputs.clipToPrevClip);
    DlssFrameGenerationHelper::copyMat(optParams.prevClipToClip, inputs.prevClipToClip);
    optParams.jitterOffset[0] = inputs.jitter.x;
    optParams.jitterOffset[1] = inputs.jitter.y;
    optParams.mvecScale[0] = inputs.motionVectorScale.x;
    optParams.mvecScale[1] = inputs.motionVectorScale.y;
    optParams.cameraPinholeOffset[0] = inputs.cameraPinholeOffset.x;
    optParams.cameraPinholeOffset[1] = inputs.cameraPinholeOffset.y;
    optParams.cameraPos[0] = inputs.cameraPos.x;
    optParams.cameraPos[1] = inputs.cameraPos.y;
    optParams.cameraPos[2] = inputs.cameraPos.z;
    optParams.cameraUp[0] = inputs.cameraUp.x;
    optParams.cameraUp[1] = inputs.cameraUp.y;
    optParams.cameraUp[2] = inputs.cameraUp.z;
    optParams.cameraRight[0] = inputs.cameraRight.x;
    optParams.cameraRight[1] = inputs.cameraRight.y;
    optParams.cameraRight[2] = inputs.cameraRight.z;
    optParams.cameraFwd[0] = inputs.cameraForward.x;
    optParams.cameraFwd[1] = inputs.cameraForward.y;
    optParams.cameraFwd[2] = inputs.cameraForward.z;
    optParams.cameraNear = inputs.cameraNear;
    optParams.cameraFar = inputs.cameraFar;
    optParams.cameraFOV = inputs.cameraFov;
    optParams.cameraAspectRatio = inputs.cameraAspectRatio;
    optParams.colorBuffersHDR = inputs.colorBuffersHDR;
    optParams.depthInverted = inputs.depthInverted;
    optParams.cameraMotionIncluded = inputs.cameraMotionIncluded;
    optParams.reset = inputs.reset;
    optParams.automodeOverrideReset = false;
    optParams.notRenderingGameFrames = inputs.notRenderingGameFrames;
    optParams.orthoProjection = false;
    optParams.motionVectorsInvalidValue = 0.0f;
    optParams.motionVectorsDilated = false;
    optParams.menuDetectionEnabled = true;
    optParams.mvecsSubrectSize = {inputs.renderWidth, inputs.renderHeight};
    optParams.depthSubrectSize = {inputs.renderWidth, inputs.renderHeight};
    optParams.backbufferSubrectSize = {inputs.displayWidth, inputs.displayHeight};
    if (evalParams.pHudless != nullptr) { optParams.hudLessSubrectSize = {inputs.displayWidth, inputs.displayHeight}; }
    if (evalParams.pUI != nullptr) { optParams.uiSubrectSize = {inputs.displayWidth, inputs.displayHeight}; }

    NVSDK_NGX_Result result =
        DlssFrameGenerationHelper::check(NGX_VK_EVALUATE_DLSSG(commandBuffer->vkCommandBuffer(), featureHandle_,
                                                               ngxContext_->parameters(), &evalParams, &optParams),
                                         __func__, __LINE__);
    return NVSDK_NGX_SUCCEED(result);
}

}
