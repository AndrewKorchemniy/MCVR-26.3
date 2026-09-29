#include "core/render/frame_generation.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/dlss_frame_generation.hpp"
#include "core/render/fsr_frame_generation.hpp"
#ifdef MCVR_UI_BACKEND_VULKAN
#    include "core/render/modules/ui_module.hpp"
#endif
#include "core/render/modules/world/world_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

class FrameGenerationHelper {
  public:
    struct InputPreparationPushConstants {
        float cameraNear;
        float cameraFar;
        uint32_t width;
        uint32_t height;
        float jitterX;
        float jitterY;
    };

    static bool validImage(const std::shared_ptr<vk::Image> &image) {
        return image != nullptr && image->vkImage() != VK_NULL_HANDLE && image->vkImageView() != VK_NULL_HANDLE;
    }

    static glm::vec3 safeNormalize(const glm::vec3 &value, const glm::vec3 &fallback) {
        const float len = glm::length(value);
        if (len <= 1e-6f || std::isnan(len)) { return fallback; }
        return value / len;
    }

    static mcvr::FrameGenerationBackend parseBackend(uint32_t rawBackend) {
        switch (static_cast<mcvr::FrameGenerationBackend>(rawBackend)) {
            case mcvr::FrameGenerationBackend::Dlss: return mcvr::FrameGenerationBackend::Dlss;
            case mcvr::FrameGenerationBackend::Fsr: return mcvr::FrameGenerationBackend::Fsr;
            case mcvr::FrameGenerationBackend::Off:
            default: return mcvr::FrameGenerationBackend::Off;
        }
    }

    static uint32_t backendMask(mcvr::FrameGenerationBackend backend) {
        return backend == mcvr::FrameGenerationBackend::Off ? 0u : (1u << static_cast<uint32_t>(backend));
    }

    static std::vector<mcvr::FrameGenerationBackend> backendCandidates(mcvr::FrameGenerationBackend requestedBackend) {
        std::vector<mcvr::FrameGenerationBackend> candidates;
        auto append = [&](mcvr::FrameGenerationBackend backend) {
            if (backend == mcvr::FrameGenerationBackend::Off) { return; }
            if (std::find(candidates.begin(), candidates.end(), backend) == candidates.end()) {
                candidates.push_back(backend);
            }
        };

        append(requestedBackend);
        append(mcvr::FrameGenerationBackend::Dlss);
        append(mcvr::FrameGenerationBackend::Fsr);
        return candidates;
    }
};

namespace mcvr {

FrameGenerationManager::~FrameGenerationManager() {
    close();
}

void FrameGenerationManager::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
    initialized_ = true;
}

void FrameGenerationManager::close() {
    if (provider_) {
        provider_->close();
        provider_ = nullptr;
    }
    interpolatedImages_.clear();
    preparedDepthImages_.clear();
    preparedMotionVectorImages_.clear();
    inputPreparationDescriptorTables_.clear();
    inputPreparationPipeline_ = nullptr;
    lastEvaluation_ = {};
    activeBackend_ = FrameGenerationBackend::Off;
    unavailableBackendMask_ = 0;
    initialized_ = false;
    historyValid_ = false;
    lastFrameTime_ = {};
}

void FrameGenerationManager::recreate() {
    interpolatedImages_.clear();
    preparedDepthImages_.clear();
    preparedMotionVectorImages_.clear();
    inputPreparationDescriptorTables_.clear();
    inputPreparationPipeline_ = nullptr;
    lastEvaluation_ = {};
    historyValid_ = false;
    lastFrameTime_ = {};
    if (provider_) { provider_->close(); }
    provider_ = nullptr;
    activeBackend_ = FrameGenerationBackend::Off;
    unavailableBackendMask_ = 0;
}

bool FrameGenerationManager::ensureProvider() {
    auto framework = framework_.lock();
    if (!initialized_ || framework == nullptr) { return false; }

    const auto requestedBackend = FrameGenerationHelper::parseBackend(Renderer::options.frameGenerationMode);
    if (requestedBackend == FrameGenerationBackend::Off) {
        if (provider_) { provider_->close(); }
        provider_ = nullptr;
        activeBackend_ = FrameGenerationBackend::Off;
        historyValid_ = false;
        return false;
    }

    if (provider_ != nullptr && !provider_->isAvailable()) {
        provider_->close();
        provider_ = nullptr;
        unavailableBackendMask_ |= FrameGenerationHelper::backendMask(activeBackend_);
        activeBackend_ = FrameGenerationBackend::Off;
        historyValid_ = false;
    }

    const auto candidates = FrameGenerationHelper::backendCandidates(requestedBackend);
    for (const auto candidateBackend : candidates) {
        if ((unavailableBackendMask_ & FrameGenerationHelper::backendMask(candidateBackend)) != 0) { continue; }

        if (provider_ != nullptr && activeBackend_ == candidateBackend) { return provider_->isAvailable(); }

        if (provider_) { provider_->close(); }
        provider_ = nullptr;
        activeBackend_ = FrameGenerationBackend::Off;
        historyValid_ = false;

        switch (candidateBackend) {
            case FrameGenerationBackend::Dlss: {
                provider_ = DlssFrameGenerationProvider::create();
                break;
            }
            case FrameGenerationBackend::Fsr: {
                provider_ = FsrFrameGenerationProvider::create();
                break;
            }
            case FrameGenerationBackend::Off:
            default: return false;
        }

        if (!provider_ || !provider_->init(framework) || !provider_->isAvailable()) {
            if (provider_) { provider_->close(); }
            provider_ = nullptr;
            unavailableBackendMask_ |= FrameGenerationHelper::backendMask(candidateBackend);
            continue;
        }

        activeBackend_ = candidateBackend;
        return true;
    }

    return false;
}

bool FrameGenerationManager::ensureOutputImages(const FrameGenerationInputs &inputs, uint32_t generatedFrameCount) {
    auto framework = framework_.lock();
    if (framework == nullptr || !FrameGenerationHelper::validImage(inputs.finalColor) || generatedFrameCount == 0 ||
        generatedFrameCount > MaxFrameGenerationGeneratedFrames) {
        return false;
    }

    const uint32_t frameCount = framework->swapchain()->imageCount();
    if (interpolatedImages_.size() != frameCount) {
        for (auto &images : interpolatedImages_) {
            for (auto &image : images) { framework->frameResourceRetainer().retain(image); }
        }
        interpolatedImages_.assign(frameCount, {});
    }

    auto context = framework->safeAcquireCurrentContext();
    auto &images = interpolatedImages_[context->frameIndex];
    if (images.size() != generatedFrameCount) {
        for (auto &image : images) { framework->frameResourceRetainer().retain(image); }
        images.assign(generatedFrameCount, nullptr);
    }

    const VkFormat format = inputs.finalColor->vkFormat();
    for (auto &image : images) {
        if (image != nullptr && image->width() == inputs.displayWidth && image->height() == inputs.displayHeight &&
            image->vkFormat() == format) {
            continue;
        }

        framework->frameResourceRetainer().retain(image);
        image = vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, inputs.displayWidth,
                                             inputs.displayHeight, 1, format,
                                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (image == nullptr) { return false; }
    }

    return true;
}

bool FrameGenerationManager::ensureInputPreparationResources(const FrameGenerationInputs &inputs) {
    auto framework = framework_.lock();
    if (framework == nullptr || !FrameGenerationHelper::validImage(inputs.linearDepth) ||
        !FrameGenerationHelper::validImage(inputs.rawMotionVector) || inputs.renderWidth == 0 ||
        inputs.renderHeight == 0) {
        return false;
    }

    const uint32_t frameCount = framework->swapchain()->imageCount();
    if (preparedDepthImages_.size() != frameCount) {
        for (auto &image : preparedDepthImages_) { framework->frameResourceRetainer().retain(image); }
        preparedDepthImages_.assign(frameCount, nullptr);
    }
    if (preparedMotionVectorImages_.size() != frameCount) {
        for (auto &image : preparedMotionVectorImages_) { framework->frameResourceRetainer().retain(image); }
        preparedMotionVectorImages_.assign(frameCount, nullptr);
    }
    if (inputPreparationDescriptorTables_.size() != frameCount) {
        inputPreparationDescriptorTables_.assign(frameCount, nullptr);
    }

    for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        if (inputPreparationDescriptorTables_[frameIndex] == nullptr) {
            inputPreparationDescriptorTables_[frameIndex] =
                vk::DescriptorTableBuilder{}
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
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .defineDescriptorLayoutSetBinding({
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .defineDescriptorLayoutSetBinding({
                        .binding = 3,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .endDescriptorLayoutSetBinding()
                    .endDescriptorLayoutSet()
                    .definePushConstant({
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .offset = 0,
                        .size = sizeof(FrameGenerationHelper::InputPreparationPushConstants),
                    })
                    .build(framework->device());
        }
    }

    auto context = framework->safeAcquireCurrentContext();
    auto &preparedDepth = preparedDepthImages_[context->frameIndex];
    if (preparedDepth == nullptr || preparedDepth->width() != inputs.renderWidth ||
        preparedDepth->height() != inputs.renderHeight || preparedDepth->vkFormat() != VK_FORMAT_R32_SFLOAT) {
        framework->frameResourceRetainer().retain(preparedDepth);
        preparedDepth = vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, inputs.renderWidth,
                                                     inputs.renderHeight, 1, VK_FORMAT_R32_SFLOAT,
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (preparedDepth == nullptr) { return false; }
    }

    auto &preparedMotionVector = preparedMotionVectorImages_[context->frameIndex];
    if (preparedMotionVector == nullptr || preparedMotionVector->width() != inputs.renderWidth ||
        preparedMotionVector->height() != inputs.renderHeight ||
        preparedMotionVector->vkFormat() != VK_FORMAT_R16G16_SFLOAT) {
        framework->frameResourceRetainer().retain(preparedMotionVector);
        preparedMotionVector = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputs.renderWidth, inputs.renderHeight, 1,
            VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (preparedMotionVector == nullptr) { return false; }
    }

    if (inputPreparationPipeline_ == nullptr) {
        auto shader = vk::Shader::create(
            framework->device(),
            (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());
        inputPreparationPipeline_ = vk::ComputePipelineBuilder{}
                                        .defineShader(shader)
                                        .definePipelineLayout(inputPreparationDescriptorTables_[0])
                                        .build(framework->device());
        if (inputPreparationPipeline_ == nullptr) { return false; }
    }

    return true;
}

bool FrameGenerationManager::prepareInputs(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                                           FrameGenerationInputs &inputs) {
    if (FrameGenerationHelper::validImage(inputs.depth) && FrameGenerationHelper::validImage(inputs.motionVector)) {
        return true;
    }
    if (commandBuffer == nullptr || !FrameGenerationHelper::validImage(inputs.linearDepth) ||
        !FrameGenerationHelper::validImage(inputs.rawMotionVector) || !ensureInputPreparationResources(inputs)) {
        return false;
    }

    auto framework = framework_.lock();
    if (framework == nullptr) { return false; }

    auto context = framework->safeAcquireCurrentContext();
    auto preparedDepth = preparedDepthImages_[context->frameIndex];
    auto preparedMotionVector = preparedMotionVectorImages_[context->frameIndex];
    auto descriptorTable = inputPreparationDescriptorTables_[context->frameIndex];
    if (!FrameGenerationHelper::validImage(preparedDepth) || !FrameGenerationHelper::validImage(preparedMotionVector) ||
        descriptorTable == nullptr) {
        return false;
    }

    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    commandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              .srcAccessMask =
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = inputs.linearDepth->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputs.linearDepth,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              .srcAccessMask =
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = inputs.rawMotionVector->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputs.rawMotionVector,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = preparedDepth->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = preparedDepth,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = preparedMotionVector->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = preparedMotionVector,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputs.linearDepth->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    inputs.rawMotionVector->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    preparedDepth->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    preparedMotionVector->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    descriptorTable->bindImage(inputs.linearDepth, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
    descriptorTable->bindImage(preparedDepth, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    descriptorTable->bindImage(inputs.rawMotionVector, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    descriptorTable->bindImage(preparedMotionVector, VK_IMAGE_LAYOUT_GENERAL, 0, 3);

    FrameGenerationHelper::InputPreparationPushConstants pushConstants{
        inputs.cameraNear, inputs.cameraFar, inputs.renderWidth, inputs.renderHeight, inputs.jitter.x, inputs.jitter.y,
    };

    commandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(inputPreparationPipeline_);
    vkCmdPushConstants(commandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer->vkCommandBuffer(), (inputs.renderWidth + 15) / 16, (inputs.renderHeight + 15) / 16, 1);

    inputs.depth = preparedDepth;
    inputs.motionVector = preparedMotionVector;
    inputs.motionVectorScale = glm::vec2(1.0f);
    inputs.depthInverted = false;
    inputs.cameraMotionIncluded = true;
    return true;
}

bool FrameGenerationManager::fillCameraInputs(FrameGenerationInputs &inputs) {
    auto worldUBOBuffer = Renderer::instance().buffers()->worldUniformBuffer();
    auto lastWorldUBOBuffer = Renderer::instance().buffers()->lastWorldUniformBuffer();
    if (worldUBOBuffer == nullptr || lastWorldUBOBuffer == nullptr || worldUBOBuffer->mappedPtr() == nullptr ||
        lastWorldUBOBuffer->mappedPtr() == nullptr) {
        return false;
    }

    const auto *worldUBO = static_cast<const vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
    const auto *lastWorldUBO = static_cast<const vk::Data::WorldUBO *>(lastWorldUBOBuffer->mappedPtr());

    inputs.cameraViewToClip = worldUBO->cameraProjMat;
    inputs.clipToCameraView = worldUBO->cameraProjMatInv;
    inputs.clipToLensClip = glm::mat4(1.0f);

    const glm::mat4 currentWorldToClip = worldUBO->cameraProjMat * worldUBO->cameraEffectedViewMat;
    const glm::mat4 previousWorldToClip = lastWorldUBO->cameraProjMat * lastWorldUBO->cameraEffectedViewMat;
    inputs.clipToPrevClip = previousWorldToClip * glm::inverse(currentWorldToClip);
    inputs.prevClipToClip = currentWorldToClip * glm::inverse(previousWorldToClip);

    inputs.jitter = worldUBO->cameraJitter;
    inputs.cameraPos = glm::vec3(worldUBO->cameraPos);

    const glm::mat4 viewInv = worldUBO->cameraEffectedViewMatInv;
    inputs.cameraRight = FrameGenerationHelper::safeNormalize(glm::vec3(viewInv[0]), glm::vec3(1.0f, 0.0f, 0.0f));
    inputs.cameraUp = FrameGenerationHelper::safeNormalize(glm::vec3(viewInv[1]), glm::vec3(0.0f, 1.0f, 0.0f));
    inputs.cameraForward = FrameGenerationHelper::safeNormalize(-glm::vec3(viewInv[2]), glm::vec3(0.0f, 0.0f, -1.0f));

    inputs.cameraNear = 0.1f;
    inputs.cameraFar = 10000.0f;
    inputs.cameraFov = 2.0f * std::atan(1.0f / std::abs(worldUBO->cameraProjMat[1][1]));
    inputs.cameraAspectRatio = inputs.displayHeight == 0 ?
                                   1.0f :
                                   static_cast<float>(inputs.displayWidth) / static_cast<float>(inputs.displayHeight);
    return true;
}

bool FrameGenerationManager::collectInputs(std::shared_ptr<FrameworkContext> context, FrameGenerationInputs &inputs) {
    auto framework = framework_.lock();
    if (framework == nullptr || context == nullptr || framework->pipeline() == nullptr) { return false; }

    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
#ifdef MCVR_UI_BACKEND_VULKAN
    if (pipelineContext == nullptr || pipelineContext->uiModuleContext == nullptr) { return false; }

    inputs.finalColor = pipelineContext->uiModuleContext->overlayDrawColorImage;
#else
    if (pipelineContext == nullptr) { return false; }

    inputs.finalColor = framework->openGLUiSurface(context->frameIndex);
#endif
    if (!FrameGenerationHelper::validImage(inputs.finalColor)) { return false; }

    inputs.displayWidth = inputs.finalColor->width();
    inputs.displayHeight = inputs.finalColor->height();

    if (pipelineContext->worldPipelineContext != nullptr &&
        FrameGenerationHelper::validImage(pipelineContext->worldPipelineContext->outputImage) &&
        pipelineContext->worldPipelineContext->outputImage->width() == inputs.displayWidth &&
        pipelineContext->worldPipelineContext->outputImage->height() == inputs.displayHeight) {
        inputs.hudlessColor = pipelineContext->worldPipelineContext->outputImage;
    }

    if (pipelineContext->worldPipelineContext != nullptr) {
        for (const auto &moduleContext : pipelineContext->worldPipelineContext->worldModuleContexts) {
            if (moduleContext != nullptr) { moduleContext->populateFrameGenerationInputs(inputs); }
        }
    }

    if ((!FrameGenerationHelper::validImage(inputs.depth) || !FrameGenerationHelper::validImage(inputs.motionVector)) &&
        (!FrameGenerationHelper::validImage(inputs.linearDepth) ||
         !FrameGenerationHelper::validImage(inputs.rawMotionVector))) {
        return false;
    }

    auto dimensionsImage = FrameGenerationHelper::validImage(inputs.depth) ? inputs.depth : inputs.linearDepth;
    if (inputs.renderWidth == 0) { inputs.renderWidth = dimensionsImage->width(); }
    if (inputs.renderHeight == 0) { inputs.renderHeight = dimensionsImage->height(); }

    if (!fillCameraInputs(inputs)) { return false; }
    inputs.reset = inputs.reset || !historyValid_;
    return true;
}

bool FrameGenerationManager::recordEvaluation(std::shared_ptr<FrameworkContext> context) {
    return recordEvaluation(context, context == nullptr ? nullptr : context->fuseCommandBuffer);
}

bool FrameGenerationManager::recordEvaluation(std::shared_ptr<FrameworkContext> context,
                                              std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    MCVR_PROFILE_SCOPE("native.FrameGenerationManager.recordEvaluation");
    lastEvaluation_ = {};
    auto framework = framework_.lock();
    if (framework == nullptr || Renderer::options.needRecreate ||
        (framework->pipeline() != nullptr && framework->pipeline()->isRecreationNeeded())) {
        historyValid_ = false;
        return false;
    }
    FrameGenerationInputs inputs{};
    if (!collectInputs(context, inputs)) { return false; }
    if (!ensureProvider() || commandBuffer == nullptr) { return false; }
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*commandBuffer, "frame_generation/prepare_inputs", 0.80f, 0.30f,
                                                       0.85f);
        if (!prepareInputs(commandBuffer, inputs)) { return false; }
    }

    const auto now = std::chrono::steady_clock::now();
    if (lastFrameTime_.time_since_epoch().count() != 0 && historyValid_) {
        inputs.frameTimeDelta = std::chrono::duration<float, std::milli>(now - lastFrameTime_).count();
    }
    lastFrameTime_ = now;

    const uint32_t requestedMultiplier = std::min<uint32_t>(
        MaxFrameGenerationMultiplier, std::max<uint32_t>(2, Renderer::options.frameGenerationMultiplier));
    const uint32_t requestedGeneratedFrames = requestedMultiplier - 1;
    const uint32_t maxGeneratedFrames = std::min<uint32_t>(
        MaxFrameGenerationGeneratedFrames, std::max<uint32_t>(1, provider_->capabilities().maxGeneratedFrameCount));
    const uint32_t generatedFrames = std::min<uint32_t>(requestedGeneratedFrames, maxGeneratedFrames);

    if (!ensureOutputImages(inputs, generatedFrames)) { return false; }

    auto &outputs = interpolatedImages_[context->frameIndex];
    for (uint32_t generatedFrameIndex = 1; generatedFrameIndex <= generatedFrames; generatedFrameIndex++) {
        auto output = outputs[generatedFrameIndex - 1];
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*commandBuffer, "frame_generation/evaluate", 0.80f, 0.30f,
                                                       0.85f);
        if (!provider_->evaluate(commandBuffer, inputs, output, generatedFrames, generatedFrameIndex)) {
            historyValid_ = false;
            return false;
        }
    }

    lastEvaluation_.evaluated = true;
    lastEvaluation_.presentGenerated = historyValid_ && generatedFrames >= 1;
    if (lastEvaluation_.presentGenerated) { lastEvaluation_.generatedColors.assign(outputs.begin(), outputs.end()); }
    lastEvaluation_.realColor = inputs.finalColor;
    historyValid_ = true;
    return true;
}

const FrameGenerationEvaluation &FrameGenerationManager::lastEvaluation() const {
    return lastEvaluation_;
}

bool FrameGenerationManager::hasPendingEvaluation() const {
    if (!lastEvaluation_.evaluated || !FrameGenerationHelper::validImage(lastEvaluation_.realColor)) { return false; }
    if (!lastEvaluation_.presentGenerated) { return true; }
    return !lastEvaluation_.generatedColors.empty() &&
           std::all_of(lastEvaluation_.generatedColors.begin(), lastEvaluation_.generatedColors.end(),
                       [](const std::shared_ptr<vk::DeviceLocalImage> &image) {
                           return FrameGenerationHelper::validImage(image);
                       });
}

void FrameGenerationManager::clearPendingEvaluation() {
    lastEvaluation_ = {};
}

}
