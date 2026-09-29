#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/modules/world/main_render/main_render_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cmath>

bool ToneMappingModule::tryParseFloat(const std::string &text, float &outValue) {
    try {
        outValue = std::stof(text);
        return std::isfinite(outValue);
    } catch (...) { return false; }
}

bool ToneMappingModule::parseBoolValue(const std::string &value, bool fallback) {
    if (value == "render_pipeline.true") return true;
    if (value == "render_pipeline.false") return false;
    return fallback;
}

ToneMappingModule::ToneMappingModule() = default;

void ToneMappingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    const uint32_t size = framework->swapchain()->imageCount();
    hdrImages_.resize(size);
    ldrImages_.resize(size);
    luminanceDistributionImages_.resize(size);
    dummySegmentationMaskImages_.resize(size);
}

bool ToneMappingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.empty() || formats.empty() || frameIndex >= hdrImages_.size()) return false;

    auto framework = framework_.lock();
    if (framework == nullptr) return false;

    if (images[0] == nullptr) {
        hdrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrImages_[frameIndex] = images[0];
    }

    return true;
}

bool ToneMappingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                                std::vector<VkFormat> &formats,
                                                uint32_t frameIndex) {
    (void)formats;
    if (images.empty() || images[0] == nullptr || frameIndex >= ldrImages_.size()) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();
    ldrImages_[frameIndex] = images[0];
    return true;
}

void ToneMappingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    bool changed = false;
    for (int i = 0; i < attributeCount && (2 * i + 1) < static_cast<int>(attributeKVs.size()); ++i) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];
        float floatValue = 0.0f;

        if (key == "render_pipeline.module.tone_mapping.attribute.exposure_ceiling") {
            if (tryParseFloat(value, floatValue)) {
                exposureCeiling_ = std::clamp(floatValue, minExposureCeilingEv, maxExposureCeilingEv);
            }
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.target_luminance") {
            if (tryParseFloat(value, floatValue)) targetLuminance_ = std::clamp(floatValue, 0.0001f, 3.2f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.metering_response") {
            if (tryParseFloat(value, floatValue)) meteringResponse_ = std::clamp(floatValue, 0.1f, 9.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.include_sky_in_meter") {
            includeSkyInMeter_ = parseBoolValue(value, includeSkyInMeter_);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.sky_meter_weight") {
            if (tryParseFloat(value, floatValue)) skyMeterWeight_ = std::clamp(floatValue, 0.0f, 6.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.include_emissive_in_meter") {
            includeEmissiveInMeter_ = parseBoolValue(value, includeEmissiveInMeter_);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.emissive_meter_weight") {
            if (tryParseFloat(value, floatValue)) emissiveMeterWeight_ = std::clamp(floatValue, 0.0f, 4.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.manual_exposure") {
            manualExposure_ = parseBoolValue(value, manualExposure_);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.manual_exposure_ev") {
            if (tryParseFloat(value, floatValue)) manualExposureEv_ = std::clamp(floatValue, 0.0f, 14.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.allow_negative_exposure") {
            allowNegativeExposure_ = parseBoolValue(value, allowNegativeExposure_);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.chroma_preservation") {
            if (tryParseFloat(value, floatValue)) chromaPreservation_ = std::clamp(floatValue, 0.0f, 2.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.output_contrast_range") {
            if (tryParseFloat(value, floatValue)) outputContrastRange_ = std::clamp(floatValue, 0.5f, 24.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.shadow_detail_floor") {
            if (tryParseFloat(value, floatValue)) shadowDetailFloor_ = std::clamp(floatValue, 0.0f, 1.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.shadow_contrast") {
            if (tryParseFloat(value, floatValue)) shadowContrast_ = std::clamp(floatValue, 0.0f, 4.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.shadow_pivot") {
            if (tryParseFloat(value, floatValue)) shadowPivot_ = std::clamp(floatValue, -24.0f, 0.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.output_offset_ev") {
            if (tryParseFloat(value, floatValue)) outputOffsetEv_ = std::clamp(floatValue, -8.0f, 8.0f);
            changed = true;
        } else if (key == "render_pipeline.module.tone_mapping.attribute.curve_lift_limit") {
            if (tryParseFloat(value, floatValue)) curveLiftLimit_ = std::clamp(floatValue, 0.0f, 24.0f);
            changed = true;
        }
    }
    if (changed) ++historyGeneration_;
}

void ToneMappingModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    if (framework == nullptr || worldPipeline == nullptr) return;

    const uint32_t size = framework->swapchain()->imageCount();
    auto shaderPack = worldPipeline->shaderPack();
    hasSegmentationMask_ = shaderPack != nullptr && shaderPack->hasSegmentationMask();

    initDescriptorTables();
    initImages();
    initBuffers();
    initRenderPass();
    initFrameBuffers();
    initPipeline();

    contexts_.resize(size);
    for (uint32_t i = 0; i < size; ++i) {
        contexts_[i] = ToneMappingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                        shared_from_this());
    }

    ++historyGeneration_;
    stateHistoryGeneration_ = 0;
    toneCurveHistoryGeneration_ = 0;
    lastTimePoint_ = std::chrono::high_resolution_clock::now();
}

std::vector<std::shared_ptr<WorldModuleContext>> &ToneMappingModule::contexts() {
    return contexts_;
}

void ToneMappingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                    std::shared_ptr<vk::DeviceLocalImage> image,
                                    int index) {
    (void)sampler;
    (void)image;
    (void)index;
}

void ToneMappingModule::preClose() {}

void ToneMappingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    if (framework == nullptr) return;

    const uint32_t size = framework->swapchain()->imageCount();
    descriptorTables_.resize(size);
    samplers_.resize(size);

    for (uint32_t i = 0; i < size; ++i) {
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet()
                                   .beginStaticDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 3,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 4,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant(VkPushConstantRange{
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       .offset = 0,
                                       .size = sizeof(ToneMappingModulePushConstant),
                                   })
                                   .build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
}

void ToneMappingModule::initImages() {
    auto framework = framework_.lock();
    if (framework == nullptr) return;

    const uint32_t size = framework->swapchain()->imageCount();
    toneCurveImage_ = vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, toneCurveSampleCount,
                                                   1, 1, VK_FORMAT_R32_SFLOAT, ToneMappingModule::toneCurveImageUsage);
    for (uint32_t i = 0; i < size; ++i) {
        luminanceDistributionImages_[i] =
            vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, toneCurveSampleCount, 1, 1,
                                         VK_FORMAT_R32_UINT, ToneMappingModule::distributionImageUsage);
        dummySegmentationMaskImages_[i] =
            vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, 1, 1, 1, VK_FORMAT_R8_UINT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        descriptorTables_[i]->bindSamplerImageForShader(samplers_[i], hdrImages_[i], 0, 0);
        descriptorTables_[i]->bindImage(luminanceDistributionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        descriptorTables_[i]->bindImage(dummySegmentationMaskImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        descriptorTables_[i]->bindImage(toneCurveImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
    }
}

void ToneMappingModule::initBuffers() {
    auto framework = framework_.lock();
    if (framework == nullptr) return;

    temporalStateBuffer_ =
        vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), sizeof(ToneMappingTemporalState),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    for (uint32_t i = 0; i < descriptorTables_.size(); ++i) {
        descriptorTables_[i]->bindBuffer(temporalStateBuffer_, 0, 2);
    }
}

void ToneMappingModule::initRenderPass() {
    renderPass_ = vk::RenderPassBuilder{}
                      .beginAttachmentDescription()
                      .defineAttachmentDescription({
                          .format = ldrImages_[0]->vkFormat(),
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#ifdef USE_AMD
                          .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                          .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                          .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                      })
                      .endAttachmentDescription()
                      .beginAttachmentReference()
                      .defineAttachmentReference({
                          .attachment = 0,
                          .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      })
                      .endAttachmentReference()
                      .beginSubpassDescription()
                      .defineSubpassDescription({
                          .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                          .colorAttachmentIndices = {0},
                      })
                      .endSubpassDescription()
                      .build(framework_.lock()->device());
}

void ToneMappingModule::initFrameBuffers() {
    auto framework = framework_.lock();
    if (framework == nullptr) return;

    const uint32_t size = framework->swapchain()->imageCount();
    framebuffers_.resize(size);
    for (uint32_t i = 0; i < size; ++i) {
        framebuffers_[i] = vk::FramebufferBuilder{}
                               .beginAttachment()
                               .defineAttachment(ldrImages_[i])
                               .endAttachment()
                               .build(framework->device(), renderPass_);
    }
}

void ToneMappingModule::initPipeline() {
    auto framework = framework_.lock();
    if (framework == nullptr) return;

    auto device = framework->device();
    const std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    const VkExtent2D outputExtent = {ldrImages_[0]->width(), ldrImages_[0]->height()};

    distributionShader_ =
        vk::Shader::create(device, (shaderPath / "world/tone_mapping/luminance_distribution_comp.spv").string());
    distributionPipeline_ = vk::ComputePipelineBuilder{}
                                .defineShader(distributionShader_)
                                .definePipelineLayout(descriptorTables_[0])
                                .build(device);

    luminanceMeasurementShader_ =
        vk::Shader::create(device, (shaderPath / "world/tone_mapping/luminance_measurement_comp.spv").string());
    luminanceMeasurementPipeline_ = vk::ComputePipelineBuilder{}
                                        .defineShader(luminanceMeasurementShader_)
                                        .definePipelineLayout(descriptorTables_[0])
                                        .build(device);

    toneCurveShader_ = vk::Shader::create(device, (shaderPath / "world/tone_mapping/tone_curve_comp.spv").string());
    toneCurvePipeline_ = vk::ComputePipelineBuilder{}
                             .defineShader(toneCurveShader_)
                             .definePipelineLayout(descriptorTables_[0])
                             .build(device);

    vertShader_ = vk::Shader::create(device, (shaderPath / "world/tone_mapping/tone_mapping_vert.spv").string());
    fragShader_ = vk::Shader::create(device, (shaderPath / "world/tone_mapping/display_composite_frag.spv").string());

    pipeline_ = vk::GraphicsPipelineBuilder{}
                    .defineRenderPass(renderPass_, 0)
                    .beginShaderStage()
                    .defineShaderStage(vertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                    .defineShaderStage(fragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .endShaderStage()
                    .defineVertexInputState<void>()
                    .defineViewportScissorState({
                        .viewport = {.x = 0,
                                     .y = 0,
                                     .width = static_cast<float>(outputExtent.width),
                                     .height = static_cast<float>(outputExtent.height),
                                     .minDepth = 0.0f,
                                     .maxDepth = 1.0f},
                        .scissor = {.offset = {.x = 0, .y = 0}, .extent = outputExtent},
                    })
                    .defineDepthStencilState({
                        .depthTestEnable = VK_FALSE,
                        .depthWriteEnable = VK_FALSE,
                        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
                        .depthBoundsTestEnable = VK_FALSE,
                        .stencilTestEnable = VK_FALSE,
                    })
                    .beginColorBlendAttachmentState()
                    .defineDefaultColorBlendAttachmentState()
                    .endColorBlendAttachmentState()
                    .definePipelineLayout(descriptorTables_[0])
                    .build(device);
}

ToneMappingModuleContext::ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                   std::shared_ptr<ToneMappingModule> toneMappingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      toneMappingModule(toneMappingModule),
      hdrImage(toneMappingModule->hdrImages_[frameworkContext->frameIndex]),
      ldrImage(toneMappingModule->ldrImages_[frameworkContext->frameIndex]),
      luminanceDistributionImage(toneMappingModule->luminanceDistributionImages_[frameworkContext->frameIndex]),
      toneCurveImage(toneMappingModule->toneCurveImage_),
      segmentationMaskImage(nullptr),
      descriptorTable(toneMappingModule->descriptorTables_[frameworkContext->frameIndex]),
      framebuffer(toneMappingModule->framebuffers_[frameworkContext->frameIndex]) {}

void ToneMappingModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.ToneMappingModuleContext.render");

    auto context = frameworkContext.lock();
    if (context == nullptr) return;
    auto framework = context->framework.lock();
    auto module = toneMappingModule.lock();
    if (framework == nullptr || module == nullptr || hdrImage == nullptr || ldrImage == nullptr) return;

    auto worldCommandBuffer = context->worldCommandBuffer;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

    segmentationMaskImage = nullptr;
    if (module->hasSegmentationMask_) {
        auto pipelineContext = worldPipelineContext.lock();
        if (pipelineContext != nullptr) {
            for (const auto &worldModuleContext : pipelineContext->worldModuleContexts) {
                auto mainRenderContext = std::dynamic_pointer_cast<MainRenderModuleContext>(worldModuleContext);
                if (mainRenderContext != nullptr) {
                    segmentationMaskImage = mainRenderContext->segmentationMaskImage;
                    break;
                }
            }
        }
    }
    if (segmentationMaskImage == nullptr) {
        segmentationMaskImage = module->dummySegmentationMaskImages_[context->frameIndex];
    }
    const bool usingDummySegmentationMask =
        segmentationMaskImage == module->dummySegmentationMaskImages_[context->frameIndex];
    descriptorTable->bindImage(segmentationMaskImage, VK_IMAGE_LAYOUT_GENERAL, 0, 3);

    auto chooseSource = [](VkImageLayout oldLayout, VkPipelineStageFlags2 fallbackStage, VkAccessFlags2 fallbackAccess,
                           VkPipelineStageFlags2 &stage, VkAccessFlags2 &access) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            access = 0;
        } else {
            stage = fallbackStage;
            access = fallbackAccess;
        }
    };

    VkPipelineStageFlags2 hdrSourceStage = 0;
    VkAccessFlags2 hdrSourceAccess = 0;
    chooseSource(hdrImage->imageLayout(),
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, hdrSourceStage, hdrSourceAccess);

    VkPipelineStageFlags2 ldrSourceStage = 0;
    VkAccessFlags2 ldrSourceAccess = 0;
    chooseSource(ldrImage->imageLayout(),
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, ldrSourceStage, ldrSourceAccess);

    VkPipelineStageFlags2 classificationSourceStage = 0;
    VkAccessFlags2 classificationSourceAccess = 0;
    chooseSource(segmentationMaskImage->imageLayout(),
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, classificationSourceStage,
                 classificationSourceAccess);

    VkPipelineStageFlags2 distributionSourceStage = 0;
    VkAccessFlags2 distributionSourceAccess = 0;
    chooseSource(luminanceDistributionImage->imageLayout(),
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, distributionSourceStage,
                 distributionSourceAccess);

    VkPipelineStageFlags2 toneCurveSourceStage = 0;
    VkAccessFlags2 toneCurveSourceAccess = 0;
    chooseSource(toneCurveImage->imageLayout(),
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, toneCurveSourceStage,
                 toneCurveSourceAccess);

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> initialImageBarriers = {
        {
            .srcStageMask = hdrSourceStage,
            .srcAccessMask = hdrSourceAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = hdrImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = hdrImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        },
        {
            .srcStageMask = ldrSourceStage,
            .srcAccessMask = ldrSourceAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = ldrImage->imageLayout(),
#ifdef USE_AMD
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = ldrImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        },
        {
            .srcStageMask = classificationSourceStage,
            .srcAccessMask = classificationSourceAccess,
            .dstStageMask = usingDummySegmentationMask ? VK_PIPELINE_STAGE_2_TRANSFER_BIT :
                                                         ToneMappingModule::toneMappingComputeStages,
            .dstAccessMask = usingDummySegmentationMask ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = segmentationMaskImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = segmentationMaskImage,
            .subresourceRange = segmentationMaskImage->fullSubresourceRange(),
        },
        {
            .srcStageMask = distributionSourceStage,
            .srcAccessMask = distributionSourceAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = luminanceDistributionImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = luminanceDistributionImage,
            .subresourceRange = luminanceDistributionImage->fullSubresourceRange(),
        },
        {
            .srcStageMask = toneCurveSourceStage,
            .srcAccessMask = toneCurveSourceAccess,
            .dstStageMask = ToneMappingModule::toneMappingComputeStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = ToneMappingModule::toneMappingComputeAccess | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = toneCurveImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = toneCurveImage,
            .subresourceRange = toneCurveImage->fullSubresourceRange(),
        },
    };

    worldCommandBuffer->barriersBufferImage({}, initialImageBarriers);
    hdrImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    segmentationMaskImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    luminanceDistributionImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toneCurveImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    if (usingDummySegmentationMask) {
        VkClearColorValue classificationClear{};
        classificationClear.uint32[0] = 1u;
        const VkImageSubresourceRange classificationRange = segmentationMaskImage->fullSubresourceRange();
        vkCmdClearColorImage(worldCommandBuffer->vkCommandBuffer(), segmentationMaskImage->vkImage(),
                             VK_IMAGE_LAYOUT_GENERAL, &classificationClear, 1, &classificationRange);
        worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                      .dstStageMask = ToneMappingModule::toneMappingComputeStages,
                                                      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .srcQueueFamilyIndex = queueIndex,
                                                      .dstQueueFamilyIndex = queueIndex,
                                                      .image = segmentationMaskImage,
                                                      .subresourceRange = classificationRange}});
    }

    const VkImageSubresourceRange distributionRange = luminanceDistributionImage->fullSubresourceRange();
    const VkImageSubresourceRange toneCurveRange = toneCurveImage->fullSubresourceRange();
    VkClearColorValue distributionClear{};
    distributionClear.uint32[0] = 0u;
    vkCmdClearColorImage(worldCommandBuffer->vkCommandBuffer(), luminanceDistributionImage->vkImage(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &distributionClear, 1, &distributionRange);

    worldCommandBuffer->barriersBufferImage({},
                                            {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                              .dstStageMask = ToneMappingModule::toneMappingComputeStages,
                                              .dstAccessMask = ToneMappingModule::toneMappingComputeAccess,
                                              .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                              .srcQueueFamilyIndex = queueIndex,
                                              .dstQueueFamilyIndex = queueIndex,
                                              .image = luminanceDistributionImage,
                                              .subresourceRange = luminanceDistributionImage->fullSubresourceRange()}});
    luminanceDistributionImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    const uint64_t targetHistoryGeneration = module->historyGeneration_;
    const bool resetState = module->stateHistoryGeneration_ != targetHistoryGeneration;
    const bool resetToneCurve = module->toneCurveHistoryGeneration_ != targetHistoryGeneration;
    if (resetToneCurve) {
        VkClearColorValue curveClear{};
        vkCmdClearColorImage(worldCommandBuffer->vkCommandBuffer(), toneCurveImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                             &curveClear, 1, &toneCurveRange);
    }
    if (resetState) {
        worldCommandBuffer->barriersBufferImage(
            {{.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .srcQueueFamilyIndex = queueIndex,
              .dstQueueFamilyIndex = queueIndex,
              .buffer = module->temporalStateBuffer_}},
            {});
        vkCmdFillBuffer(worldCommandBuffer->vkCommandBuffer(), module->temporalStateBuffer_->vkBuffer(), 0,
                        VK_WHOLE_SIZE, 0u);
    }

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> toneCurveResetBarriers;
    if (resetToneCurve) {
        toneCurveResetBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = ToneMappingModule::toneMappingComputeStages,
            .dstAccessMask = ToneMappingModule::toneMappingComputeAccess,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = toneCurveImage,
            .subresourceRange = toneCurveRange,
        });
    }

    worldCommandBuffer->barriersBufferImage(
        {{.srcStageMask = resetState ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
          .srcAccessMask =
              resetState ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
          .dstStageMask = ToneMappingModule::toneMappingComputeStages,
          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .srcQueueFamilyIndex = queueIndex,
          .dstQueueFamilyIndex = queueIndex,
          .buffer = module->temporalStateBuffer_}},
        toneCurveResetBarriers);

    const auto currentTimePoint = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = currentTimePoint - module->lastTimePoint_;
    module->lastTimePoint_ = currentTimePoint;
    float deltaTimeSec = static_cast<float>(elapsedTime.count());
    if (!std::isfinite(deltaTimeSec)) deltaTimeSec = 1.0f / 60.0f;
    deltaTimeSec = std::clamp(deltaTimeSec, 0.0f, 1.0f);

    ToneMappingModulePushConstant pc{};
    pc.logLuminanceMin = ToneMappingModule::logLuminanceMin;
    pc.logLuminanceMax = ToneMappingModule::logLuminanceMax;
    pc.deltaTimeSec = deltaTimeSec;
    pc.exposureCeiling = std::clamp(module->exposureCeiling_, ToneMappingModule::minExposureCeilingEv,
                                    ToneMappingModule::maxExposureCeilingEv);
    pc.targetLuminance = std::clamp(module->targetLuminance_, 0.0001f, 3.2f);
    pc.meteringResponse = std::clamp(module->meteringResponse_, 0.1f, 9.0f);
    pc.skyMeterWeight = module->includeSkyInMeter_ ? std::clamp(module->skyMeterWeight_, 0.0f, 6.0f) : 0.0f;
    pc.emissiveMeterWeight =
        module->includeEmissiveInMeter_ ? std::clamp(module->emissiveMeterWeight_, 0.0f, 4.0f) : 0.0f;
    pc.manualExposureEv = std::clamp(module->manualExposureEv_, 0.0f, 14.0f);
    pc.chromaPreservation = std::clamp(module->chromaPreservation_, 0.0f, 2.0f);
    pc.outputContrastRange = std::clamp(module->outputContrastRange_, 0.5f, 24.0f);
    pc.shadowDetailFloor = std::clamp(module->shadowDetailFloor_, 0.0f, 1.0f);
    pc.shadowContrast = std::clamp(module->shadowContrast_, 0.0f, 4.0f);
    pc.shadowPivot = std::clamp(module->shadowPivot_, -24.0f, 0.0f);
    pc.outputOffsetEv = std::clamp(module->outputOffsetEv_, -8.0f, 8.0f);
    pc.curveLiftLimit = std::clamp(module->curveLiftLimit_, 0.0f, 24.0f);
    pc.flags = (module->manualExposure_ ? ToneMappingModule::manualExposureFlag : 0u) |
               (module->allowNegativeExposure_ ? ToneMappingModule::allowNegativeExposureFlag : 0u) |
               (module->hasSegmentationMask_ ? ToneMappingModule::segmentationMaskAvailableFlag : 0u) |
               (resetState ? ToneMappingModule::resetExposureFlag : 0u) |
               (resetToneCurve ? ToneMappingModule::resetToneCurveFlag : 0u);

    auto buffers = Renderer::instance().buffers();
    auto skyUniformBuffer = buffers != nullptr ? buffers->skyUniformBuffer() : nullptr;
    if (skyUniformBuffer != nullptr && skyUniformBuffer->mappedPtr() != nullptr) {
        const auto *skyUbo = static_cast<const vk::Data::SkyUBO *>(skyUniformBuffer->mappedPtr());
        if (skyUbo->skyType == 1u) {
            glm::vec3 sunDirection(skyUbo->sunDirection);
            const float lengthSquared = glm::dot(sunDirection, sunDirection);
            if (std::isfinite(lengthSquared) && lengthSquared > 1e-8f) {
                sunDirection *= 1.0f / std::sqrt(lengthSquared);
                constexpr float southOffsetSinAngle = 0.17364817766693033f;
                constexpr float southOffsetCosAngle = 0.984807753012208f;
                const float sunElevation = sunDirection.y * southOffsetCosAngle - sunDirection.z * southOffsetSinAngle;
                constexpr float fullNightSunElevation = -0.08f;
                constexpr float fullDaySunElevation = 0.12f;
                float daylightFactor = std::clamp(
                    (sunElevation - fullNightSunElevation) / (fullDaySunElevation - fullNightSunElevation), 0.0f, 1.0f);
                daylightFactor = daylightFactor * daylightFactor * (3.0f - 2.0f * daylightFactor);
                pc.nightFactor = 1.0f - daylightFactor;
            }
        }
    }

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(ToneMappingModulePushConstant), &pc);

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "tone_mapping/luminance_measurement", 0.65f,
                                                       0.45f, 0.95f);
        worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->luminanceMeasurementPipeline_);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), 1, 1, 1);
    }

    worldCommandBuffer->barriersBufferImage(
        {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
          .srcQueueFamilyIndex = queueIndex,
          .dstQueueFamilyIndex = queueIndex,
          .buffer = module->temporalStateBuffer_}},
        {});

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "tone_mapping/luminance_distribution",
                                                       0.60f, 0.50f, 0.95f);
        worldCommandBuffer->bindComputePipeline(module->distributionPipeline_);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->width_ + 15u) / 16u,
                      (module->height_ + 15u) / 16u, 1);
    }

    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = queueIndex,
              .dstQueueFamilyIndex = queueIndex,
              .image = luminanceDistributionImage,
              .subresourceRange = distributionRange}});

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "tone_mapping/tone_curve", 0.55f, 0.55f,
                                                       0.95f);
        worldCommandBuffer->bindComputePipeline(module->toneCurvePipeline_);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), 1, 1, 1);
    }

    worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                  .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                                  .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                  .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                  .srcQueueFamilyIndex = queueIndex,
                                                  .dstQueueFamilyIndex = queueIndex,
                                                  .image = toneCurveImage,
                                                  .subresourceRange = toneCurveImage->fullSubresourceRange()}});

    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "tone_mapping/display_composite", 0.50f,
                                                       0.60f, 0.95f);
        worldCommandBuffer->beginRenderPass({
            .renderPass = module->renderPass_,
            .framebuffer = framebuffer,
            .renderAreaExtent = {ldrImage->width(), ldrImage->height()},
            .clearValues = {{.color = {0.0f, 0.0f, 0.0f, 1.0f}}},
        });
        ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        worldCommandBuffer->bindGraphicsPipeline(module->pipeline_)
            ->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
            ->draw(3, 1)
            ->endRenderPass();
    }
#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif

    if (resetState) module->stateHistoryGeneration_ = targetHistoryGeneration;
    if (resetToneCurve) module->toneCurveHistoryGeneration_ = targetHistoryGeneration;
}
