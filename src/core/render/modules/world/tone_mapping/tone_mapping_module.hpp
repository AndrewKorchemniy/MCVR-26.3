#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <chrono>
#include <cstdint>

#include "core/render/modules/world/world_module.hpp"

class Framework;
struct FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct ToneMappingModuleContext;

struct ToneMappingTemporalState {
    float exposureEv;
    float measuredLogLuminance;
    uint32_t frameSerial;
    uint32_t valid;
};

struct ToneMappingModulePushConstant {
    float logLuminanceMin;
    float logLuminanceMax;
    float deltaTimeSec;
    float exposureCeiling;
    float targetLuminance;
    float meteringResponse;
    float skyMeterWeight;
    float emissiveMeterWeight;
    float manualExposureEv;
    float chromaPreservation;
    float outputContrastRange;
    float shadowDetailFloor;
    float shadowContrast;
    float shadowPivot;
    float outputOffsetEv;
    float curveLiftLimit;
    uint32_t flags;
    float nightFactor;
};

static_assert(sizeof(ToneMappingTemporalState) == 16);
static_assert(sizeof(ToneMappingModulePushConstant) == 72);

class ToneMappingModule : public WorldModule, public SharedObject<ToneMappingModule> {
    friend ToneMappingModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.tone_mapping.name";
    constexpr static uint32_t inputImageNum = 1;
    constexpr static uint32_t outputImageNum = 1;

    ToneMappingModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

  private:
    static constexpr uint32_t toneCurveSampleCount = 256;
    static constexpr float logLuminanceMin = -24.0f;
    static constexpr float logLuminanceMax = 4.0f;
    static constexpr float minExposureCeilingEv = 0.0f;
    static constexpr float maxExposureCeilingEv = 4.0f;
    static constexpr float defaultExposureCeilingEv = 2.25f;
    static constexpr VkImageUsageFlags distributionImageUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    static constexpr VkImageUsageFlags toneCurveImageUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    static constexpr VkPipelineStageFlags2 toneMappingComputeStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    static constexpr VkAccessFlags2 toneMappingComputeAccess =
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    static constexpr uint32_t manualExposureFlag = 1u << 0u;
    static constexpr uint32_t allowNegativeExposureFlag = 1u << 1u;
    static constexpr uint32_t segmentationMaskAvailableFlag = 1u << 2u;
    static constexpr uint32_t resetExposureFlag = 1u << 3u;
    static constexpr uint32_t resetToneCurveFlag = 1u << 4u;

    static bool tryParseFloat(const std::string &text, float &outValue);
    static bool parseBoolValue(const std::string &value, bool fallback);

    void initDescriptorTables();
    void initImages();
    void initBuffers();
    void initRenderPass();
    void initFrameBuffers();
    void initPipeline();

  private:
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> ldrImages_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::vector<std::shared_ptr<vk::Sampler>> samplers_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> luminanceDistributionImages_;
    std::shared_ptr<vk::DeviceLocalImage> toneCurveImage_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> dummySegmentationMaskImages_;
    std::shared_ptr<vk::DeviceLocalBuffer> temporalStateBuffer_;

    std::shared_ptr<vk::Shader> distributionShader_;
    std::shared_ptr<vk::ComputePipeline> distributionPipeline_;
    std::shared_ptr<vk::Shader> luminanceMeasurementShader_;
    std::shared_ptr<vk::ComputePipeline> luminanceMeasurementPipeline_;
    std::shared_ptr<vk::Shader> toneCurveShader_;
    std::shared_ptr<vk::ComputePipeline> toneCurvePipeline_;

    std::shared_ptr<vk::Shader> vertShader_;
    std::shared_ptr<vk::Shader> fragShader_;
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> pipeline_;

    float exposureCeiling_ = defaultExposureCeilingEv;
    float targetLuminance_ = 0.18f;
    float meteringResponse_ = 3.0f;
    bool includeSkyInMeter_ = true;
    float skyMeterWeight_ = 2.0f;
    bool includeEmissiveInMeter_ = true;
    float emissiveMeterWeight_ = 1.0f;
    bool manualExposure_ = false;
    float manualExposureEv_ = 4.5f;
    bool allowNegativeExposure_ = true;

    float chromaPreservation_ = 0.515f;
    float outputContrastRange_ = 6.5f;
    float shadowDetailFloor_ = 0.02f;
    float shadowContrast_ = 0.0f;
    float shadowPivot_ = -8.0f;
    float outputOffsetEv_ = 0.0f;
    float curveLiftLimit_ = 2.75f;

    bool hasSegmentationMask_ = false;
    uint64_t historyGeneration_ = 1;
    uint64_t stateHistoryGeneration_ = 0;
    uint64_t toneCurveHistoryGeneration_ = 0;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTimePoint_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

struct ToneMappingModuleContext : public WorldModuleContext, SharedObject<ToneMappingModuleContext> {
    std::weak_ptr<ToneMappingModule> toneMappingModule;

    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> ldrImage;
    std::shared_ptr<vk::DeviceLocalImage> luminanceDistributionImage;
    std::shared_ptr<vk::DeviceLocalImage> toneCurveImage;
    std::shared_ptr<vk::DeviceLocalImage> segmentationMaskImage;
    std::shared_ptr<vk::DescriptorTable> descriptorTable;
    std::shared_ptr<vk::Framebuffer> framebuffer;
    ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                             std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                             std::shared_ptr<ToneMappingModule> toneMappingModule);

    void render() override;
};
