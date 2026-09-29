#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <glm/glm.hpp>

#include <chrono>
#include <vector>

class Framework;
struct FrameworkContext;
struct PipelineContext;

namespace mcvr {

enum class FrameGenerationBackend : uint32_t {
    Off = 0,
    Dlss = 1,
    Fsr = 2,
};

constexpr uint32_t MaxFrameGenerationMultiplier = 4;
constexpr uint32_t MaxFrameGenerationGeneratedFrames = MaxFrameGenerationMultiplier - 1;

struct FrameGenerationInputs {
    std::shared_ptr<vk::Image> finalColor;
    std::shared_ptr<vk::Image> hudlessColor;
    std::shared_ptr<vk::Image> uiColorAlpha;
    std::shared_ptr<vk::Image> depth;
    std::shared_ptr<vk::Image> motionVector;
    std::shared_ptr<vk::Image> linearDepth;
    std::shared_ptr<vk::Image> rawMotionVector;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth = 0;
    uint32_t displayHeight = 0;

    glm::mat4 cameraViewToClip = glm::mat4(1.0f);
    glm::mat4 clipToCameraView = glm::mat4(1.0f);
    glm::mat4 clipToLensClip = glm::mat4(1.0f);
    glm::mat4 clipToPrevClip = glm::mat4(1.0f);
    glm::mat4 prevClipToClip = glm::mat4(1.0f);

    glm::vec2 jitter = glm::vec2(0.0f);
    glm::vec2 motionVectorScale = glm::vec2(1.0f);
    glm::vec2 cameraPinholeOffset = glm::vec2(0.0f);
    glm::vec3 cameraPos = glm::vec3(0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);

    float cameraNear = 0.1f;
    float cameraFar = 10000.0f;
    float cameraFov = 0.0f;
    float cameraAspectRatio = 1.0f;
    float frameTimeDelta = 16.67f;

    bool colorBuffersHDR = false;
    bool depthInverted = false;
    bool cameraMotionIncluded = true;
    bool reset = false;
    bool notRenderingGameFrames = false;
};

struct FrameGenerationCapabilities {
    bool supported = false;
    uint32_t maxGeneratedFrameCount = 1;
};

class IFrameGenerationProvider {
  public:
    virtual ~IFrameGenerationProvider() = default;

    virtual bool init(std::shared_ptr<Framework> framework) = 0;
    virtual void close() = 0;
    virtual bool isAvailable() const = 0;
    virtual FrameGenerationCapabilities capabilities() const = 0;
    virtual bool evaluate(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                          const FrameGenerationInputs &inputs,
                          std::shared_ptr<vk::DeviceLocalImage> outputInterpolated,
                          uint32_t multiFrameCount,
                          uint32_t multiFrameIndex) = 0;
};

struct FrameGenerationEvaluation {
    bool evaluated = false;
    bool presentGenerated = false;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> generatedColors;
    std::shared_ptr<vk::Image> realColor;
};

class FrameGenerationManager : public SharedObject<FrameGenerationManager> {
  public:
    FrameGenerationManager() = default;
    ~FrameGenerationManager();

    void init(std::shared_ptr<Framework> framework);
    void close();
    void recreate();

    bool recordEvaluation(std::shared_ptr<FrameworkContext> context);
    bool recordEvaluation(std::shared_ptr<FrameworkContext> context, std::shared_ptr<vk::CommandBuffer> commandBuffer);
    const FrameGenerationEvaluation &lastEvaluation() const;
    bool hasPendingEvaluation() const;
    void clearPendingEvaluation();

  private:
    bool ensureProvider();
    bool ensureOutputImages(const FrameGenerationInputs &inputs, uint32_t generatedFrameCount);
    bool ensureInputPreparationResources(const FrameGenerationInputs &inputs);
    bool prepareInputs(std::shared_ptr<vk::CommandBuffer> commandBuffer, FrameGenerationInputs &inputs);
    bool collectInputs(std::shared_ptr<FrameworkContext> context, FrameGenerationInputs &inputs);
    bool fillCameraInputs(FrameGenerationInputs &inputs);

    std::weak_ptr<Framework> framework_;
    std::shared_ptr<IFrameGenerationProvider> provider_;
    std::vector<std::vector<std::shared_ptr<vk::DeviceLocalImage>>> interpolatedImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> preparedDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> preparedMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> inputPreparationDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> inputPreparationPipeline_;
    FrameGenerationEvaluation lastEvaluation_{};
    std::chrono::steady_clock::time_point lastFrameTime_{};
    FrameGenerationBackend activeBackend_ = FrameGenerationBackend::Off;
    uint32_t unavailableBackendMask_ = 0;
    bool initialized_ = false;
    bool historyValid_ = false;
};

}
