#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include <nlohmann/json.hpp>
#include <random>
#include <variant>

#include "core/render/modules/world/world_module.hpp"

class Framework;
struct FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;
class ShaderPack;

struct PostRenderModuleContext;

class PostRenderModule : public WorldModule, public SharedObject<PostRenderModule> {
    friend PostRenderModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.post_render.name";
    constexpr static uint32_t inputImageNum = 5;
    constexpr static uint32_t outputImageNum = 1;

    PostRenderModule();

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
    using ExecutionVariable = ShaderPack::ExecutionVariable;
    using ExecutionVariables = ShaderPack::ExecutionVariables;
    using PostRenderPassVariant = std::variant<std::shared_ptr<FullScreenPass>, std::shared_ptr<RenderPass>>;

    static constexpr uint32_t histSize = 256;
    static constexpr int weatherPostFlag = 0b0001;
    static constexpr int particlePostFlag = 0b0010;
    static constexpr int textPostFlag = 0b0100;
    static constexpr int nameTagPostFlag = 0b1000;
    static constexpr int worldOverlayPostFlag = 0b10000;

    static constexpr std::string_view TARGET_LDR = "out:ldr";
    static constexpr std::string_view TARGET_FIRST_HIT_DEPTH = "out:first_hit_depth";
    static constexpr std::string_view TARGET_HDR = "out:hdr";
    static constexpr std::string_view TARGET_MOTION_VECTOR = "out:motion_vector";
    static constexpr std::string_view TARGET_NORMAL_ROUGHNESS = "out:normal_roughness";
    // Match the pack resource so refraction copies and samples at post-render resolution.
    static constexpr std::string_view TARGET_POST_SCENE_COLOR = "post_scene_color";

    struct PostRenderExecutionRuntime;
    struct PostRenderPassExecutionOp;

    void initDescriptorTables();
    void initImages();
    void initBuffers();
    void initRenderPass();
    void initFrameBuffers();
    void initPipeline();
    void initExecutionVariables();
    void ensureDynamicPipelines();
    void compileExecutionProgram();
    std::vector<ExpressionEvaluator::Variable> executionExpressionVariables() const;
    double evaluateNumericExpression(const std::string &expression, const ExecutionVariables &variables);
    std::optional<std::reference_wrapper<ShaderPackLoader::VariableConfig>>
    findExecutionVariableConfig(std::string_view name);
    void uploadExecutionBuffer(const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                               PostRenderModuleContext &context,
                               const std::unordered_map<std::string, ExecutionVariable> &variables);
    void bindPostRefractionSourceImage(uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalImage> findBuiltInImage(const std::string &name, uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalImage> findTargetImage(const std::string &target, uint32_t frameIndex);
    static RenderPass::Target parseRenderContent(const std::string &content);
    static int renderTargetPostFlag(RenderPass::Target target);
    static VkCompareOp parseDepthCompare(const std::string &value);
    static const RenderPass::ShaderVariant *findShaderVariant(const RenderPass &pass, std::string_view contentName);
    static float sampleUnitFloat(std::mt19937 &randomGenerator);
    static glm::vec3 sampleUnitSphereDirection(std::mt19937 &randomGenerator);
    static vk::VertexFormat::EntityPBRVertex buildStarVertex(glm::vec3 direction, glm::vec4 color);
    uint32_t scaledInputExtent(uint32_t outputExtent) const;
    void updateInputExtent();

  private:
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> ldrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> worldPostDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> postRefractionSourceImages_;
    std::vector<std::shared_ptr<vk::Sampler>> postPassColorSamplers_;
    std::vector<std::shared_ptr<vk::Sampler>> postPassDepthSamplers_;
    std::vector<std::shared_ptr<vk::Sampler>> postRefractionSourceSamplers_;

    std::vector<std::shared_ptr<vk::Sampler>> samplers_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    std::shared_ptr<vk::Shader> worldPostColorToDepthVertShader_;
    std::shared_ptr<vk::Shader> worldPostColorToDepthFragShader_;
    std::shared_ptr<vk::RenderPass> worldPostColorToDepthRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldPostColorToDepthFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> worldPostColorToDepthPipeline_;

    std::shared_ptr<vk::DeviceLocalBuffer> starFieldVertexBuffer;
    std::shared_ptr<vk::Shader> fullScreenVertexShader_;

    std::shared_ptr<ShaderPack> shaderPack_;
    std::unordered_map<std::string, ShaderPackLoader::VariableConfig> executionVariableConfigs_;
    std::unordered_map<std::string, std::string> globalVariables_;
    std::vector<std::shared_ptr<FullScreenPass>> fullScreenPasses_;
    std::unordered_map<std::string, std::shared_ptr<FullScreenPass>> passNameToPass_;
    std::vector<std::shared_ptr<RenderPass>> renderPasses_;
    std::unordered_map<std::string, std::shared_ptr<RenderPass>> renderPassNameToPass_;
    std::unique_ptr<ShaderPack::SequentialOp> compiledExecutionProgram_;
    bool isDynamicPipelinesReady_ = false;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> postRenderedImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t starCount_ = 3000;
    float starSizeMin_ = 0.5f;
    float starSizeMax_ = 0.7f;
    float starRadius_ = 400.0f;
    float renderPrecision_ = 1.0f;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t inputWidth_ = 0;
    uint32_t inputHeight_ = 0;
};

struct PostRenderModuleContext : public WorldModuleContext, SharedObject<PostRenderModuleContext> {
    std::weak_ptr<PostRenderModule> postRenderModule;

    std::shared_ptr<vk::DeviceLocalImage> ldrImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> segmentationMaskImage;

    std::shared_ptr<vk::DeviceLocalImage> worldPostDepthImage;
    std::shared_ptr<vk::DescriptorTable> descriptorTable;
    std::shared_ptr<vk::Framebuffer> worldPostColorToDepthFramebuffer;

    std::shared_ptr<vk::DeviceLocalImage> postRenderedImage;
    std::shared_ptr<vk::DeviceLocalImage> outputImage;

    PostRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<PostRenderModule> postRenderModule);

    void render() override;
};
