#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

class ExpressionEvaluator {
  public:
    struct Variable {
        std::string name;
        double value = 0.0;
    };

    class CompiledExpression {
      public:
        CompiledExpression();
        CompiledExpression(const std::string &expression,
                           const std::vector<std::string> &variableNames,
                           const std::vector<Variable> &staticVariables);
        ~CompiledExpression();

        CompiledExpression(CompiledExpression &&) noexcept;
        CompiledExpression &operator=(CompiledExpression &&) noexcept;
        CompiledExpression(const CompiledExpression &) = delete;
        CompiledExpression &operator=(const CompiledExpression &) = delete;

        double evaluate(const std::vector<Variable> &variables) const;

      private:
        struct CompiledExpressionState;
        std::unique_ptr<CompiledExpressionState> state_;
    };

    static bool isValidIdentifier(std::string_view name);
    static bool isReservedIdentifier(std::string_view name);
    static double evaluate(const std::string &expression, const std::vector<Variable> &variables);

  private:
    static double trueValue();
    static double falseValue();
    static double ifValue(double condition, double ifTrueValue, double ifFalseValue);
};

class ShaderPackLoader {
  public:
    enum class Stage {
        RayTracing,
        PostRender,
    };

    enum class AccelerationStructureBuildMode {
        Sync,
        Deferred,
    };

    enum class NrdMode {
        Probabilistic,
        Checkerboard,
        HalfRateCheckerboard,
    };

    struct DefineConfig {
        std::optional<std::string> direct;
        std::unordered_map<std::string, std::string> expressions;
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> variants;
    };

    struct AttributeConfig {
        struct VisibilityCondition {
            std::string attribute;
            std::string module;
            std::vector<std::string> values;
        };

        struct PresetConfig {
            std::string customValue;
            std::unordered_map<std::string, std::unordered_map<std::string, std::string>> profiles;
        };

        std::string name;
        std::string type;
        std::string defaultValue;
        std::string description;
        std::string group;
        std::string performanceImpact;
        DefineConfig define;
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> linkedValues;
        std::optional<PresetConfig> preset;
        std::vector<VisibilityCondition> visibleWhen;
    };

    struct VariableConfig {
        std::string name;
        std::string type;
        std::string defaultValue;
    };

    enum class TextureDimension {
        Texture2D,
        Texture2DArray,
        Texture3D,
        Cube,
    };

    struct TextureConfig {
        std::string name;
        bool imported = false;
        bool shared = false;
        TextureDimension dimension = TextureDimension::Texture2D;
        fs::path sourcePath;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::string widthExpression;
        std::string heightExpression;
        std::string depthExpression;
        uint32_t importedWidth = 0;
        uint32_t importedHeight = 0;
        uint32_t importedDepth = 1;
        float scale = 0.0f;
        std::optional<uint32_t> sampledBinding;
        std::optional<uint32_t> storageBinding;
        bool transferSrc = false;
        VkFilter filter = VK_FILTER_LINEAR;
        VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    };

    struct BufferConfig {
        std::string name;
        bool shared = false;
        std::string sizeExpression;
        uint32_t binding = 0;
    };

    enum class ResourceAccessKind {
        Read,
        Write,
        ReadWrite,
        Overwrite,
        Append,
        AppendCounter,
        IndirectRead,
    };

    struct ResourceAccess {
        std::string name;
        ResourceAccessKind kind = ResourceAccessKind::Read;
    };

    struct ResourceList {
        std::vector<std::string> images;
        std::vector<std::string> buffers;
        std::vector<ResourceAccess> imageAccesses;
        std::vector<ResourceAccess> bufferAccesses;
    };

    struct HitGroupConfig {
        std::string name;
        std::optional<fs::path> anyHit;
        std::optional<fs::path> closestHit;
        std::optional<fs::path> intersection;
        VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        std::unordered_map<std::string, std::string> definitions;
    };

    struct MissShaderConfig {
        std::string name;
        fs::path shaderPath;
    };

    struct FullScreenPassConfig {
        std::string name;
        fs::path fragmentShaderPath;
        std::string target;
        ResourceList inputs;
        ResourceList outputs;
        std::optional<uint32_t> face;
        std::optional<uint32_t> layer;
        std::unordered_map<std::string, std::string> definitions;
    };

    struct RenderShaderConfig {
        fs::path vertexShaderPath;
        fs::path fragmentShaderPath;
    };

    struct RenderPipelineConfig {
        RenderShaderConfig shader;
        bool depthTest = false;
        bool depthWrite = false;
        std::string depthCompare;
        bool colorBlend = false;
        std::vector<std::string> colorWriteMasks;
        std::string cull;
        std::string frontFace;
        std::unordered_map<std::string, std::string> definitions;
    };

    struct RenderPassConfig {
        struct IndirectDrawConfig {
            std::vector<std::string> batchSortLayers;
        };

        std::string name;
        std::unordered_map<std::string, RenderPipelineConfig> pipelineConfigs;
        std::vector<std::string> layerOrder;
        std::string content;
        ResourceList inputs;
        ResourceList outputs;
        bool depthLoad = false;
        std::unordered_map<std::string, std::array<uint32_t, 4>> clearOutputs;
        std::optional<IndirectDrawConfig> indirectDraw;
        std::unordered_map<std::string, std::string> definitions;
    };

    struct RayTracingPassConfig {
        std::string name;
        std::string asyncGroup;
        bool querySharc = false;
        fs::path rayGenShaderPath;
        std::string defaultHitGroupName;
        ResourceList inputs;
        ResourceList outputs;
        std::optional<std::string> indirectArgsBuffer;
        std::string width = "RENDER_WIDTH";
        std::string height = "RENDER_HEIGHT";
        std::string depth = "1";
        uint32_t maxRayRecursionDepth = 1;
        std::vector<HitGroupConfig> hitGroups;
        std::vector<MissShaderConfig> missShaders;
        std::unordered_map<std::string, std::string> definitions;
    };

    struct ComputePassConfig {
        std::string name;
        std::string asyncGroup;
        bool querySharc = false;
        fs::path computeShaderPath;
        ResourceList inputs;
        ResourceList outputs;
        std::optional<std::string> indirectArgsBuffer;
        std::string gx = "1";
        std::string gy = "1";
        std::string gz = "1";
        std::unordered_map<std::string, std::string> definitions;
    };

    struct PassConfig {
        enum class Type {
            FullScreen,
            Render,
            RayTracing,
            Compute,
        };

        Type type = Type::RayTracing;
        Stage stage = Stage::RayTracing;
        FullScreenPassConfig fullScreen;
        RenderPassConfig render;
        RayTracingPassConfig rayTracing;
        ComputePassConfig compute;
    };

    struct ExecutionCommand {
        struct Assignment {
            std::optional<std::string> type;
            bool explicitType = false;
            nlohmann::json value;
        };

        struct BufferUpdate {
            std::string bufferName;
            uint32_t offset = 0;
            std::vector<uint32_t> uint32Values;
        };

        enum class Type {
            Assign,
            BufferUpdate,
            SharcUpdate,
            Pass,
            IfElse,
            While,
        };

        Type type = Type::Pass;
        std::unordered_map<std::string, Assignment> assignments;
        BufferUpdate bufferUpdate;
        std::string passName;
        nlohmann::json condition;
        std::vector<ExecutionCommand> thenCommands;
        std::vector<ExecutionCommand> elseCommands;
    };

    struct ExecutionConfig {
        std::vector<VariableConfig> globalVariables;
        std::vector<VariableConfig> variables;
        std::vector<ExecutionCommand> commands;
        bool hasSharcUpdateCommand = false;
    };

    struct SharcConfig {
        std::string updatePassName;
        fs::path resolveCompShaderPath;
    };

    struct ShaderPack {
        fs::path rootPath;
        bool requiresEmission = false;
        bool segmentationMaskEnabled = false;
        AccelerationStructureBuildMode accelerationStructureBuildMode = AccelerationStructureBuildMode::Sync;
        NrdMode nrdMode = NrdMode::Probabilistic;
        std::vector<std::string> includeDirectories;
        std::vector<AttributeConfig> attributes;
        std::unordered_map<std::string, std::string> translations;
        std::vector<TextureConfig> textures;
        std::vector<BufferConfig> buffers;
        std::vector<PassConfig> passes;
        std::optional<SharcConfig> sharc;
        ExecutionConfig rayTracingExecution;
        ExecutionConfig postRenderExecution;
    };

    struct LoadResult {
        bool success = false;
        ShaderPack shaderPack;
        std::string error;
    };

    static constexpr char KEY_ATTRIBUTES[] = "attributes";
    static constexpr char KEY_REQUIRES_EMISSION[] = "requires_emission";
    static constexpr char KEY_SEGMENTATION_MASK[] = "segmentation_mask";
    static constexpr char KEY_ENABLED[] = "enabled";
    static constexpr char KEY_TEXTURES[] = "textures";
    static constexpr char KEY_BUFFERS[] = "buffers";
    static constexpr char KEY_PASSES[] = "passes";
    static constexpr char KEY_NAME[] = "name";
    static constexpr char KEY_TYPE[] = "type";
    static constexpr char KEY_DEFAULT_VALUE[] = "default_value";
    static constexpr char KEY_DESCRIPTION[] = "description";
    static constexpr char KEY_GROUP[] = "group";
    static constexpr char KEY_PERFORMANCE_IMPACT[] = "performance_impact";
    static constexpr char KEY_LINKED_VALUES[] = "linked_values";
    static constexpr char KEY_PRESET[] = "preset";
    static constexpr char KEY_CUSTOM_VALUE[] = "custom_value";
    static constexpr char KEY_PROFILES[] = "profiles";
    static constexpr char KEY_VISIBLE_WHEN[] = "visible_when";
    static constexpr char KEY_ATTRIBUTE[] = "attribute";
    static constexpr char KEY_MODULE[] = "module";
    static constexpr char KEY_VALUES[] = "values";
    static constexpr char KEY_DEFINE[] = "define";
    static constexpr char KEY_DEFINES[] = "defines";
    static constexpr char KEY_DEFINITIONS[] = "definitions";
    static constexpr char KEY_DIMENSION[] = "dimension";
    static constexpr char KEY_PATH[] = "path";
    static constexpr char KEY_FORMAT[] = "format";
    static constexpr char KEY_WIDTH[] = "width";
    static constexpr char KEY_HEIGHT[] = "height";
    static constexpr char KEY_DEPTH[] = "depth";
    static constexpr char KEY_SIZE[] = "size";
    static constexpr char KEY_SCALE[] = "scale";
    static constexpr char KEY_SHARED[] = "shared";
    static constexpr char KEY_FILTER[] = "filter";
    static constexpr char KEY_ADDRESS_MODE[] = "address_mode";
    static constexpr char KEY_BINDING[] = "binding";
    static constexpr char KEY_SAMPLED_BINDING[] = "sampled_binding";
    static constexpr char KEY_STORAGE_BINDING[] = "storage_binding";
    static constexpr char KEY_TRANSFER_SRC[] = "transfer_src";
    static constexpr char KEY_VERTEX[] = "vertex";
    static constexpr char KEY_FRAGMENT[] = "fragment";
    static constexpr char KEY_COMPUTE[] = "compute";
    static constexpr char KEY_TARGET[] = "target";
    static constexpr char KEY_CONTENT[] = "content";
    static constexpr char KEY_INPUTS[] = "inputs";
    static constexpr char KEY_OUTPUTS[] = "outputs";
    static constexpr char KEY_IMAGES[] = "images";
    static constexpr char KEY_ACCESS[] = "access";
    static constexpr char KEY_FACE[] = "face";
    static constexpr char KEY_LAYER[] = "layer";
    static constexpr char KEY_GX[] = "gx";
    static constexpr char KEY_GY[] = "gy";
    static constexpr char KEY_GZ[] = "gz";
    static constexpr char KEY_SHARC[] = "sharc";
    static constexpr char KEY_UPDATE_PASS[] = "update_pass";
    static constexpr char KEY_RESOLVE_COMP[] = "resolve_comp";
    static constexpr char KEY_QUERY_SHARC[] = "query_sharc";
    static constexpr char KEY_ASYNC_GROUP[] = "async_group";
    static constexpr char KEY_INDIRECT_ARGS[] = "indirect_args";
    static constexpr char KEY_RGEN[] = "rgen";
    static constexpr char KEY_MAX_RAY_RECURSION_DEPTH[] = "max_ray_recursion_depth";
    static constexpr char KEY_DEFAULT_HIT_GROUP[] = "default_hit_group";
    static constexpr char KEY_HIT_GROUPS[] = "hit_groups";
    static constexpr char KEY_SHADERS[] = "shaders";
    static constexpr char KEY_RCHIT[] = "rchit";
    static constexpr char KEY_RAHIT[] = "rahit";
    static constexpr char KEY_RINT[] = "rint";
    static constexpr char KEY_MISS[] = "miss";
    static constexpr char KEY_SHADER[] = "shader";
    static constexpr char KEY_PIPELINE[] = "pipeline";
    static constexpr char KEY_LAYER_ORDER[] = "layer_order";
    static constexpr char KEY_LANG[] = "lang";
    static constexpr char KEY_EXECUTION[] = "execution";
    static constexpr char KEY_EXECUTION_POST[] = "execution_post";
    static constexpr char KEY_GLOBAL_VARIABLES[] = "global_variables";
    static constexpr char KEY_COMMANDS[] = "commands";
    static constexpr char KEY_ASSIGN[] = "assign";
    static constexpr char KEY_BUFFER_UPDATE[] = "buffer_update";
    static constexpr char KEY_BUFFER[] = "buffer";
    static constexpr char KEY_OFFSET[] = "offset";
    static constexpr char KEY_UINT32_VALUES[] = "uint32_values";
    static constexpr char KEY_SHARC_UPDATE[] = "sharc_update";
    static constexpr char KEY_PASS[] = "pass";
    static constexpr char KEY_IF_ELSE[] = "if_else";
    static constexpr char KEY_WHILE[] = "while";
    static constexpr char KEY_CONDITION[] = "condition";
    static constexpr char KEY_THEN[] = "then";
    static constexpr char KEY_ELSE[] = "else";
    static constexpr char KEY_VALUE[] = "value";
    static constexpr char KEY_STAGE[] = "stage";
    static constexpr char KEY_DEPTH_TEST[] = "depth_test";
    static constexpr char KEY_DEPTH_WRITE[] = "depth_write";
    static constexpr char KEY_DEPTH_COMPARE[] = "depth_compare";
    static constexpr char KEY_DEPTH_LOAD[] = "depth_load";
    static constexpr char KEY_CLEAR_OUTPUTS[] = "clear_outputs";
    static constexpr char KEY_COLOR_BLEND[] = "color_blend";
    static constexpr char KEY_COLOR_WRITE_MASK[] = "color_write_mask";
    static constexpr char KEY_CULL[] = "cull";
    static constexpr char KEY_FRONT_FACE[] = "front_face";
    static constexpr char KEY_INDIRECT_DRAW[] = "indirect_draw";
    static constexpr char KEY_COMMANDS_BUFFER[] = "commands_buffer";
    static constexpr char KEY_WORKGROUP_SIZE[] = "workgroup_size";
    static constexpr char KEY_BATCH_SORT_LAYERS[] = "batch_sort_layers";
    static constexpr char KEY_RAY_TRACING[] = "ray_tracing";
    static constexpr char KEY_POST_RENDER[] = "post_render";
    static constexpr char KEY_SYNC_ACCELERATION_STRUCTURES[] = "sync_acceleration_structures";
    static constexpr char KEY_ACCELERATION_STRUCTURE_BUILD_MODE[] = "acceleration_structure_build_mode";
    static constexpr char KEY_NRD_MODE[] = "nrd_mode";

    static constexpr char VALUE_FULL_SCREEN[] = "full_screen";
    static constexpr char VALUE_RENDER[] = "render";
    static constexpr char VALUE_RAY_TRACING[] = "ray_tracing";
    static constexpr char VALUE_COMPUTE[] = "compute";
    static constexpr char VALUE_SYNC[] = "sync";
    static constexpr char VALUE_DEFERRED[] = "deferred";
    static constexpr char VALUE_NRD_PROBABILISTIC[] = "probabilistic";
    static constexpr char VALUE_NRD_CHECKERBOARD[] = "checkerboard";
    static constexpr char VALUE_NRD_HALF_RATE_CHECKERBOARD[] = "half_rate_checkerboard";
    static constexpr char VALUE_FILE[] = "file";
    static constexpr char VALUE_INTERMEDIATE[] = "intermediate";
    static constexpr char VALUE_2D[] = "2d";
    static constexpr char VALUE_2D_ARRAY[] = "2d_array";
    static constexpr char VALUE_3D[] = "3d";
    static constexpr char VALUE_CUBE[] = "cube";
    static constexpr char VALUE_TRIANGLE[] = "triangle";
    static constexpr char VALUE_AABB[] = "aabb";
    static constexpr uint32_t EXECUTION_BINDING = 0;

    static LoadResult load(const fs::path &path, const fs::path &builtInPath, const std::string &language);

    static std::unordered_map<std::string, std::string>
    buildAttributeDefinitions(const std::vector<AttributeConfig> &attributes,
                              const std::unordered_map<std::string, std::string> &currentValues);
    static std::string buildExecutionSource(const std::vector<VariableConfig> &variables, uint32_t executionSet);
    static std::string toLower(std::string value);

  private:
    struct ClassifiedHitShaderPaths {
        std::optional<fs::path> anyHit;
        std::optional<fs::path> closestHit;
        std::optional<fs::path> intersection;
    };

    static std::string trim(std::string value);
    static std::string stableFolderName(const fs::path &path);
    static bool extractZip(const fs::path &zipPath, const fs::path &destinationPath, std::string &error);
    static fs::path resolveShaderPackRoot(const fs::path &path, std::string &error);
    static std::optional<std::reference_wrapper<const nlohmann::json>>
    findKey(const nlohmann::json &value, std::initializer_list<std::string_view> keys);
    static const nlohmann::json &
    requireObject(const nlohmann::json &value, std::string_view key, const std::string &context);
    static const nlohmann::json &
    requireArray(const nlohmann::json &value, std::string_view key, const std::string &context);
    static std::string requireString(const nlohmann::json &value, std::string_view key, const std::string &context);
    static std::optional<std::string>
    optionalString(const nlohmann::json &value, std::string_view key, const std::string &context);
    static bool
    optionalBool(const nlohmann::json &value, std::string_view key, bool fallback, const std::string &context);
    static VkFormat parseFormat(const std::string &value);
    static VkFilter parseFilter(const std::string &value);
    static VkSamplerAddressMode parseAddressMode(const std::string &value);
    static TextureDimension parseTextureDimension(const std::string &value);
    static AccelerationStructureBuildMode parseAccelerationStructureBuildMode(const std::string &value,
                                                                              const std::string &context);
    static NrdMode parseNrdMode(const std::string &value, const std::string &context);
    static AccelerationStructureBuildMode
    inferAccelerationStructureBuildMode(const ShaderPackLoader::ShaderPack &shaderPack);
    static std::unordered_map<std::string, std::string> parseTranslations(const fs::path &langPath);
    static std::unordered_map<std::string, std::string> loadLanguage(const fs::path &rootPath,
                                                                     const std::string &language);
    static std::vector<std::string>
    parseStringArray(const nlohmann::json &value, std::string_view key, const std::string &context);
    static fs::path
    resolveRelativeFile(const fs::path &rootPath, const std::string &reference, const std::string &context);
    static std::vector<fs::path> collectShaderFiles(const fs::path &rootPath);
    static std::unordered_map<std::string, ClassifiedHitShaderPaths>
    classifyHitShaders(const std::vector<fs::path> &shaderFiles);
    static std::unordered_map<std::string, std::string> parseDefinitions(const nlohmann::json &jsonValue,
                                                                         const std::string &context);
    static std::optional<std::string> inferScalarType(const nlohmann::json &value, bool allowString);
    static std::string parseScalarValue(const nlohmann::json &value, const std::string &context);
    static std::string parseNumericExpressionValue(const nlohmann::json &value, const std::string &context);
    static DefineConfig parseAttributeDefinition(const nlohmann::json &attributeJson, const std::string &context);
    static AttributeConfig parseAttributeConfig(const nlohmann::json &attributeJson, const std::string &context);
    static VariableConfig parseVariableConfig(const nlohmann::json &variableJson, const std::string &context);
    static TextureConfig
    parseTextureConfig(const nlohmann::json &textureJson, const fs::path &rootPath, const std::string &context);
    static BufferConfig parseBufferConfig(const nlohmann::json &bufferJson, const std::string &context);
    static Stage parseStage(const nlohmann::json &value, const std::string &context);
    static ResourceAccessKind parseResourceAccessKind(const std::string &value, const std::string &context);
    static ResourceList
    parsePassResourceList(const nlohmann::json &value, std::string_view key, const std::string &context);
    static FullScreenPassConfig
    parseFullScreenPassConfig(const nlohmann::json &passJson, const fs::path &rootPath, const std::string &context);
    static RenderShaderConfig
    parseRenderShaderConfig(const nlohmann::json &shaderJson, const fs::path &rootPath, const std::string &context);
    static RenderPipelineConfig
    parseRenderPipelineConfig(const nlohmann::json &pipelineJson, const fs::path &rootPath, const std::string &context);
    static RenderPassConfig
    parseRenderPassConfig(const nlohmann::json &passJson, const fs::path &rootPath, const std::string &context);
    static RayTracingPassConfig
    parseRayTracingPassConfig(const nlohmann::json &passJson,
                              const fs::path &rootPath,
                              const std::unordered_map<std::string, ClassifiedHitShaderPaths> &classifiedHitShaders,
                              const std::string &context);
    static ComputePassConfig
    parseComputePassConfig(const nlohmann::json &passJson, const fs::path &rootPath, const std::string &context);
    static PassConfig
    parsePassConfig(const nlohmann::json &passJson,
                    const fs::path &rootPath,
                    const std::unordered_map<std::string, ClassifiedHitShaderPaths> &classifiedHitShaders,
                    const std::string &context);
    static ExecutionCommand parseExecutionCommand(const nlohmann::json &commandJson, const std::string &context);
    static std::vector<ExecutionCommand> parseExecutionCommands(const nlohmann::json &commandsJson,
                                                                const std::string &context);
    static ExecutionConfig parseExecutionConfig(const nlohmann::json &configJson, const std::string &context);
    static ShaderPack parseConfigFile(const fs::path &rootPath, const std::string &language);
    static void validateExecutionCommands(const std::vector<ExecutionCommand> &commands,
                                          Stage stage,
                                          const std::optional<SharcConfig> &sharc,
                                          const std::string &context,
                                          bool &hasSharcUpdateCommand);
    static void collectExecutionVariables(const std::vector<ExecutionCommand> &commands,
                                          std::vector<VariableConfig> &variables,
                                          std::unordered_map<std::string, size_t> &variableIndices,
                                          std::unordered_map<std::string, bool> &placeholderTypes);
    static uint64_t calculateFnv1a64Hash(std::string_view text);
    static std::optional<std::string> inferAssignmentType(const ExecutionCommand::Assignment &assignment);
    static std::string defaultValueForType(const std::string &type);
    static std::string parseValue(const std::string &type, const std::string &currentValue);
    static std::string substitutePlaceholder(const std::string &expression, const std::string &value);
    static void applyDefinitionExpressions(std::unordered_map<std::string, std::string> &definitions,
                                           const std::unordered_map<std::string, std::string> &expressions,
                                           const std::string &value);
};

class Framework;

struct HitShaderGroup {
    std::string name;
    std::shared_ptr<vk::Shader> closestHitShader;
    std::shared_ptr<vk::Shader> anyHitShader;
    std::shared_ptr<vk::Shader> intersectionShader;
    VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
};

struct MissShader {
    std::string name;
    uint32_t index = 0;
    std::shared_ptr<vk::Shader> shader;
};

struct FullScreenPass {
    ShaderPackLoader::FullScreenPassConfig config;
    std::shared_ptr<vk::RenderPass> renderPass;
    std::vector<std::vector<std::shared_ptr<vk::Framebuffer>>> framebuffers;
    std::vector<std::shared_ptr<vk::Shader>> fragmentShaders;
    std::vector<std::shared_ptr<vk::GraphicsPipeline>> pipelines;
    std::shared_ptr<vk::DeviceLocalBuffer> executionBuffer;
};

struct RenderPass {
    enum class Target {
        Weather,
        Particle,
        Text,
        NameTag,
        WorldOverlay,
        Star,
    };

    struct ShaderVariant {
        std::shared_ptr<vk::Shader> vertexShader;
        std::shared_ptr<vk::Shader> fragmentShader;
        std::shared_ptr<vk::GraphicsPipeline> pipeline;
        ShaderPackLoader::RenderPipelineConfig config;
    };

    ShaderPackLoader::RenderPassConfig config;
    Target target = Target::Particle;
    bool writesFirstHitDepth = false;
    bool rendersPrimaryRaster = false;
    bool rendersHandOnly = false;
    std::shared_ptr<vk::RenderPass> renderPass;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers;
    std::unordered_map<std::string, ShaderVariant> shaderVariants;
    std::shared_ptr<vk::Shader> indirectCommandComputeShader;
    std::shared_ptr<vk::ComputePipeline> indirectCommandComputePipeline;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indirectMetadataBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indirectCommandBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> handMetadataBuffers;
    std::shared_ptr<vk::DeviceLocalBuffer> executionBuffer;
};

struct RayTracingPass {
    ShaderPackLoader::RayTracingPassConfig config;
    bool querySharcEnabled = false;
    bool isSharcUpdatePass = false;
    std::vector<MissShader> missShaders;
    std::vector<HitShaderGroup> hitShaderGroups;
    std::unordered_map<std::string, uint32_t> hitGroupNameToIndex;
    uint32_t shadowHitGroupIndex = 0;
    uint32_t fallbackHitGroupIndex = 0;
    uint32_t missGroupCount = 0;
    uint32_t hitGroupCount = 0;
    std::shared_ptr<vk::Shader> rayGenUpdateShader;
    std::shared_ptr<vk::Shader> rayGenQueryShader;
    std::shared_ptr<vk::RayTracingPipeline> updatePipeline;
    std::shared_ptr<vk::RayTracingPipeline> queryPipeline;
    std::vector<std::shared_ptr<vk::SBT>> updateSbts;
    std::vector<std::shared_ptr<vk::SBT>> querySbts;
    std::shared_ptr<vk::Shader> sharcResolveCompShader;
    std::shared_ptr<vk::ComputePipeline> sharcResolvePipeline;
    std::shared_ptr<vk::DeviceLocalBuffer> executionBuffer;

    struct HitAssignment {
        size_t groupIndex;
        int shaderType;
    };
    size_t missRequestCount_ = 0;
    size_t hitRequestCount_ = 0;
    size_t rayGenRequestCount_ = 0;
    size_t sharcRequestCount_ = 0;
    std::vector<HitAssignment> hitAssignments_;
};

struct ComputePass {
    ShaderPackLoader::ComputePassConfig config;
    std::shared_ptr<vk::Shader> computeShader;
    std::shared_ptr<vk::ComputePipeline> pipeline;
    std::shared_ptr<vk::DeviceLocalBuffer> executionBuffer;
};

class ShaderPack {
  public:
    struct BuildConfig {
        std::string shaderPackPath;
        bool shouldUseSharc = true;
        std::unordered_map<std::string, std::string> staticAttributes;
        std::unordered_map<std::string, std::string> injectedDefinitions;
        std::string language = "en_us";
    };

    struct RuntimeTexture {
        ShaderPackLoader::TextureConfig config;
        std::shared_ptr<vk::Sampler> sampler;
        std::shared_ptr<vk::DeviceLocalImage> importedImage;
        std::vector<std::shared_ptr<vk::DeviceLocalImage>> frameImages;
        uint32_t sampledViewIndex = 0;
        uint32_t singleLayerViewBaseIndex = 0;
    };

    struct RuntimeBuffer {
        ShaderPackLoader::BufferConfig config;
        std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> frameBuffers;
    };

    struct StageRuntime {
        std::unordered_map<std::string, ShaderPackLoader::VariableConfig> executionVariableConfigs;
        std::unordered_map<std::string, std::string> globalVariables;
    };

    struct ExecutionVariable {
        std::string name;
        std::string value;
        std::string type;
    };

    struct ShaderCreateInfo {
        fs::path path;
        VkShaderStageFlagBits stage;
        std::unordered_map<std::string, std::string> definitions;
        ShaderPackLoader::Stage executionStage = ShaderPackLoader::Stage::RayTracing;
        uint32_t executionSet = 0;
    };

#ifdef DEBUG
    struct ShaderBatchStats {
        size_t requestCount = 0;
        size_t uniqueShaderCount = 0;
        size_t cacheHitCount = 0;
        size_t cacheMissCount = 0;
        size_t cacheReadFailureCount = 0;
    };
#endif

    using ExecutionVariables = std::unordered_map<std::string, ExecutionVariable>;
    struct ExecutionRuntime {
        virtual ~ExecutionRuntime() = default;

        ExecutionVariables executionVariables;
        std::vector<ExpressionEvaluator::Variable> runtimeExpressionVariables;
        uint32_t loopLimit = 1u << 16;

        const std::vector<ExpressionEvaluator::Variable> &expressionVariables() const;
        void invalidateExpressionVariables() const;
        ExecutionVariable &findOrCreateVariable(const std::string &name,
                                                const ShaderPackLoader::VariableConfig &variableConfig);

      private:
        mutable bool expressionVariablesDirty = true;
        mutable std::vector<ExpressionEvaluator::Variable> cachedExpressionVariables;
    };

    struct ExecutionOp {
        virtual ~ExecutionOp() = default;
        virtual void operator()(ExecutionRuntime &runtime) const = 0;
    };

    struct SequentialOp final : ExecutionOp {
        std::vector<std::unique_ptr<ExecutionOp>> executionOps;

        void operator()(ExecutionRuntime &runtime) const override;
    };

    using ExecutionOpBuilder = std::function<std::unique_ptr<ExecutionOp>(const std::string &passName)>;
    using BufferUpdateExecutionOpBuilder =
        std::function<std::unique_ptr<ExecutionOp>(const ShaderPackLoader::ExecutionCommand::BufferUpdate &update)>;

    explicit ShaderPack(std::shared_ptr<Framework> framework);

    static std::filesystem::path builtInShaderPackPath();
    static std::filesystem::path emissionFallbackShaderPackPath();
    static BuildConfig buildConfigFromMainRenderAttributes(const std::vector<std::string> &attributeKVs);
    static std::shared_ptr<vk::DeviceLocalBuffer> createPassExecutionBuffer(std::shared_ptr<vk::Device> device,
                                                                            std::shared_ptr<vk::VMA> vma,
                                                                            size_t executionBufferSize);

    bool initialize(const BuildConfig &config, std::string &error);
    void ensureRuntimeResources(uint32_t referenceWidth, uint32_t referenceHeight);
    bool refreshRuntimeBuffers();
    void setRuntimeResourceExpressionVariables(std::vector<ExpressionEvaluator::Variable> variables);
    void bindRuntimeResources(const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                              uint32_t setIndex,
                              uint32_t frameIndex);
    void defineRuntimeResourceDescriptorSet(vk::DescriptorTableBuilder &builder,
                                            VkShaderStageFlags sampledImageStageFlags,
                                            VkShaderStageFlags storageImageStageFlags,
                                            VkShaderStageFlags storageBufferStageFlags) const;
    void defineExecutionDescriptorSet(vk::DescriptorTableBuilder &builder,
                                      ShaderPackLoader::Stage stage,
                                      VkShaderStageFlags stageFlags) const;
    void copyStageExecutionState(ShaderPackLoader::Stage stage,
                                 std::unordered_map<std::string, ShaderPackLoader::VariableConfig> &variableConfigs,
                                 std::unordered_map<std::string, std::string> &globalVariables) const;
    std::unordered_set<std::string> activePassNames(ShaderPackLoader::Stage stage, bool includeAttributeDefines) const;
    uint32_t executionSet(uint32_t runtimeResourceSet) const;
    bool hasRuntimeResources() const;
    void preClose();

    std::shared_ptr<vk::Shader> createShader(std::shared_ptr<vk::Device> device,
                                             const std::filesystem::path &path,
                                             VkShaderStageFlagBits stage,
                                             const std::unordered_map<std::string, std::string> &definitions,
                                             ShaderPackLoader::Stage executionStage,
                                             uint32_t executionSet) const;
#ifdef DEBUG
    std::vector<std::shared_ptr<vk::Shader>> createShaders(std::shared_ptr<vk::Device> device,
                                                           const std::vector<ShaderCreateInfo> &requests,
                                                           ShaderBatchStats *stats = nullptr) const;
#else
    std::vector<std::shared_ptr<vk::Shader>> createShaders(std::shared_ptr<vk::Device> device,
                                                           const std::vector<ShaderCreateInfo> &requests) const;
#endif

    const ShaderPackLoader::ShaderPack &shaderPack() const;
    bool hasSegmentationMask() const;
    ShaderPackLoader::AccelerationStructureBuildMode accelerationStructureBuildMode() const;
    const std::unordered_map<std::string, std::string> &shaderAttributes() const;
    ShaderPackLoader::NrdMode nrdMode() const;
    const StageRuntime &stageRuntime(ShaderPackLoader::Stage stage) const;
    const ShaderPackLoader::ExecutionConfig &execution(ShaderPackLoader::Stage stage) const;
    std::string executionSource(ShaderPackLoader::Stage stage, uint32_t executionSet) const;
    double evaluateNumericExpression(ShaderPackLoader::Stage stage,
                                     const std::string &expression,
                                     const ExecutionVariables &variables,
                                     const std::vector<ExpressionEvaluator::Variable> &runtimeExpressionVariables,
                                     bool includeAttributeDefines) const;
    double evaluateNumericExpression(ShaderPackLoader::Stage stage,
                                     const std::string &expression,
                                     const std::vector<ExpressionEvaluator::Variable> &expressionVariables,
                                     bool includeAttributeDefines) const;
    void uploadExecutionBuffer(ShaderPackLoader::Stage stage,
                               const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                               const ExecutionVariables &variables,
                               const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                               const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                               uint32_t executionSet,
                               uint32_t queueIndex,
                               VkShaderStageFlags shaderStages) const;
    std::unique_ptr<SequentialOp>
    compileExecutionProgram(ShaderPackLoader::Stage stage,
                            bool includeAttributeDefines,
                            const ExecutionOpBuilder &passExecutionOpBuilder,
                            const ExecutionOpBuilder &sharcUpdateExecutionOpBuilder = {},
                            const BufferUpdateExecutionOpBuilder &bufferUpdateExecutionOpBuilder = {}) const;

    bool hasSharcRuntime() const;
    bool runtimeResourcesReady() const;

    std::optional<std::reference_wrapper<RuntimeTexture>> findRuntimeTexture(std::string_view name);
    std::optional<std::reference_wrapper<const RuntimeTexture>> findRuntimeTexture(std::string_view name) const;
    std::optional<std::reference_wrapper<RuntimeBuffer>> findRuntimeBuffer(std::string_view name);
    std::optional<std::reference_wrapper<const RuntimeBuffer>> findRuntimeBuffer(std::string_view name) const;

    std::shared_ptr<vk::DeviceLocalImage> findRuntimeVKTexture(RuntimeTexture &runtimeTexture, uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalImage> findRuntimeVKTexture(const RuntimeTexture &runtimeTexture,
                                                               uint32_t frameIndex) const;
    uint32_t runtimeTextureSingleLayerViewIndex(const RuntimeTexture &runtimeTexture, uint32_t layer) const;
    std::shared_ptr<vk::DeviceLocalBuffer> findRuntimeVKBuffer(RuntimeBuffer &runtimeBuffer, uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalBuffer> findRuntimeVKBuffer(const RuntimeBuffer &runtimeBuffer,
                                                               uint32_t frameIndex) const;

  private:
    static std::unordered_map<std::string, std::string>
    mergeDefinitions(const std::unordered_map<std::string, std::string> &lhs,
                     const std::unordered_map<std::string, std::string> &rhs);

    double evaluateNumericExpression(const std::string &expression) const;
    std::vector<ExpressionEvaluator::Variable> staticExpressionVariables(bool includeAttributeDefines) const;
    std::vector<std::string> expressionVariableNames(ShaderPackLoader::Stage stage, bool includeAttributeDefines) const;
    double evaluateCachedNumericExpression(ShaderPackLoader::Stage stage,
                                           const std::string &expression,
                                           const std::vector<ExpressionEvaluator::Variable> &expressionVariables,
                                           bool includeAttributeDefines) const;
    struct CompiledExecutionCondition;
    struct ActiveRuntimeResources {
        std::unordered_set<std::string> textures;
        std::unordered_set<std::string> buffers;

        bool hasTexture(const std::string &name) const;
        bool hasBuffer(const std::string &name) const;
    };
    struct AssignmentExecutionOp;
    struct IfExecutionOp;
    struct WhileExecutionOp;
    static std::string trimString(std::string_view text);
    static std::optional<std::string> unwrapScalarConstructor(std::string_view expression,
                                                              std::string_view constructorName);
    static std::string parseAttributeValueForDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                              const std::string &currentValue);
    static std::string substitutePlaceholderInDefineExpression(const std::string &expression,
                                                               const std::string &placeholderValue);
    static std::optional<double> tryEvaluateScalarDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                                   const std::string &currentValue,
                                                                   const std::string &expression);
    static bool parseExecutionBoolValue(const std::string &rawValue);
    static bool isExecutionIntType(const std::string &type);
    static bool isExecutionFloatType(const std::string &type);
    static bool isExecutionNumericExpressionType(const std::string &type);
    static std::string formatExecutionFloat(double numericValue);
    static double parseExecutionNumber(const std::string &name, const std::string &type, const std::string &rawValue);
    static std::string formatExecutionValue(const std::string &type, double numericValue);
    static std::string defaultExecutionValueForType(const std::string &type);
    static bool isShaderIdentifierStart(char ch);
    static bool isShaderIdentifierContinue(char ch);
    static std::string stripShaderComments(std::string_view line, bool &insideBlockComment);
    static void collectShaderAttributeIdentifiers(std::string_view source,
                                                  const std::unordered_set<std::string> &attributeKeys,
                                                  std::set<std::string> &usedAttributes);
    static std::optional<std::string> parseShaderInclude(std::string_view line);
    static std::optional<std::filesystem::path>
    resolveShaderInclude(const std::filesystem::path &requestingPath,
                         std::string_view includePath,
                         const std::vector<std::filesystem::path> &includeDirectories);
    static std::optional<std::set<std::string>> collectShaderAttributeDependencies(
        const std::filesystem::path &shaderPath,
        const std::unordered_set<std::string> &attributeKeys,
        const std::vector<std::filesystem::path> &includeDirectories,
        std::unordered_map<std::string, std::optional<std::set<std::string>>> &dependencyCache,
        std::unordered_set<std::string> &activeFiles);
    static std::unordered_map<std::string, std::string>
    filterShaderAttributes(const std::unordered_map<std::string, std::string> &allAttributes,
                           const std::filesystem::path &shaderPath,
                           const std::unordered_set<std::string> &attributeKeys,
                           const std::vector<std::filesystem::path> &includeDirectories,
                           std::unordered_map<std::string, std::optional<std::set<std::string>>> &dependencyCache);
    static std::unordered_map<std::string, std::string>
    stripAttributeDefinitions(const std::unordered_map<std::string, std::string> &definitions,
                              const std::unordered_set<std::string> &attributeKeys);
    static CompiledExecutionCondition
    compileExecutionCondition(const nlohmann::json &condition,
                              const std::vector<std::string> &variableNames,
                              const std::vector<ExpressionEvaluator::Variable> &staticVariables);
    static ShaderPackLoader::VariableConfig buildExecutionVariableConfigForAssignment(
        const std::string &variableName,
        const ShaderPackLoader::ExecutionCommand::Assignment &assignment,
        const std::unordered_map<std::string, ShaderPackLoader::VariableConfig> &variableConfigs);
    void initStageRuntime(StageRuntime &stageRuntime, const ShaderPackLoader::ExecutionConfig &execution);
    ActiveRuntimeResources collectActiveRuntimeResources() const;
    void initRuntimeTextures();
    void initRuntimeBuffers();
    void loadRuntimeResources();

  private:
    std::weak_ptr<Framework> framework_;

    ShaderPackLoader::ShaderPack shaderPack_;
    std::unordered_map<std::string, std::string> shaderAttributes_;
    std::vector<ExpressionEvaluator::Variable> injectedExpressionVariables_;
    StageRuntime rayTracingStageRuntime_;
    StageRuntime postRenderStageRuntime_;
    std::unordered_map<std::string, std::string> staticAttributeValues_;

    std::vector<RuntimeTexture> runtimeTextures_;
    std::unordered_map<std::string, size_t> runtimeTextureIndices_;
    std::vector<RuntimeBuffer> runtimeBuffers_;
    std::unordered_map<std::string, size_t> runtimeBufferIndices_;
    ActiveRuntimeResources activeRuntimeResources_;
    std::vector<ExpressionEvaluator::Variable> runtimeResourceExpressionVariables_;
    bool runtimeResourceExpressionsDirty_ = true;
    mutable std::mutex numericExpressionCacheMutex_;
    mutable std::unordered_map<std::string, ExpressionEvaluator::CompiledExpression> numericExpressionCache_;

    uint32_t referenceWidth_ = 0;
    uint32_t referenceHeight_ = 0;
    bool runtimeResourcesReady_ = false;
    bool hasSharcRuntime_ = false;
};
