#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <string>

class Textures;
class Framework;
class Buffers;
class World;
struct VulkanHandles;

struct Options {
    uint32_t maxFps = 1e6;
    uint32_t inactivityFpsLimit = 1e6;
    bool vsync = true;
    uint32_t dlssMode = 1;
    uint32_t upscalerType = 1;
    uint32_t upscalerQuality = 0;
    uint32_t denoiserMode = 1;
    uint32_t frameGenerationMode = 0;
    uint32_t frameGenerationMultiplier = 2;
    uint32_t rayBounces = 4;
    uint32_t debugMode = 0;
    bool needRecreate = false;

    uint32_t chunkBuildingBatchSize = 8;
    uint32_t chunkBuildingTotalBatches = 4;
    uint32_t chunkBuildingThreads = 4;
    bool collectChunkEmission = true;
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;

    ~Renderer();

    std::shared_ptr<Framework> framework();
    std::shared_ptr<Textures> textures();
    std::shared_ptr<Buffers> buffers();
    std::shared_ptr<World> world();

    void close();

  private:
    Renderer(VulkanHandles handles);

    template <typename T, typename... Args>
    static std::shared_ptr<T> createWithMemoryTag(const std::string &tag, Args &&...args);

    std::shared_ptr<Framework> framework_;
    std::shared_ptr<Textures> textures_;
    std::shared_ptr<Buffers> buffers_;
    std::shared_ptr<World> world_;
    std::mutex closeMtx_;
    std::condition_variable closeCv_;
    bool closing_ = false;
    bool closed_ = false;
};
