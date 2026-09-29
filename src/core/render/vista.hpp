#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Framework;
struct Chunk1;
struct ChunkRenderRecord;

namespace vk {
class BLAS;
class CommandBuffer;
class DeviceLocalBuffer;
}

struct VistaSpan {
    int32_t x, z;
    float minY, maxY;
    float minX, maxX, minZ, maxZ;
    uint32_t colorRGBA8, material;
    std::array<uint32_t, 4> sideMaterials{};
    uint32_t sideShading = 0xffffffffu;
};

static_assert(sizeof(VistaSpan) == 60);
static_assert(offsetof(VistaSpan, minY) == 8);
static_assert(offsetof(VistaSpan, minX) == 16);
static_assert(offsetof(VistaSpan, colorRGBA8) == 32);
static_assert(offsetof(VistaSpan, sideShading) == 56);

using VistaTerrain = std::vector<VistaSpan>;

std::span<const VistaSpan> vistaTerrainView(int count, const VistaSpan *spans);
std::shared_ptr<const VistaTerrain> copyVistaTerrain(std::span<const VistaSpan> spans);

struct VistaGeometry {
    std::vector<uint32_t> indices;
    glm::dvec3 lower = glm::dvec3(std::numeric_limits<double>::max());
    glm::dvec3 upper = glm::dvec3(std::numeric_limits<double>::lowest());

    static std::array<VistaGeometry, 2> build(std::span<const vk::VertexFormat::ChunkPBRVertex> vertices);
};

struct VistaCoverage {
    enum class State { Uncovered, Partial, Covered };

    glm::ivec3 first = glm::ivec3(0);
    glm::ivec3 size = glm::ivec3(0);
    std::vector<uint32_t> words;
    std::vector<uint32_t> fallback;
    std::vector<uint32_t> borders;

    bool contains(glm::ivec3 section) const;
    size_t index(glm::ivec3 section) const;
    void setReady(glm::ivec3 section);
    bool isReady(glm::ivec3 section) const;
    void retainCompleteColumns();
    bool isColumnReady(glm::ivec2 column) const;
    void setBorder(glm::ivec2 column, int direction);
    void addFallback(glm::ivec2 lower, glm::ivec2 upper);
    bool usesFallback(glm::ivec2 column) const;
    std::vector<uint32_t> packedColumnFlags() const;
    State classify(glm::ivec3 lower, glm::ivec3 upper) const;
};

struct VistaBorderGeometry {
    enum class Owner { Unavailable, Near, Vista };
    static constexpr uint32_t waterMaterial = 0xfffffffeu;

    std::vector<vk::VertexFormat::ChunkPBRVertex> vertices;

    static VistaBorderGeometry build(bool axisX, int32_t plane, int32_t tangentStart, int32_t tangentEnd,
                                     Owner negativeOwner, std::span<const VistaSpan> negative,
                                     Owner positiveOwner, std::span<const VistaSpan> positive,
                                     glm::ivec3 origin = glm::ivec3(0));

  private:
    static bool valid(const VistaSpan &span);
    static bool opaque(const VistaSpan &span);
    static void sortUnique(std::vector<double> &values);
    static const VistaSpan *materialAt(std::span<const VistaSpan *const> spans, double lower, double upper);
    void append(bool axisX, int32_t plane, double start, double end, double bottom, double top,
                bool fromNegative, const VistaSpan &span, glm::ivec3 origin);
};

struct VistaBuildTask {
    int64_t worldVersion;
    int64_t id;
    int x, y, z;
    int width;
    std::vector<vk::VertexFormat::ChunkPBRVertex> vertices;
    std::shared_ptr<const VistaTerrain> terrain;
};

struct VistaRenderData : public SharedObject<VistaRenderData> {
    int64_t id;
    int x, y, z;
    int width;
    std::shared_ptr<const VistaTerrain> terrain;
    glm::dvec3 offset = glm::dvec3(0.0);
    glm::mat4 previousTransform = glm::mat4(1.0f);
    uint64_t lastFrame = 0;
    glm::ivec3 firstSection;
    glm::ivec3 lastSection;
    bool isOpaque = false;
    uint64_t coverageVersion = 0;
    VistaCoverage::State coverageState = VistaCoverage::State::Partial;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::BLAS> blas;
};

struct VistaRenderInstance {
    std::shared_ptr<VistaRenderData> data;
    bool isOpaque;
    bool isBorder = false;
    bool hasHistory = false;
};

class Vista : public SharedObject<Vista> {
  public:
    Vista(std::shared_ptr<Framework> framework);
    void reset(int64_t worldVersion);
    bool queueVistaBuild(VistaBuildTask task);
    std::shared_ptr<const VistaTerrain> terrain(int64_t worldVersion, int64_t id,
                                               std::span<const VistaSpan> spans);
    void invalidateVista(int64_t worldVersion, int64_t id);
    void updateRenderList(int64_t worldVersion, std::vector<int64_t> ids);
    void updateObjectList(int64_t worldVersion, std::vector<int64_t> ids);
    void moveObject(int64_t worldVersion, int64_t id, glm::dvec3 offset);
    bool isReady(int64_t worldVersion, int64_t id);
    std::vector<VistaRenderInstance> prepare(std::shared_ptr<vk::CommandBuffer> commandBuffer);
    void updateCoverage(const std::vector<std::shared_ptr<Chunk1>> &chunks,
                        const std::vector<ChunkRenderRecord> &records,
                        glm::ivec4 gridInfo, glm::ivec4 storageSectionPos, uint64_t renderVersion);
    VkDeviceAddress prepareCoverage(std::shared_ptr<vk::CommandBuffer> commandBuffer);
    void close();

  private:
    std::weak_ptr<Framework> framework_;
    std::mutex mutex_;
    int64_t worldVersion_ = 0;
    uint64_t frame_ = 0;
    size_t pendingBytes_ = 0;
    std::deque<VistaBuildTask> pending_;
    std::unordered_set<int64_t> building_;
    std::unordered_map<int64_t, std::vector<std::shared_ptr<VistaRenderData>>> resident_;
    std::vector<int64_t> selected_;
    std::vector<int64_t> objects_;
    std::unordered_map<int64_t, glm::dvec3> objectOffsets_;
    std::vector<std::shared_ptr<VistaRenderData>> snapshots_;
    bool snapshotsDirty_ = true;
    bool bordersDirty_ = true;
    VistaCoverage coverage_;
    uint64_t coverageVersion_ = 1;
    struct CoverageChunk {
        std::shared_ptr<const VistaTerrain> terrain;
        glm::ivec3 position{};
        bool empty = false;
    };
    std::vector<CoverageChunk> coverageChunks_;
    glm::ivec4 coverageGridInfo_{};
    glm::ivec4 coverageStoragePos_{};
    uint64_t coverageRenderVersion_ = UINT64_MAX;
    std::shared_ptr<vk::DeviceLocalBuffer> coverageBuffer_;
    std::vector<uint32_t> coverageWords_;
    glm::ivec3 coverageUploadFirst_{}, coverageUploadSize_{};
    std::unordered_map<int64_t, std::weak_ptr<const VistaTerrain>> terrain_;
    using ColumnTerrain = std::vector<std::shared_ptr<const VistaTerrain>>;
    std::map<std::pair<int, int>, ColumnTerrain> nearTerrain_;
    struct Border {
        ColumnTerrain negative;
        ColumnTerrain positive;
        bool nearNegative;
        std::vector<std::shared_ptr<VistaRenderData>> geometry;
    };
    std::map<std::tuple<bool, int, int>, Border> borders_;
    void prepareBorders(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                        std::vector<VistaRenderInstance> &instances);
};
