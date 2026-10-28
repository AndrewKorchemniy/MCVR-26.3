#include "core/render/vista.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/chunks.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

std::span<const VistaSpan> vistaTerrainView(int count, const VistaSpan *spans) {
    if (count < 0 || count > 4194304 || (count != 0 && spans == nullptr)) {
        throw std::invalid_argument("Invalid Vista boundary terrain");
    }
    return {spans, static_cast<size_t>(count)};
}

std::shared_ptr<const VistaTerrain> copyVistaTerrain(std::span<const VistaSpan> spans) {
    for (const auto &span : spans) {
        if (!std::isfinite(span.minY) || !std::isfinite(span.maxY) ||
            !std::isfinite(span.minX) || !std::isfinite(span.maxX) ||
            !std::isfinite(span.minZ) || !std::isfinite(span.maxZ) ||
            span.minY >= span.maxY || span.minX >= span.maxX || span.minZ >= span.maxZ) {
            throw std::invalid_argument("Invalid Vista boundary prism");
        }
    }
    return std::make_shared<const VistaTerrain>(spans.begin(), spans.end());
}

std::array<VistaGeometry, 2> VistaGeometry::build(std::span<const vk::VertexFormat::ChunkPBRVertex> vertices) {
    std::array<VistaGeometry, 2> geometries;
    for (uint32_t base = 0; base < vertices.size(); base += 4) {
        bool opaque = true;
        for (uint32_t i = 0; i < 4; ++i) {
            const auto &vertex = vertices[base + i];
            opaque &= (vertex.packedData & 0xffffu & ~32u) == 17u &&
                      vertex.textureID != 0xfffffffeu &&
                      (vertex.textureID & 0xffff0000u) != 0x800c0000u;
        }
        auto &geometry = geometries[opaque ? 0 : 1];
        for (uint32_t i = 0; i < 4; ++i) {
            const auto &vertex = vertices[base + i];
            const glm::dvec3 position(vertex.posX, vertex.posY, vertex.posZ);
            geometry.lower = glm::min(geometry.lower, position);
            geometry.upper = glm::max(geometry.upper, position);
        }
        geometry.indices.insert(geometry.indices.end(), {base, base + 1, base + 2, base + 2, base + 3, base});
    }
    return geometries;
}

bool VistaCoverage::contains(glm::ivec3 section) const {
    return glm::all(glm::greaterThanEqual(section, first)) &&
           glm::all(glm::lessThan(section, first + size));
}

size_t VistaCoverage::index(glm::ivec3 section) const {
    const int x = (section.x % size.x + size.x) % size.x;
    const int z = (section.z % size.z + size.z) % size.z;
    return (static_cast<size_t>(z) * size.y + section.y - first.y) * size.x + x;
}

void VistaCoverage::setReady(glm::ivec3 section) {
    if (!contains(section)) { return; }
    const size_t bit = index(section);
    words[bit / 32] |= 1u << (bit % 32);
}

bool VistaCoverage::isReady(glm::ivec3 section) const {
    if (!contains(section)) { return false; }
    const size_t bit = index(section);
    return bit / 32 < words.size() && (words[bit / 32] & (1u << (bit % 32))) != 0;
}

void VistaCoverage::retainCompleteColumns() {
    for (int z = first.z; z < first.z + size.z; ++z) {
        for (int x = first.x; x < first.x + size.x; ++x) {
            bool ready = true;
            for (int y = first.y; y < first.y + size.y; ++y) { ready &= isReady({x, y, z}); }
            if (ready) { continue; }
            for (int y = first.y; y < first.y + size.y; ++y) {
                const size_t bit = index({x, y, z});
                words[bit / 32] &= ~(1u << (bit % 32));
            }
        }
    }
}

bool VistaCoverage::isColumnReady(glm::ivec2 column) const { return isReady({column.x, first.y, column.y}); }

void VistaCoverage::setBorder(glm::ivec2 column, int direction) {
    if (!contains({column.x, first.y, column.y})) { return; }
    const size_t slot = static_cast<size_t>((column.y % size.z + size.z) % size.z) * size.x +
                        (column.x % size.x + size.x) % size.x;
    const size_t bit = slot * 4 + direction;
    borders[bit / 32] |= 1u << (bit % 32);
}

void VistaCoverage::addFallback(glm::ivec2 lower, glm::ivec2 upper) {
    const glm::ivec2 begin = glm::max(lower, glm::ivec2(first.x, first.z));
    const glm::ivec2 end = glm::min(upper, glm::ivec2(first.x + size.x, first.z + size.z));
    for (int z = begin.y; z < end.y; ++z) {
        for (int x = begin.x; x < end.x; ++x) {
            const size_t bit = static_cast<size_t>((z % size.z + size.z) % size.z) * size.x +
                               (x % size.x + size.x) % size.x;
            fallback[bit / 32] |= 1u << (bit % 32);
        }
    }
}

bool VistaCoverage::usesFallback(glm::ivec2 column) const {
    if (!contains({column.x, first.y, column.y}) || isColumnReady(column)) { return false; }
    const size_t bit = static_cast<size_t>((column.y % size.z + size.z) % size.z) * size.x +
                       (column.x % size.x + size.x) % size.x;
    return bit / 32 < fallback.size() && (fallback[bit / 32] & (1u << (bit % 32))) != 0;
}

std::vector<uint32_t> VistaCoverage::packedColumnFlags() const {
    if (glm::any(glm::lessThanEqual(size, glm::ivec3(0)))) return {};
    std::vector<uint32_t> packed((static_cast<size_t>(size.x) * size.z + 3) / 4, 0);
    for (int z = 0; z < size.z; ++z) for (int x = 0; x < size.x; ++x) {
        const glm::ivec2 column(first.x + x, first.z + z);
        const size_t slot = index({column.x, first.y, column.y});
        const size_t columnSlot = (slot / (size_t(size.x) * size.y)) * size.x + slot % size.x;
        const uint32_t ready = slot / 32 < words.size() ? (words[slot / 32] >> (slot % 32)) & 1u : 0;
        const uint32_t distant = columnSlot / 32 < fallback.size() ?
            (fallback[columnSlot / 32] >> (columnSlot % 32)) & 1u : 0;
        const size_t borderBit = columnSlot * 4;
        const uint32_t border = borderBit / 32 < borders.size() ?
            (borders[borderBit / 32] >> (borderBit % 32)) & 15u : 0;
        const size_t relative = size_t(z) * size.x + x;
        packed[relative / 4] |= (ready | (distant << 1) | (border << 2)) << ((relative % 4) * 8);
    }
    return packed;
}

VistaCoverage::State VistaCoverage::classify(glm::ivec3 lower, glm::ivec3 upper) const {
    if (words.empty() || glm::any(glm::lessThanEqual(size, glm::ivec3(0)))) { return State::Uncovered; }
    const glm::ivec3 begin = glm::max(lower, first);
    const glm::ivec3 end = glm::min(upper, first + size - 1);
    if (glm::any(glm::greaterThan(begin, end))) { return State::Uncovered; }
    bool uncovered = begin != lower || end != upper;
    bool covered = false;
    for (int z = begin.z; z <= end.z; ++z) {
        for (int y = begin.y; y <= end.y; ++y) {
            for (int x = begin.x; x <= end.x; ++x) {
                const bool ready = isReady(glm::ivec3(x, y, z));
                covered |= ready;
                uncovered |= !ready;
                if (covered && uncovered) { return State::Partial; }
            }
        }
    }
    return covered ? State::Covered : State::Uncovered;
}

VistaBorderGeometry VistaBorderGeometry::build(bool axisX, int32_t plane, int32_t tangentStart, int32_t tangentEnd,
                                               Owner negativeOwner, std::span<const VistaSpan> negative,
                                               Owner positiveOwner, std::span<const VistaSpan> positive,
                                               glm::ivec3 origin) {
    struct Face {
        double start, end;
        const VistaSpan *span;
    };
    VistaBorderGeometry result;
    if (negativeOwner == positiveOwner || negativeOwner == Owner::Unavailable ||
        positiveOwner == Owner::Unavailable || plane % 16 != 0 || tangentStart >= tangentEnd) { return result; }

    std::array<std::vector<Face>, 2> faces;
    std::vector<double> tangents;
    auto collect = [&](std::span<const VistaSpan> spans, size_t side) {
        for (const auto &span : spans) {
            if (!valid(span)) { continue; }
            const double lower = axisX ? double(span.x) + span.minX : double(span.z) + span.minZ;
            const double upper = axisX ? double(span.x) + span.maxX : double(span.z) + span.maxZ;
            if (side == 0 ? !(lower < plane && upper >= plane) : !(lower <= plane && upper > plane)) {
                continue;
            }
            const double start = std::max(double(tangentStart),
                axisX ? double(span.z) + span.minZ : double(span.x) + span.minX);
            const double end = std::min(double(tangentEnd),
                axisX ? double(span.z) + span.maxZ : double(span.x) + span.maxX);
            if (start >= end) { continue; }
            faces[side].push_back({start, end, &span});
            tangents.push_back(start);
            tangents.push_back(end);
        }
    };
    collect(negative, 0);
    collect(positive, 1);
    sortUnique(tangents);

    for (size_t column = 1; column < tangents.size(); ++column) {
        const double start = tangents[column - 1], end = tangents[column];
        std::array<std::vector<const VistaSpan *>, 2> active;
        std::vector<double> heights;
        for (size_t side = 0; side < faces.size(); ++side) {
            for (const auto &face : faces[side]) {
                if (face.start > start || face.end < end) { continue; }
                active[side].push_back(face.span);
                heights.push_back(face.span->minY);
                heights.push_back(face.span->maxY);
            }
        }
        sortUnique(heights);
        const VistaSpan *previous = nullptr;
        bool previousNegative = false;
        double lower = 0, upper = 0;
        auto flush = [&]() {
            if (previous != nullptr) {
                result.append(axisX, plane, start, end, lower, upper, previousNegative, *previous, origin);
            }
        };
        for (size_t row = 1; row < heights.size(); ++row) {
            const double bottom = heights[row - 1], top = heights[row];
            const VistaSpan *a = materialAt(active[0], bottom, top);
            const VistaSpan *b = materialAt(active[1], bottom, top);
            const VistaSpan *surface = nullptr;
            bool fromNegative = false;
            if (a != nullptr && (b == nullptr || (opaque(*a) && !opaque(*b)))) {
                surface = a;
                fromNegative = true;
            } else if (b != nullptr && (a == nullptr || (opaque(*b) && !opaque(*a)))) {
                surface = b;
            }
            if (surface != nullptr && previous != nullptr && previousNegative == fromNegative &&
                previous->material == surface->material && previous->colorRGBA8 == surface->colorRGBA8 &&
                previous->sideMaterials == surface->sideMaterials &&
                previous->sideShading == surface->sideShading &&
                upper == bottom) {
                upper = top;
                continue;
            }
            flush();
            previous = surface;
            previousNegative = fromNegative;
            lower = bottom;
            upper = top;
        }
        flush();
    }
    return result;
}

bool VistaBorderGeometry::valid(const VistaSpan &span) {
    return std::isfinite(span.minY) && std::isfinite(span.maxY) &&
           std::isfinite(span.minX) && std::isfinite(span.maxX) &&
           std::isfinite(span.minZ) && std::isfinite(span.maxZ) &&
           span.minY < span.maxY && span.minX < span.maxX && span.minZ < span.maxZ &&
           (span.material == waterMaterial || (span.colorRGBA8 >> 24) != 0);
}

bool VistaBorderGeometry::opaque(const VistaSpan &span) {
    return span.material != waterMaterial && (span.colorRGBA8 >> 24) == 255;
}

void VistaBorderGeometry::sortUnique(std::vector<double> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

const VistaSpan *VistaBorderGeometry::materialAt(std::span<const VistaSpan *const> spans, double lower, double upper) {
    const VistaSpan *result = nullptr;
    for (const VistaSpan *span : spans) {
        if (span->minY > lower || span->maxY < upper) { continue; }
        if (result == nullptr || (opaque(*span) && !opaque(*result)) ||
            (opaque(*span) == opaque(*result) &&
                (span->material < result->material ||
                    (span->material == result->material && span->colorRGBA8 < result->colorRGBA8)))) {
            result = span;
        }
    }
    return result;
}

void VistaBorderGeometry::append(bool axisX, int32_t plane, double start, double end, double bottom, double top,
                                  bool fromNegative, const VistaSpan &span, glm::ivec3 origin) {
    const double normalOrigin = axisX ? origin.x : origin.z;
    const double tangentOrigin = axisX ? origin.z : origin.x;
    const float p = float(double(plane) - normalOrigin);
    const float a = float(start - tangentOrigin), b = float(end - tangentOrigin);
    const float low = float(bottom - double(origin.y)), high = float(top - double(origin.y));
    if (a >= b || low >= high) { return; }
    std::array<glm::vec3, 4> positions = axisX ?
        std::array<glm::vec3, 4>{{{p, low, a}, {p, high, a}, {p, high, b}, {p, low, b}}} :
        std::array<glm::vec3, 4>{{{a, low, p}, {b, low, p}, {b, high, p}, {a, high, p}}};
    if (!fromNegative) { std::swap(positions[1], positions[3]); }
    const uint32_t normal = axisX ? (fromNegative ? 0x00007fffu : 0x00008001u) :
                                   (fromNegative ? 0x00000000u : 0x7fff7fffu);
    const uint32_t face = axisX ? (fromNegative ? 1u : 0u) : (fromNegative ? 3u : 2u);
    const uint32_t material = span.sideMaterials[face] != 0u ? span.sideMaterials[face] : span.material;
    const uint32_t shade = (span.sideShading >> (face * 8u)) & 255u;
    uint32_t color = span.colorRGBA8 & 0xff000000u;
    for (uint32_t shift = 0; shift < 24; shift += 8) {
        color |= (((span.colorRGBA8 >> shift) & 255u) * shade / 255u) << shift;
    }
    const uint32_t type = (material >> 16u) & 15u;
    const uint32_t emission = (material & 0xfff00000u) == 0x80000000u && (type == 6u || type == 15u)
        ? 0x3c000000u : 0u;
    const uint32_t packed = 1u | 16u | (opaque(span) ? 0u : 2u << 8u) | emission;
    for (const auto &position : positions) {
        const glm::vec2 uv = axisX ? glm::vec2(fromNegative ? -position.z : position.z, -position.y) :
                                    glm::vec2(fromNegative ? position.x : -position.x, -position.y);
        vertices.push_back({position.x, position.y, position.z, normal, uv,
                           color, 0u, material, packed});
    }
}

static std::vector<std::shared_ptr<VistaRenderData>> buildVistaGeometry(
    const std::shared_ptr<Framework> &framework, const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
    VistaBuildTask &task) {
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    auto pbrBuffer = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false,
                                                   task.vertices.size() * sizeof(task.vertices[0]), usage);
    pbrBuffer->uploadToStagingBuffer(task.vertices.data());
    pbrBuffer->flushStagingBuffer();
    pbrBuffer->uploadToBuffer(commandBuffer);
    auto geometries = VistaGeometry::build(task.vertices);
    std::vector<std::shared_ptr<VistaRenderData>> parts;
    for (size_t part = 0; part < geometries.size(); ++part) {
        auto &geometry = geometries[part];
        if (geometry.indices.empty()) { continue; }
        auto data = VistaRenderData::create();
        data->id = task.id;
        data->x = task.x;
        data->y = task.y;
        data->z = task.z;
        data->width = task.width;
        data->terrain = task.terrain;
        data->isOpaque = part == 0;
        const glm::dvec3 lower = geometry.lower + glm::dvec3(task.x, task.y, task.z);
        const glm::dvec3 upper = geometry.upper + glm::dvec3(task.x, task.y, task.z);
        const glm::dvec3 margin = glm::dvec3(0.001) + glm::max(glm::abs(lower), glm::abs(upper)) *
                                                           (4.0 * std::numeric_limits<float>::epsilon());
        data->firstSection = glm::ivec3(glm::floor((lower - margin) / 16.0));
        data->lastSection = glm::ivec3(glm::floor((upper + margin) / 16.0));
        const uint32_t vertexCount = static_cast<uint32_t>(task.vertices.size());
        const uint32_t indexCount = static_cast<uint32_t>(geometry.indices.size());
        data->pbrBuffer = pbrBuffer;
        data->indexBuffer = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), false,
                                                          geometry.indices.size() * sizeof(uint32_t), usage);
        data->indexBuffer->uploadToStagingBuffer(geometry.indices.data());
        data->indexBuffer->flushStagingBuffer();
        data->indexBuffer->uploadToBuffer(commandBuffer);
        commandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT,
        }});
        auto builder = vk::BLASBuilder::create();
        builder->beginGeometries()->defineTriangleGeomrtry<vk::VertexFormat::ChunkPBRVertex>(
            data->pbrBuffer->bufferAddress(), vertexCount, data->indexBuffer->bufferAddress(), indexCount, false);
        builder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
            ->querySizeInfo(framework->device())
            ->allocateBuffers(framework->physicalDevice(), framework->device(), framework->vma());
        data->blas = builder->buildAndSubmit(framework->device(), commandBuffer);
        commandBuffer->retainResource(builder);
        commandBuffer->retainResource(data);
        parts.push_back(data);
    }
    commandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});
    return parts;
}

Vista::Vista(std::shared_ptr<Framework> framework) : framework_(framework) {}

void Vista::reset(int64_t worldVersion) {
    std::lock_guard lock(mutex_);
    worldVersion_ = worldVersion;
    frame_ = 0;
    pending_.clear();
    building_.clear();
    pendingBytes_ = 0;
    resident_.clear();
    selected_.clear();
    objects_.clear();
    objectOffsets_.clear();
    snapshots_.clear();
    snapshotsDirty_ = true;
    bordersDirty_ = true;
    coverageChunks_.clear();
    coverageGridInfo_ = {};
    coverageStoragePos_ = {};
    coverageRenderVersion_ = UINT64_MAX;
    coverage_ = {};
    coverageVersion_ = 1;
    coverageBuffer_ = nullptr;
    coverageWords_.clear();
    terrain_.clear();
    nearTerrain_.clear();
    borders_.clear();
}

std::shared_ptr<const VistaTerrain> Vista::terrain(
    int64_t worldVersion, int64_t id, std::span<const VistaSpan> spans) {
    std::lock_guard lock(mutex_);
    if (worldVersion != worldVersion_) { return nullptr; }
    if (auto found = terrain_.find(id); found != terrain_.end()) {
        if (auto terrain = found->second.lock()) { return terrain; }
    }
    auto terrain = copyVistaTerrain(spans);
    terrain_[id] = terrain;
    return terrain;
}

bool Vista::queueVistaBuild(VistaBuildTask task) {
    std::lock_guard lock(mutex_);
    const size_t bytes = task.vertices.size() * sizeof(vk::VertexFormat::ChunkPBRVertex);
    if (task.worldVersion != worldVersion_) { return false; }
    if (resident_.contains(task.id) || building_.contains(task.id)) { return true; }
    for (const auto &pending : pending_) {
        if (pending.id == task.id) { return true; }
    }
    // A Voxy section can carry 262,144 quads (about 40 MiB of vertices), and thousands of sections are queued.
    if (pendingBytes_ + bytes > 256 * 1024 * 1024) { return false; }
    pendingBytes_ += bytes;
    pending_.push_back(std::move(task));
    return true;
}

void Vista::invalidateVista(int64_t worldVersion, int64_t id) {
    std::lock_guard lock(mutex_);
    if (worldVersion != worldVersion_) { return; }
    resident_.erase(id);
    snapshotsDirty_ = true;
    objectOffsets_.erase(id);
    std::erase(objects_, id);
    building_.erase(id);
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->id == id) {
            pendingBytes_ -= it->vertices.size() * sizeof(vk::VertexFormat::ChunkPBRVertex);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

void Vista::updateRenderList(int64_t worldVersion, std::vector<int64_t> ids) {
    std::lock_guard lock(mutex_);
    if (worldVersion == worldVersion_ && selected_ != ids) {
        selected_ = std::move(ids);
        snapshotsDirty_ = true;
    }
}

void Vista::updateObjectList(int64_t worldVersion, std::vector<int64_t> ids) {
    std::lock_guard lock(mutex_);
    if (worldVersion == worldVersion_) { objects_ = std::move(ids); }
}

void Vista::moveObject(int64_t worldVersion, int64_t id, glm::dvec3 offset) {
    std::lock_guard lock(mutex_);
    if (worldVersion == worldVersion_) { objectOffsets_[id] = offset; }
}

bool Vista::isReady(int64_t worldVersion, int64_t id) {
    std::lock_guard lock(mutex_);
    return worldVersion == worldVersion_ && resident_.contains(id);
}

std::vector<VistaRenderInstance> Vista::prepare(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    MCVR_PROFILE_SCOPE("native.Vista.prepare");
    auto framework = framework_.lock();
    if (framework == nullptr) { return {}; }
    std::deque<VistaBuildTask> tasks;
    int64_t worldVersion;
    {
        std::lock_guard lock(mutex_);
        worldVersion = worldVersion_;
        size_t bytes = 0;
        while (!pending_.empty() && bytes < 8 * 1024 * 1024) {
            const size_t size = pending_.front().vertices.size() * sizeof(vk::VertexFormat::ChunkPBRVertex);
            bytes += size;
            pendingBytes_ -= size;
            building_.insert(pending_.front().id);
            tasks.push_back(std::move(pending_.front()));
            pending_.pop_front();
        }
    }
    for (auto &task : tasks) {
        auto parts = buildVistaGeometry(framework, commandBuffer, task);
        std::lock_guard lock(mutex_);
        if (worldVersion == worldVersion_ && building_.erase(task.id) != 0) {
            resident_[task.id] = std::move(parts);
            snapshotsDirty_ = true;
        }
    }
    if (!tasks.empty()) {
        commandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        }});
    }
    std::lock_guard lock(mutex_);
    ++frame_;
    const bool ready = snapshotsDirty_ &&
        std::all_of(selected_.begin(), selected_.end(), [&](int64_t id) { return resident_.contains(id); });
    if (ready) {
        snapshots_.clear();
        for (int64_t id : selected_) {
            const auto &parts = resident_.at(id);
            snapshots_.insert(snapshots_.end(), parts.begin(), parts.end());
        }
        snapshotsDirty_ = false;
        bordersDirty_ = true;
    }
    MCVR_PROFILE_METRIC("vista.resident_meshes", resident_.size());
    std::vector<VistaRenderInstance> instances;
    instances.reserve(snapshots_.size());
    coverage_.fallback.assign((static_cast<size_t>(coverage_.size.x) * coverage_.size.z + 31) / 32, 0);
    size_t opaqueCount = 0, coveredCount = 0;
    for (const auto &data : snapshots_) {
        coverage_.addFallback({data->x / 16, data->z / 16},
                              {(data->x + data->width) / 16, (data->z + data->width) / 16});
        if (data->coverageVersion != coverageVersion_) {
            data->coverageState = coverage_.classify({data->firstSection.x, coverage_.first.y, data->firstSection.z},
                                                   {data->lastSection.x, coverage_.first.y, data->lastSection.z});
            data->coverageVersion = coverageVersion_;
        }
        const auto state = data->coverageState;
        if (state == VistaCoverage::State::Covered) {
            ++coveredCount;
            continue;
        }
        const bool isOpaque = data->isOpaque && state == VistaCoverage::State::Uncovered;
        opaqueCount += isOpaque;
        instances.push_back({data, isOpaque});
        commandBuffer->retainResource(data);
    }
    prepareBorders(commandBuffer, instances);
    for (int64_t id : objects_) {
        auto found = resident_.find(id);
        if (found == resident_.end()) { continue; }
        for (const auto &data : found->second) {
            data->offset = objectOffsets_[id];
            instances.push_back({data, data->isOpaque, true, data->lastFrame != 0 && data->lastFrame + 1 == frame_});
            data->lastFrame = frame_;
            commandBuffer->retainResource(data);
        }
    }
    MCVR_PROFILE_METRIC("vista.rendered_meshes", instances.size());
    MCVR_PROFILE_METRIC("vista.opaque_meshes", opaqueCount);
    MCVR_PROFILE_METRIC("vista.covered_meshes", coveredCount);
    return instances;
}

void Vista::close() {
    reset(0);
}

void Vista::prepareBorders(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                           std::vector<VistaRenderInstance> &instances) {
    MCVR_PROFILE_SCOPE("native.Vista.prepareBorders");
    if (!bordersDirty_) {
        for (const auto &[key, border] : borders_) {
            for (const auto &data : border.geometry) {
                instances.push_back({data, data->isOpaque, true});
                commandBuffer->retainResource(data);
            }
        }
        return;
    }
    coverage_.borders.assign((static_cast<size_t>(coverage_.size.x) * coverage_.size.z * 4 + 31) / 32, 0);
    std::map<std::pair<int, int>, std::shared_ptr<const VistaTerrain>> vistaTerrain;
    for (const auto &data : snapshots_) {
        if (data->terrain == nullptr) { continue; }
        const int minX = std::max(data->x / 16, coverage_.first.x - 1);
        const int minZ = std::max(data->z / 16, coverage_.first.z - 1);
        const int maxX = std::min((data->x + data->width) / 16, coverage_.first.x + coverage_.size.x + 1);
        const int maxZ = std::min((data->z + data->width) / 16, coverage_.first.z + coverage_.size.z + 1);
        for (int x = minX; x < maxX; ++x) {
            for (int z = minZ; z < maxZ; ++z) { vistaTerrain[{x, z}] = data->terrain; }
        }
    }
    std::map<std::tuple<bool, int, int>, Border> borders;
    constexpr glm::ivec2 directions[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto &[column, nearTerrain] : nearTerrain_) {
        if (nearTerrain.size() != static_cast<size_t>(coverage_.size.y)) { continue; }
        for (int direction = 0; direction < 4; ++direction) {
            const glm::ivec2 neighbour = glm::ivec2(column.first, column.second) + directions[direction];
            if (coverage_.isColumnReady(neighbour)) { continue; }
            const auto vista = vistaTerrain.find({neighbour.x, neighbour.y});
            if (vista == vistaTerrain.end()) { continue; }
            const bool axisX = direction < 2;
            const bool nearNegative = direction == 1 || direction == 3;
            const int plane = ((axisX ? column.first : column.second) + (nearNegative ? 1 : 0)) * 16;
            const int tangent = (axisX ? column.second : column.first) * 16;
            const auto key = std::make_tuple(axisX, plane, tangent);
            const ColumnTerrain negative = nearNegative ? nearTerrain : ColumnTerrain{vista->second};
            const ColumnTerrain positive = nearNegative ? ColumnTerrain{vista->second} : nearTerrain;
            auto cached = borders_.find(key);
            Border border;
            if (cached != borders_.end() && cached->second.nearNegative == nearNegative &&
                cached->second.negative == negative && cached->second.positive == positive) {
                border = std::move(cached->second);
            } else {
                border.negative = negative;
                border.positive = positive;
                border.nearNegative = nearNegative;
                auto collect = [&](const ColumnTerrain &terrainData, bool negativeSide) {
                    std::vector<VistaSpan> spans;
                    for (const auto &terrain : terrainData) {
                        for (const auto &span : *terrain) {
                            const double lower = axisX ? double(span.x) + span.minX : double(span.z) + span.minZ;
                            const double upper = axisX ? double(span.x) + span.maxX : double(span.z) + span.maxZ;
                            const double first = axisX ? double(span.z) + span.minZ : double(span.x) + span.minX;
                            const double last = axisX ? double(span.z) + span.maxZ : double(span.x) + span.maxX;
                            const bool touches = negativeSide ? lower < plane && upper >= plane :
                                                               lower <= plane && upper > plane;
                            if (touches && first < tangent + 16 && last > tangent) { spans.push_back(span); }
                        }
                    }
                    return spans;
                };
                const auto negativeSpans = collect(negative, true);
                const auto positiveSpans = collect(positive, false);
                const glm::ivec3 origin = axisX ? glm::ivec3(plane, 0, tangent) : glm::ivec3(tangent, 0, plane);
                auto geometry = VistaBorderGeometry::build(axisX, plane, tangent, tangent + 16,
                    nearNegative ? VistaBorderGeometry::Owner::Near : VistaBorderGeometry::Owner::Vista,
                    negativeSpans,
                    nearNegative ? VistaBorderGeometry::Owner::Vista : VistaBorderGeometry::Owner::Near,
                    positiveSpans, origin);
                if (!geometry.vertices.empty()) {
                    VistaBuildTask task{
                        .worldVersion = worldVersion_, .id = 0, .x = origin.x, .y = origin.y, .z = origin.z,
                        .width = 16, .vertices = std::move(geometry.vertices), .terrain = nullptr};
                    border.geometry = buildVistaGeometry(framework_.lock(), commandBuffer, task);
                }
            }
            coverage_.setBorder({column.first, column.second}, direction);
            for (const auto &data : border.geometry) {
                instances.push_back({data, data->isOpaque, true});
                commandBuffer->retainResource(data);
            }
            borders.emplace(key, std::move(border));
        }
    }
    borders_ = std::move(borders);
    bordersDirty_ = false;
    std::erase_if(terrain_, [](const auto &entry) { return entry.second.expired(); });
}

void Vista::updateCoverage(const std::vector<std::shared_ptr<Chunk1>> &chunks,
                           const std::vector<ChunkRenderRecord> &records,
                           glm::ivec4 gridInfo,
                           glm::ivec4 storageSectionPos, uint64_t renderVersion) {
    MCVR_PROFILE_SCOPE("native.Vista.updateCoverage");
    std::lock_guard lock(mutex_);
    bool changed = coverageGridInfo_ != gridInfo || coverageStoragePos_ != storageSectionPos ||
                   coverageRenderVersion_ != renderVersion || coverageChunks_.size() != chunks.size();
    coverageChunks_.resize(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto &chunk = chunks[i];
        auto &cached = coverageChunks_[i];
        const auto *terrain = chunk ? chunk->terrain.get() : nullptr;
        const glm::ivec3 position = chunk ? glm::ivec3(chunk->x, chunk->y, chunk->z) : glm::ivec3(0);
        const bool empty = chunk && chunk->terrainEmpty;
        if (cached.terrain.get() != terrain || cached.position != position || cached.empty != empty) {
            cached = {chunk ? chunk->terrain : nullptr, position, empty};
            changed = true;
        }
    }
    MCVR_PROFILE_METRIC("vista.coverage_rebuilt", changed);
    if (!changed) return;
    coverageGridInfo_ = gridInfo;
    coverageStoragePos_ = storageSectionPos;
    coverageRenderVersion_ = renderVersion;
    bordersDirty_ = true;
    VistaCoverage coverage;
    coverage.size = glm::ivec3(gridInfo);
    coverage.first = {storageSectionPos.x - gridInfo.x / 2, gridInfo.w,
                      storageSectionPos.z - gridInfo.z / 2};
    if (glm::any(glm::lessThanEqual(coverage.size, glm::ivec3(0)))) {
        coverage_ = {};
        nearTerrain_.clear();
        ++coverageVersion_;
        return;
    }
    coverage.words.resize((static_cast<size_t>(gridInfo.x) * gridInfo.y * gridInfo.z + 31) / 32, 0);
    VistaCoverage withTerrain = coverage;
    for (const auto &chunk : chunks) {
        if (chunk == nullptr || chunk->terrain == nullptr) { continue; }
        withTerrain.setReady({chunk->x / 16, chunk->y / 16, chunk->z / 16});
        if (chunk->terrainEmpty) {
            coverage.setReady({chunk->x / 16, chunk->y / 16, chunk->z / 16});
        }
    }
    for (const auto &record : records) {
        if (record.positionAndGeometryCount.w > 0 && (record.blasAddress.x | record.blasAddress.y) != 0 &&
            (record.offsetsMaskFlags.z & 1u) != 0 &&
            withTerrain.isReady(glm::ivec3(record.positionAndGeometryCount) / 16)) {
            coverage.setReady(glm::ivec3(record.positionAndGeometryCount) / 16);
        }
    }
    coverage.retainCompleteColumns();
    if (coverage.size != coverage_.size || coverage.first != coverage_.first || coverage.words != coverage_.words) {
        ++coverageVersion_;
    }
    coverage_ = std::move(coverage);
    nearTerrain_.clear();
    for (const auto &chunk : chunks) {
        if (chunk != nullptr && chunk->terrain != nullptr &&
            coverage_.isColumnReady({chunk->x / 16, chunk->z / 16})) {
            nearTerrain_[{chunk->x / 16, chunk->z / 16}].push_back(chunk->terrain);
        }
    }
}

VkDeviceAddress Vista::prepareCoverage(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    std::lock_guard lock(mutex_);
    auto framework = framework_.lock();
    if (framework == nullptr || coverage_.words.empty()) { return 0; }
    std::vector<uint32_t> words = coverage_.words;
    words.insert(words.end(), coverage_.fallback.begin(), coverage_.fallback.end());
    words.insert(words.end(), coverage_.borders.begin(), coverage_.borders.end());
    if (coverageBuffer_ == nullptr || coverageUploadFirst_ != coverage_.first || coverageUploadSize_ != coverage_.size ||
        coverageWords_.size() < words.size() || !std::equal(words.begin(), words.end(), coverageWords_.begin())) {
        const auto columns = coverage_.packedColumnFlags();
        words.insert(words.end(), columns.begin(), columns.end());
        coverageUploadFirst_ = coverage_.first;
        coverageUploadSize_ = coverage_.size;
        coverageWords_ = std::move(words);
        coverageBuffer_ = vk::DeviceLocalBuffer::create(
            framework->vma(), framework->device(), false, coverageWords_.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        coverageBuffer_->uploadToStagingBuffer(coverageWords_.data());
        coverageBuffer_->flushStagingBuffer();
        coverageBuffer_->uploadToBuffer(commandBuffer);
        commandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        }});
    }
    commandBuffer->retainResource(coverageBuffer_);
    return coverageBuffer_->bufferAddress();
}
