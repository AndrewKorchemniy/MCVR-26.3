#include "core/render/buffers.hpp"

#include "common/shared.hpp"
#include "core/render/chunks.hpp"
#ifdef MCVR_UI_BACKEND_VULKAN
#    include "core/render/modules/ui_module.hpp"
#endif
#include "core/profiler/profiler.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

std::ostream &buffersCerr() {
    return std::cerr << "[Buffers] ";
}

Buffers::Buffers(std::shared_ptr<Framework> framework) {
    MCVR_PROFILE_SCOPE("native.Buffers.init");
    uint32_t size = framework->swapchain()->imageCount();
    auto device = framework->device();
    auto vma = framework->vma();
    auto alignTo = [](uint32_t value, uint32_t alignment) {
        if (alignment <= 1 || value == 0) return value;
        return ((value + alignment - 1) / alignment) * alignment;
    };
    overlayDrawUniformAlignment_ =
        std::max<uint32_t>(1, framework->physicalDevice()->properties().limits.minUniformBufferOffsetAlignment);
    overlayDrawUniformDeviceLimit_ =
        std::max<uint32_t>(1, framework->physicalDevice()->properties().limits.maxUniformBufferRange);
    overlayDrawUniformDescriptorRange_ =
        std::min<uint32_t>(overlayDrawUniformDeviceLimit_, overlayDrawUniformInitialDescriptorRange);
    overlayPostUniformDescriptorRange_ = sizeof(vk::Data::OverlayPostUBO);
    overlayPostUniformStride_ = alignTo(overlayPostUniformDescriptorRange_, overlayDrawUniformAlignment_);

    validOverlayIndex_.resize(size);
    overlayIndexVertexBuffer_.resize(size);

    overlayDrawUniformBuffer_.resize(size);
    overlayPostUniformBuffer_.resize(size);
    overlayDrawUniformData_.resize(size);
    overlayDrawUniformWriteOffset_.resize(size, 0);
    overlayPostUniformData_.resize(size);
    overlayPostUniformCount_.resize(size, 0);
    importantIndexVertexBuffer_ = std::make_shared<std::vector<ImportantWorldUpload>>();

    worldUniformBuffer_.resize(size);
    lastWorldUniformBuffer_.resize(size);
    skyUniformBuffer_.resize(size);
    textureMappingBuffer_.resize(size);
    exposureDataBuffer_.resize(size);

    for (uint32_t i = 0; i < size; i++) {
        overlayDrawUniformBuffer_[i] = vk::HostVisibleBuffer::create(
            vma, device, std::max<uint32_t>(overlayDrawUniformInitialSize, overlayDrawUniformAlignment_),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        overlayDrawUniformData_[i].reserve(overlayDrawUniformBuffer_[i]->size());
        overlayPostUniformBuffer_[i] = vk::HostVisibleBuffer::create(
            vma, device, std::max<uint32_t>(overlayPostUniformInitialSize, overlayPostUniformStride_),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        overlayPostUniformData_[i].reserve(overlayPostUniformBuffer_[i]->size());
    }
}

bool Buffers::ensureOverlayDrawUniformBufferCapacityLocked(std::shared_ptr<Framework> framework,
                                                           uint32_t frameIndex,
                                                           uint32_t requiredBufferSize) {
    auto &buffer = overlayDrawUniformBuffer_[frameIndex];
    if (requiredBufferSize <= buffer->size()) { return false; }

    uint32_t newSize = std::max<uint32_t>(buffer->size(), overlayDrawUniformInitialSize);
    while (newSize < requiredBufferSize) { newSize *= 2; }

    auto newBuffer = vk::HostVisibleBuffer::create(framework->vma(), framework->device(), newSize,
                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    framework->frameResourceRetainer().retain(buffer);
    overlayDrawUniformBuffer_[frameIndex] = newBuffer;
    overlayDrawUniformData_[frameIndex].reserve(newSize);
    return true;
}

void Buffers::resetFrame() {
    MCVR_PROFILE_SCOPE("native.Buffers.resetFrame");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto &frr = framework->frameResourceRetainer();

    hasWorldHistory_ = worldUniformUpdated_;
    worldUniformUpdated_ = false;

    validOverlayIndex_[context->frameIndex].clear();

    overlayNextID_ = 0;

    frr.retain(importantIndexVertexBuffer_);
    importantIndexVertexBuffer_ = std::make_shared<std::vector<ImportantWorldUpload>>();

    overlayDrawUniformData_[context->frameIndex].clear();
    overlayDrawUniformWriteOffset_[context->frameIndex] = 0;
    overlayPostUniformData_[context->frameIndex].clear();
    overlayPostUniformCount_[context->frameIndex] = 0;
}

std::pair<uint32_t, uint32_t> Buffers::createAndUploadVertexIndexBuffers(uint8_t *vertexData,
                                                                         uint32_t vertexSize,
                                                                         uint8_t *sortedIndexData,
                                                                         uint32_t indexSize,
                                                                         int indexType,
                                                                         int drawMode,
                                                                         int vertexCount,
                                                                         int expectedIndexCount) {
    MCVR_PROFILE_SCOPE("native.Buffers.createAndUploadVertexIndexBuffers");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto frameIndex = context->frameIndex;
    auto &validBuffers = validOverlayIndex_[frameIndex];
    auto &buffers = overlayIndexVertexBuffer_[frameIndex];

    auto allocateBuffer = [&]() {
        uint32_t id = overlayNextID_++;
        validBuffers.emplace(id, -1);
        buffers.try_emplace(id, nullptr);
        return id;
    };
    auto initializeBuffer = [&](uint32_t id, uint32_t size, VkBufferUsageFlags usageFlags) {
        auto bufferIter = buffers.find(id);
        if (!validBuffers.contains(id) || bufferIter == buffers.end()) {
            throw std::runtime_error("Overlay buffer allocation was lost before initialization");
        }

        validBuffers.at(id) = static_cast<int32_t>(size);
        auto &buffer = bufferIter->second;
        uint32_t capacity = buffer == nullptr ? baseBlockSize : buffer->size();
        while (capacity < size) { capacity *= 2; }
        if (buffer == nullptr || capacity != buffer->size()) {
            framework->frameResourceRetainer().retain(buffer);
            buffer = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), capacity, usageFlags);
        }
        return buffer;
    };

    const uint32_t vertexId = allocateBuffer();
    const auto vertexBuffer = initializeBuffer(vertexId, vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (vertexSize > 0) { vertexBuffer->uploadToStagingBuffer(vertexData, vertexSize, 0); }

    const uint32_t indexId = allocateBuffer();
    const auto indexBuffer = initializeBuffer(indexId, indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    if (sortedIndexData != nullptr) {
        if (indexSize > 0) { indexBuffer->uploadToStagingBuffer(sortedIndexData, indexSize, 0); }
        return {vertexId, indexId};
    }

    auto buildQuadIndices = [indexBuffer, vertexCount, expectedIndexCount]<typename V>() {
        const int indexCount = vertexCount / 4 * 6;
        if (indexCount != expectedIndexCount) { throw std::runtime_error("index count not match!"); }

        std::vector<V> indices;
        indices.reserve(static_cast<size_t>(indexCount));
        for (int vertex = 0; vertex < vertexCount; vertex += 4) {
            indices.push_back(vertex + 0);
            indices.push_back(vertex + 1);
            indices.push_back(vertex + 2);
            indices.push_back(vertex + 2);
            indices.push_back(vertex + 3);
            indices.push_back(vertex + 0);
        }
        if (!indices.empty()) {
            indexBuffer->uploadToStagingBuffer(reinterpret_cast<uint8_t *>(indices.data()),
                                               static_cast<uint32_t>(indices.size() * sizeof(V)), 0);
        }
    };

    switch (drawMode) {
        case 7:
            switch (indexType) {
                case 0: buildQuadIndices.template operator()<uint16_t>(); break;
                case 1: buildQuadIndices.template operator()<uint32_t>(); break;
            }
            break;
        default:

            throw std::runtime_error("Unsupported draw mode: " + std::to_string(drawMode));
    }

    return {vertexId, indexId};
}

void Buffers::queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                        std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer) {
    MCVR_PROFILE_SCOPE("native.Buffers.queueImportantWorldUploadPair");
    Renderer::instance().framework()->safeAcquireCurrentContext();
    Renderer::instance().framework()->safeAcquireCurrentContext();
    queueImportantWorldUpload(vertexBuffer);
    queueImportantWorldUpload(indexBuffer);
}

void Buffers::queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer) {
    if (buffer == nullptr) { return; }
    queueImportantWorldUpload(buffer, buffer->size(), 0);
}

void Buffers::queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer, size_t size, size_t offset) {
    MCVR_PROFILE_SCOPE("native.Buffers.queueImportantWorldUpload");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    if (buffer == nullptr) return;
    if (size == 0) { return; }
    if (offset > buffer->size() || size > buffer->size() - offset) {
        throw std::runtime_error("important world upload range exceeds buffer size");
    }
    importantIndexVertexBuffer_->push_back({
        .buffer = std::move(buffer),
        .size = size,
        .offset = offset,
    });
}

void Buffers::performQueuedUpload() {
    performQueuedUpload(true);
}

void Buffers::performQueuedWorldUpload() {
    performQueuedUpload(false);
}

void Buffers::performQueuedUpload(bool includeOverlayUploads) {
    MCVR_PROFILE_SCOPE("native.Buffers.performQueuedUpload");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto frameIndex = context->frameIndex;
    std::shared_ptr<vk::CommandBuffer> cmdBuffer = context->uploadCommandBuffer;

    auto physicalDevice = framework->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> uploadPreBufferBarriers, uploadPostBufferBarriers;

    if (includeOverlayUploads) {
        for (auto [bufferId, size] : validOverlayIndex_[frameIndex]) {
            auto buffer = overlayIndexVertexBuffer_[frameIndex].at(bufferId);
            uploadPreBufferBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = buffer,
            });
            uploadPostBufferBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .buffer = buffer,
            });
        }
    }

    for (const auto &upload : *importantIndexVertexBuffer_) {
        auto buffer = upload.buffer;
        uploadPreBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
            .offset = upload.offset,
            .size = upload.size,
        });
        uploadPostBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
            .offset = upload.offset,
            .size = upload.size,
        });
    }

    cmdBuffer->barriersBufferImage(uploadPreBufferBarriers, {});

    if (includeOverlayUploads) {
        for (auto [bufferId, size] : validOverlayIndex_[frameIndex]) {
            auto buffer = overlayIndexVertexBuffer_[frameIndex].at(bufferId);
            if (size > 0) { buffer->uploadToBuffer(cmdBuffer, size, 0, 0); }
        }
    }

    for (const auto &upload : *importantIndexVertexBuffer_) {
        upload.buffer->uploadToBuffer(cmdBuffer, upload.size, upload.offset, upload.offset);
    }

    cmdBuffer->barriersBufferImage(uploadPostBufferBarriers, {});
}

void Buffers::appendOverlayDrawUniform(uint8_t *srcPointer, uint32_t size, uint32_t &uniformOffset) {
    MCVR_PROFILE_SCOPE("native.Buffers.appendOverlayDrawUniform");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto frameIndex = context->frameIndex;
    auto alignTo = [](uint32_t value, uint32_t alignment) {
        if (alignment <= 1 || value == 0) return value;
        return ((value + alignment - 1) / alignment) * alignment;
    };

    uniformOffset = overlayDrawUniformWriteOffset_[frameIndex];
    uint32_t alignedSize = alignTo(size, overlayDrawUniformAlignment_);
    if (size > overlayDrawUniformDescriptorRange_) {
        throw std::runtime_error("Overlay draw uniform exceeds fixed descriptor range");
    }

    uint32_t requiredSize = uniformOffset + size;
    auto &cpuData = overlayDrawUniformData_[frameIndex];
    if (cpuData.size() < requiredSize) { cpuData.resize(requiredSize, 0); }

    if (size > 0) { std::memcpy(cpuData.data() + uniformOffset, srcPointer, size); }

    overlayDrawUniformWriteOffset_[frameIndex] = uniformOffset + alignedSize;

    if (ensureOverlayDrawUniformBufferCapacityLocked(framework, frameIndex,
                                                     uniformOffset + overlayDrawUniformDescriptorRange_)) {
#ifdef MCVR_UI_BACKEND_VULKAN
        auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
        pipelineContext->uiModuleContext->refreshOverlayDescriptorTable();
#endif
    }
}

bool Buffers::registerOverlayDrawUniformSize(uint32_t size) {
    MCVR_PROFILE_SCOPE("native.Buffers.registerOverlayDrawUniformSize");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return false;

    {
        std::unique_lock<std::recursive_mutex> lck(mtx_);
        uint32_t requiredRange = std::max<uint32_t>(1, size);
        if (requiredRange > overlayDrawUniformDeviceLimit_) {
            throw std::runtime_error("Overlay draw uniform exceeds device maxUniformBufferRange");
        }

        if (requiredRange <= overlayDrawUniformDescriptorRange_) { return false; }

        overlayDrawUniformDescriptorRange_ = requiredRange;
        for (uint32_t frameIndex = 0; frameIndex < overlayDrawUniformBuffer_.size(); ++frameIndex) {
            ensureOverlayDrawUniformBufferCapacityLocked(framework, frameIndex, requiredRange);
        }
    }

    return true;
}

void Buffers::appendOverlayPostUniform(vk::Data::OverlayPostUBO &ubo) {
    MCVR_PROFILE_SCOPE("native.Buffers.appendOverlayPostUniform");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto frameIndex = context->frameIndex;

    uint32_t uniformOffset = overlayPostUniformCount_[frameIndex] * overlayPostUniformStride_;
    uint32_t requiredSize = uniformOffset + overlayPostUniformDescriptorRange_;
    auto &cpuData = overlayPostUniformData_[frameIndex];
    if (cpuData.size() < requiredSize) { cpuData.resize(requiredSize, 0); }

    std::memcpy(cpuData.data() + uniformOffset, &ubo, sizeof(vk::Data::OverlayPostUBO));

    auto &buffer = overlayPostUniformBuffer_[frameIndex];
    if (uniformOffset + overlayPostUniformDescriptorRange_ > buffer->size()) {
        uint32_t requiredBufferSize = uniformOffset + overlayPostUniformDescriptorRange_;
        uint32_t newSize = std::max<uint32_t>(buffer->size(), overlayPostUniformInitialSize);
        while (newSize < requiredBufferSize) { newSize *= 2; }

        auto newBuffer = vk::HostVisibleBuffer::create(framework->vma(), framework->device(), newSize,
                                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        framework->frameResourceRetainer().retain(buffer);
        overlayPostUniformBuffer_[frameIndex] = newBuffer;
        cpuData.reserve(newSize);

#ifdef MCVR_UI_BACKEND_VULKAN
        auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
        pipelineContext->uiModuleContext->refreshOverlayDescriptorTable();
#endif
    }

    overlayPostUniformCount_[frameIndex]++;
}

void Buffers::buildAndUploadOverlayUniformBuffer() {
    MCVR_PROFILE_SCOPE("native.Buffers.buildAndUploadOverlayUniformBuffer");
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto frameIndex = context->frameIndex;

    auto &drawCpuData = overlayDrawUniformData_[frameIndex];
    auto &drawBuffer = overlayDrawUniformBuffer_[frameIndex];
    if (!drawCpuData.empty()) {
        if (drawCpuData.size() > drawBuffer->size()) {
            throw std::runtime_error("Overlay draw uniform buffer exhausted before upload");
        }
        drawBuffer->uploadToBuffer(drawCpuData.data(), drawCpuData.size(), 0);
    }

    auto &postCpuData = overlayPostUniformData_[frameIndex];
    auto &postBuffer = overlayPostUniformBuffer_[frameIndex];
    if (!postCpuData.empty()) {
        if (postCpuData.size() > postBuffer->size()) {
            throw std::runtime_error("Overlay post uniform buffer exhausted before upload");
        }
        postBuffer->uploadToBuffer(postCpuData.data(), postCpuData.size(), 0);
    }
}

static size_t sequenceIndex = 0;

glm::vec2 halton(int index) {
    const glm::vec2 coprimes = glm::vec2(3.0F, 5.0F);
    glm::vec2 s = glm::vec2(index, index);
    glm::vec4 a = glm::vec4(1, 1, 0, 0);
    while (s.x > 0. || s.y > 0.) {
        a.x = a.x / coprimes.x;
        a.y = a.y / coprimes.y;
        a.z += a.x * fmod(s.x, coprimes.x);
        a.w += a.y * fmod(s.y, coprimes.y);
        s.x = floorf(s.x / coprimes.x);
        s.y = floorf(s.y / coprimes.y);
    }
    return glm::vec2(a.z, a.w);
}

void Buffers::setAndUploadWorldUniformBuffer(vk::Data::WorldUBO &ubo) {
    MCVR_PROFILE_SCOPE("native.Buffers.setAndUploadWorldUniformBuffer");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    glm::mat4 mapGLToVulkan(1.0f);
    mapGLToVulkan[1][1] = -1.0f;
    mapGLToVulkan[2][2] = 0.5f;
    mapGLToVulkan[3][2] = 0.5f;

    ubo.cameraProjMat = mapGLToVulkan * ubo.cameraProjMat;

    ubo.cameraViewMatInv = glm::inverse(ubo.cameraViewMat);
    ubo.cameraEffectedViewMatInv = glm::inverse(ubo.cameraEffectedViewMat);
    ubo.cameraProjMatInv = glm::inverse(ubo.cameraProjMat);

    {
        static std::random_device seed;
        static std::ranlux48 engine(seed());
        static std::uniform_int_distribution<> distrib;
        ubo.seed = distrib(engine);
    }

    ubo.cameraJitter = useJitter_ ? halton(++sequenceIndex) - glm::vec2(0.5) : glm::vec2(0.0);

    auto world = Renderer::instance().world();
    ubo.cameraPos.x = world->getCameraPos().x;
    ubo.cameraPos.y = world->getCameraPos().y;
    ubo.cameraPos.z = world->getCameraPos().z;
    ubo.cameraPos.w = 0;
    if (auto chunks = world->chunks(); chunks != nullptr) {
        ubo.chunkGridInfo = chunks->chunkGridInfo();
        ubo.chunkStorageSectionPos = chunks->chunkStorageSectionPos();
    } else {
        ubo.chunkGridInfo = glm::ivec4(0);
        ubo.chunkStorageSectionPos = glm::ivec4(0);
    }

    if (worldUniformBuffer_[context->frameIndex] == nullptr) {
        worldUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::WorldUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    if (lastWorldUniformBuffer_[context->frameIndex] == nullptr) {
        lastWorldUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::WorldUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    worldUniformBuffer_[context->frameIndex]->uploadToBuffer(&ubo);
    lastWorldUniformBuffer_[context->frameIndex]->uploadToBuffer(hasWorldHistory_ ? &lastWorldUniformData_ : &ubo);

    lastWorldUniformData_ = ubo;
    worldUniformUpdated_ = true;
}

void Buffers::resetWorldHistory() {
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    worldUniformUpdated_ = false;
    hasWorldHistory_ = false;
}

bool Buffers::hasWorldHistory() const {
    return hasWorldHistory_;
}

void Buffers::setAndUploadSkyUniformBuffer(vk::Data::SkyUBO &ubo) {
    MCVR_PROFILE_SCOPE("native.Buffers.setAndUploadSkyUniformBuffer");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (skyUniformBuffer_[context->frameIndex] == nullptr) {
        skyUniformBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::SkyUBO),
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    skyUniformBuffer_[context->frameIndex]->uploadToBuffer(&ubo);
}

void Buffers::setAndUploadTextureMappingBuffer(vk::Data::TextureMapping &mapping) {
    MCVR_PROFILE_SCOPE("native.Buffers.setAndUploadTextureMappingBuffer");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (textureMappingBuffer_[context->frameIndex] == nullptr) {
        textureMappingBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::TextureMapping),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    textureMappingBuffer_[context->frameIndex]->uploadToBuffer(&mapping);
}

void Buffers::setAndUploadExposureDataBuffer(vk::Data::ExposureData &exposureData) {
    MCVR_PROFILE_SCOPE("native.Buffers.setAndUploadExposureDataBuffer");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto vma = framework->vma();
    auto device = framework->device();

    if (exposureDataBuffer_[context->frameIndex] == nullptr) {
        exposureDataBuffer_[context->frameIndex] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(vk::Data::ExposureData),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    exposureDataBuffer_[context->frameIndex]->uploadToBuffer(&exposureData);
}

int Buffers::getPostID() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    return static_cast<int>(overlayPostUniformCount_[context->frameIndex]) - 1;
}

std::shared_ptr<vk::DeviceLocalBuffer> Buffers::getBuffer(uint32_t id) {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    auto bufferIter = overlayIndexVertexBuffer_[context->frameIndex].find(id);
    if (!validOverlayIndex_[context->frameIndex].contains(id) ||
        bufferIter == overlayIndexVertexBuffer_[context->frameIndex].end()) {
        buffersCerr() << "The given buffer id: " << id << " is not allocated for buffer" << std::endl;
        exit(EXIT_FAILURE);
    }

    return bufferIter->second;
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::overlayDrawUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    return overlayDrawUniformBuffer_[context->frameIndex];
}

uint32_t Buffers::overlayDrawUniformDescriptorRange() {
    return overlayDrawUniformDescriptorRange_;
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::overlayPostUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();
    return overlayPostUniformBuffer_[context->frameIndex];
}

uint32_t Buffers::overlayPostUniformDescriptorRange() {
    return overlayPostUniformDescriptorRange_;
}

uint32_t Buffers::overlayPostUniformOffset(int postID) {
    if (postID < 0) return 0;
    return static_cast<uint32_t>(postID) * overlayPostUniformStride_;
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::worldUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (worldUniformBuffer_[context->frameIndex]) {
        return worldUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::lastWorldUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (lastWorldUniformBuffer_[context->frameIndex]) {
        return lastWorldUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::skyUniformBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (skyUniformBuffer_[context->frameIndex]) {
        return skyUniformBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::textureMappingBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (textureMappingBuffer_[context->frameIndex]) {
        return textureMappingBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

std::shared_ptr<vk::HostVisibleBuffer> Buffers::exposureDataBuffer() {
    auto context = Renderer::instance().framework()->safeAcquireCurrentContext();

    if (exposureDataBuffer_[context->frameIndex]) {
        return exposureDataBuffer_[context->frameIndex];
    } else {
        return nullptr;
    }
}

void Buffers::setUseJitter(bool useJitter) {
    useJitter_ = useJitter;
}
