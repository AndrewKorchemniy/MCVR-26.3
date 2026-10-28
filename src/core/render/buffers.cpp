#include "core/render/buffers.hpp"

#include "common/shared.hpp"
#include "core/render/chunks.hpp"
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

    importantIndexVertexBuffer_ = std::make_shared<std::vector<ImportantWorldUpload>>();

    worldUniformBuffer_.resize(size);
    lastWorldUniformBuffer_.resize(size);
    skyUniformBuffer_.resize(size);
    textureMappingBuffer_.resize(size);
    exposureDataBuffer_.resize(size);
}

void Buffers::resetFrame() {
    MCVR_PROFILE_SCOPE("native.Buffers.resetFrame");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    auto &frr = framework->frameResourceRetainer();

    hasWorldHistory_ = worldUniformUpdated_;
    worldUniformUpdated_ = false;

    frr.retain(importantIndexVertexBuffer_);
    importantIndexVertexBuffer_ = std::make_shared<std::vector<ImportantWorldUpload>>();
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
    MCVR_PROFILE_SCOPE("native.Buffers.performQueuedUpload");
    std::unique_lock<std::recursive_mutex> lck(mtx_);
    auto framework = Renderer::instance().framework();
    auto context = framework->safeAcquireCurrentContext();
    std::shared_ptr<vk::CommandBuffer> cmdBuffer = context->uploadCommandBuffer;

    auto physicalDevice = framework->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> uploadPreBufferBarriers, uploadPostBufferBarriers;

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

    for (const auto &upload : *importantIndexVertexBuffer_) {
        upload.buffer->uploadToBuffer(cmdBuffer, upload.size, upload.offset, upload.offset);
    }

    cmdBuffer->barriersBufferImage(uploadPostBufferBarriers, {});
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
