
#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

class Framework;

class Buffers : public SharedObject<Buffers> {
  public:
    Buffers(std::shared_ptr<Framework> framework);

    void resetFrame();
    std::pair<uint32_t, uint32_t> createAndUploadVertexIndexBuffers(uint8_t *vertexData,
                                                                    uint32_t vertexSize,
                                                                    uint8_t *sortedIndexData,
                                                                    uint32_t indexSize,
                                                                    int indexType,
                                                                    int drawMode,
                                                                    int vertexCount,
                                                                    int expectedIndexCount);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer, size_t size, size_t offset);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                   std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer);
    void performQueuedUpload();
    void performQueuedWorldUpload();

    bool registerOverlayDrawUniformSize(uint32_t size);
    void appendOverlayDrawUniform(uint8_t *srcPointer, uint32_t size, uint32_t &uniformOffset);
    void appendOverlayPostUniform(vk::Data::OverlayPostUBO &ubo);
    void buildAndUploadOverlayUniformBuffer();

    void setAndUploadWorldUniformBuffer(vk::Data::WorldUBO &ubo);
    void resetWorldHistory();
    bool hasWorldHistory() const;
    void setAndUploadSkyUniformBuffer(vk::Data::SkyUBO &ubo);
    void setAndUploadTextureMappingBuffer(vk::Data::TextureMapping &mapping);
    void setAndUploadExposureDataBuffer(vk::Data::ExposureData &exposureData);

    int getPostID();

    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer(uint32_t id);

    std::shared_ptr<vk::HostVisibleBuffer> overlayDrawUniformBuffer();
    uint32_t overlayDrawUniformDescriptorRange();
    std::shared_ptr<vk::HostVisibleBuffer> overlayPostUniformBuffer();
    uint32_t overlayPostUniformDescriptorRange();
    uint32_t overlayPostUniformOffset(int postID);

    std::shared_ptr<vk::HostVisibleBuffer> worldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> lastWorldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> skyUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> textureMappingBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> exposureDataBuffer();

    void setUseJitter(bool useJitter);

  private:
    static constexpr uint32_t baseBlockSize = 16 * 1024;
    static constexpr uint32_t overlayPostUniformInitialSize = 512 * 1024;
    static constexpr uint32_t overlayDrawUniformInitialSize = 8 * 1024 * 1024;
    static constexpr uint32_t overlayDrawUniformInitialDescriptorRange = 4 * 1024;

    bool ensureOverlayDrawUniformBufferCapacityLocked(std::shared_ptr<Framework> framework,
                                                      uint32_t frameIndex,
                                                      uint32_t requiredBufferSize);
    void performQueuedUpload(bool includeOverlayUploads);

    std::vector<std::map<uint32_t, int32_t>> validOverlayIndex_;
    std::vector<std::map<uint32_t, std::shared_ptr<vk::DeviceLocalBuffer>>> overlayIndexVertexBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> overlayDrawUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> overlayPostUniformBuffer_;
    uint32_t overlayNextID_;
    std::vector<std::vector<uint8_t>> overlayDrawUniformData_;
    std::vector<uint32_t> overlayDrawUniformWriteOffset_;
    uint32_t overlayDrawUniformAlignment_ = 1;
    uint32_t overlayDrawUniformDeviceLimit_ = 1;
    uint32_t overlayDrawUniformDescriptorRange_ = 1;
    std::vector<std::vector<uint8_t>> overlayPostUniformData_;
    std::vector<uint32_t> overlayPostUniformCount_;
    uint32_t overlayPostUniformStride_ = 1;
    uint32_t overlayPostUniformDescriptorRange_ = 1;

    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> worldUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> lastWorldUniformBuffer_;
    vk::Data::WorldUBO lastWorldUniformData_{};
    bool worldUniformUpdated_ = false;
    bool hasWorldHistory_ = false;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> skyUniformBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> textureMappingBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> exposureDataBuffer_;

    struct ImportantWorldUpload {
        std::shared_ptr<vk::DeviceLocalBuffer> buffer;
        size_t size = 0;
        size_t offset = 0;
    };
    std::shared_ptr<std::vector<ImportantWorldUpload>> importantIndexVertexBuffer_;

    bool useJitter_ = true;
    std::recursive_mutex mtx_;
};
