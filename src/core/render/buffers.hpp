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
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> buffer, size_t size, size_t offset);
    void queueImportantWorldUpload(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                   std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer);
    void performQueuedUpload();

    void setAndUploadWorldUniformBuffer(vk::Data::WorldUBO &ubo);
    void resetWorldHistory();
    bool hasWorldHistory() const;
    void setAndUploadSkyUniformBuffer(vk::Data::SkyUBO &ubo);
    void setAndUploadTextureMappingBuffer(vk::Data::TextureMapping &mapping);
    void setAndUploadExposureDataBuffer(vk::Data::ExposureData &exposureData);

    std::shared_ptr<vk::HostVisibleBuffer> worldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> lastWorldUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> skyUniformBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> textureMappingBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> exposureDataBuffer();

    void setUseJitter(bool useJitter);

  private:
    static constexpr uint32_t baseBlockSize = 16 * 1024;

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
