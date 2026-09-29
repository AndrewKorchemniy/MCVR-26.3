#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace vk {
class VMA;
class Device;
class CommandBuffer;
class TemporaryStagingBuffer;

struct StagingBufferLease {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    size_t size = 0;
    ResourceMemoryCategory memoryCategory = ResourceMemoryCategory::Other;

    explicit operator bool() const {
        return buffer != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE && allocationInfo.pMappedData != nullptr;
    }
};

class HostVisibleStagingBufferPool : public SharedObject<HostVisibleStagingBufferPool> {
  public:
    HostVisibleStagingBufferPool(std::shared_ptr<VMA> vma,
                                 size_t maxCachedBytes,
                                 ResourceMemoryCategory defaultCategory);
    ~HostVisibleStagingBufferPool();

    StagingBufferLease acquire(size_t requiredSize, ResourceMemoryCategory category);
    void release(StagingBufferLease lease);

    size_t cachedBytes() const;
    size_t cachedCount() const;

  private:
    void destroyLease(StagingBufferLease &lease);

    std::shared_ptr<VMA> vma_;
    size_t maxCachedBytes_;
    ResourceMemoryCategory defaultCategory_;
    mutable std::mutex mutex_;
    std::vector<StagingBufferLease> entries_;
    size_t cachedBytes_ = 0;
};

class Buffer {
  public:
    virtual size_t size() = 0;
    virtual VkBuffer &vkBuffer() = 0;
};

class HostVisibleBuffer : public Buffer, public SharedObject<HostVisibleBuffer> {
  public:
    HostVisibleBuffer(std::shared_ptr<VMA> vma, std::shared_ptr<Device> device, size_t size, VkBufferUsageFlags usage);
    HostVisibleBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      size_t size,
                      VkBufferUsageFlags usage,
                      VkDeviceSize minAlignment);
    ~HostVisibleBuffer();

    void downloadFromBuffer();
    void downloadFromBuffer(size_t size, size_t offset);

    void uploadToBuffer(void *src);
    void uploadToBuffer(void *src, size_t size, size_t offset);

    void flush();
    void flush(size_t size, size_t offset);

    void setMemoryCategory(ResourceMemoryCategory category);

    size_t size() override;
    VkBuffer &vkBuffer() override;
    void *mappedPtr();
    VkDeviceAddress &bufferAddress();

  private:
    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Device> device_;

    size_t size_;
    VkBufferUsageFlags bufferUsage_;
    void *mappedPtr_ = nullptr;
    VkDeviceAddress bufferAddress_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo_;
    ResourceMemoryCategory memoryCategory_ = ResourceMemoryCategory::Other;
};

class DeviceLocalBuffer : public Buffer, public SharedObject<DeviceLocalBuffer> {
  public:
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer,
                      VmaAllocationCreateFlags allocationFlag,
                      VmaMemoryUsage vmaUsage);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer,
                      VmaAllocationCreateFlags allocationFlag,
                      VmaMemoryUsage vmaUsage,
                      VkDeviceSize minAlignment);
    ~DeviceLocalBuffer();

    void downloadFromStagingBuffer(void *dest);
    void downloadFromStagingBuffer(void *dest, size_t size, size_t offset);

    void uploadToStagingBuffer(void *src);
    void uploadToStagingBuffer(void *src, size_t size, size_t offset);
    void flushStagingBuffer();
    void releaseStaging();
    void releaseStagingDeferred();
    bool hasStagingBuffer() const;
    void setStagingBufferPool(std::shared_ptr<HostVisibleStagingBufferPool> pool);

    void downloadFromBuffer(VkCommandBuffer cmdBuffer);
    void downloadFromBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    void uploadToBuffer(VkCommandBuffer cmdBuffer);
    void uploadToBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    void uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer);
    void uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    void setMemoryCategory(ResourceMemoryCategory category);

    size_t size() override;
    VkBuffer &vkStagingBuffer();
    VkBuffer &vkBuffer() override;
    void *mappedPtr();
    VkDeviceAddress &bufferAddress();

  private:
    void createStagingBuffer();

    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Device> device_;

    bool persistStaging_;
    std::shared_ptr<HostVisibleStagingBufferPool> stagingBufferPool_ = nullptr;
    std::shared_ptr<TemporaryStagingBuffer> transientStagingRetainer_ = nullptr;
    size_t size_;
    size_t stagingBufferSize_ = 0;
    void *mappedPtr_ = nullptr;
    VkBufferUsageFlags bufferUsage_;
    VmaAllocationCreateFlags vmaAllocationFlags_;
    VmaMemoryUsage vmaUsage_;
    VkDeviceAddress bufferAddress_ = 0;
    VkBuffer stagingBuffer_ = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocationInfo_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo_;
    ResourceMemoryCategory memoryCategory_ = ResourceMemoryCategory::Other;
};

class TemporaryStagingBuffer : public SharedObject<TemporaryStagingBuffer> {
  public:
    TemporaryStagingBuffer(std::shared_ptr<VMA> vma,
                           VkBuffer buffer,
                           VmaAllocation allocation,
                           ResourceMemoryCategory memoryCategory);
    TemporaryStagingBuffer(std::shared_ptr<HostVisibleStagingBufferPool> pool, StagingBufferLease lease);
    ~TemporaryStagingBuffer();

  private:
    std::shared_ptr<VMA> vma_;
    std::shared_ptr<HostVisibleStagingBufferPool> pool_;
    StagingBufferLease pooledLease_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    ResourceMemoryCategory memoryCategory_ = ResourceMemoryCategory::Other;
};
};
