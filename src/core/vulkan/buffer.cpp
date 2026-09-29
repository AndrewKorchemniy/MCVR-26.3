#include "core/vulkan/buffer.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/vma.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>

std::ostream &bufferCerr() {
    return std::cerr << "[Buffer] ";
}

vk::HostVisibleBuffer::HostVisibleBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usage)
    : vma_(vma), device_(device), size_(size), bufferUsage_(usage) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;
    const uint32_t queueFamilies[] = {device_->mainQueueFamilyIndex(), device_->secondaryQueueFamilyIndex()};
    if (queueFamilies[0] != queueFamilies[1]) {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices = queueFamilies;
    }

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    vk::requireSuccess(
        vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &allocationInfo_),
        "vmaCreateBuffer(host-visible buffer)");
    trackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
    mappedPtr_ = allocationInfo_.pMappedData;

    if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::HostVisibleBuffer::HostVisibleBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usage,
                                         VkDeviceSize minAlignment)
    : vma_(vma), device_(device), size_(size), bufferUsage_(usage) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;
    const uint32_t queueFamilies[] = {device_->mainQueueFamilyIndex(), device_->secondaryQueueFamilyIndex()};
    if (queueFamilies[0] != queueFamilies[1]) {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices = queueFamilies;
    }

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    vk::requireSuccess(vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment,
                                                    &buffer_, &allocation_, &allocationInfo_),
                       "vmaCreateBufferWithAlignment(host-visible buffer)");
    trackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
    mappedPtr_ = allocationInfo_.pMappedData;

    if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::HostVisibleBuffer::~HostVisibleBuffer() {
    untrackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
    vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);
}

void vk::HostVisibleBuffer::downloadFromBuffer() {
    downloadFromBuffer(size_, 0);
}

void vk::HostVisibleBuffer::downloadFromBuffer(size_t size, size_t offset) {
    vmaInvalidateAllocation(vma_->allocator(), allocation_, offset, size);
}

void vk::HostVisibleBuffer::uploadToBuffer(void *src) {
    uploadToBuffer(src, size_, 0);
}

void vk::HostVisibleBuffer::uploadToBuffer(void *src, size_t size, size_t offset) {
    std::memcpy(static_cast<uint8_t *>(mappedPtr_) + offset, src, size);
    vmaFlushAllocation(vma_->allocator(), allocation_, offset, size);
}

void vk::HostVisibleBuffer::flush() {
    flush(size_, 0);
}

void vk::HostVisibleBuffer::flush(size_t size, size_t offset) {
    if (size == 0) { return; }
    vmaFlushAllocation(vma_->allocator(), allocation_, offset, size);
}

void vk::HostVisibleBuffer::setMemoryCategory(ResourceMemoryCategory category) {
    retagVmaAllocation(vma_->allocator(), allocation_, memoryCategory_, category);
    memoryCategory_ = category;
}

size_t vk::HostVisibleBuffer::size() {
    return size_;
}

VkBuffer &vk::HostVisibleBuffer::vkBuffer() {
    return buffer_;
}

void *vk::HostVisibleBuffer::mappedPtr() {
    return mappedPtr_;
}

VkDeviceAddress &vk::HostVisibleBuffer::bufferAddress() {
    if (!(bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bufferCerr() << "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT not specified when try to get bufferAddress"
                     << std::endl;
        exit(EXIT_FAILURE);
    }
    return bufferAddress_;
}

vk::TemporaryStagingBuffer::TemporaryStagingBuffer(std::shared_ptr<VMA> vma,
                                                   VkBuffer buffer,
                                                   VmaAllocation allocation,
                                                   ResourceMemoryCategory memoryCategory)
    : vma_(vma), buffer_(buffer), allocation_(allocation), memoryCategory_(memoryCategory) {}

vk::TemporaryStagingBuffer::TemporaryStagingBuffer(std::shared_ptr<HostVisibleStagingBufferPool> pool,
                                                   StagingBufferLease lease)
    : pool_(std::move(pool)), pooledLease_(lease) {}

vk::TemporaryStagingBuffer::~TemporaryStagingBuffer() {
    if (pool_ != nullptr) {
        pool_->release(pooledLease_);
        return;
    }
    if (buffer_ != VK_NULL_HANDLE || allocation_ != VK_NULL_HANDLE) {
        untrackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
        vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);
    }
}

vk::HostVisibleStagingBufferPool::HostVisibleStagingBufferPool(std::shared_ptr<VMA> vma,
                                                               size_t maxCachedBytes,
                                                               ResourceMemoryCategory defaultCategory)
    : vma_(std::move(vma)), maxCachedBytes_(maxCachedBytes), defaultCategory_(defaultCategory) {}

vk::HostVisibleStagingBufferPool::~HostVisibleStagingBufferPool() {
    std::vector<StagingBufferLease> entries;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries.swap(entries_);
        cachedBytes_ = 0;
    }
    for (auto &entry : entries) { destroyLease(entry); }
}

vk::StagingBufferLease vk::HostVisibleStagingBufferPool::acquire(size_t requiredSize, ResourceMemoryCategory category) {
    MCVR_PROFILE_SCOPE("native.HostVisibleStagingBufferPool.acquire");
    MCVR_PROFILE_METRIC("staging_pool.acquire_required_bytes", requiredSize);
    if (requiredSize == 0) { return {}; }

    const ResourceMemoryCategory resolvedCategory =
        category == ResourceMemoryCategory::Count ? defaultCategory_ : category;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t bestIndex = entries_.size();
        for (size_t i = 0; i < entries_.size(); i++) {
            const auto &entry = entries_[i];
            if (entry.size < requiredSize) { continue; }
            if (bestIndex == entries_.size() || entry.size < entries_[bestIndex].size) { bestIndex = i; }
        }

        if (bestIndex != entries_.size()) {
            StagingBufferLease lease = entries_[bestIndex];
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(bestIndex));
            cachedBytes_ -= std::min(cachedBytes_, lease.size);
            lock.unlock();

            retagVmaAllocation(vma_->allocator(), lease.allocation, lease.memoryCategory, resolvedCategory);
            lease.memoryCategory = resolvedCategory;
            MCVR_PROFILE_METRIC("staging_pool.acquire_hit", 1);
            MCVR_PROFILE_METRIC("staging_pool.acquire_capacity_bytes", lease.size);
            MCVR_PROFILE_METRIC("staging_pool.cached_bytes_after_acquire", cachedBytes());
            MCVR_PROFILE_METRIC("staging_pool.cached_count_after_acquire", cachedCount());
            return lease;
        }

        MCVR_PROFILE_METRIC("staging_pool.acquire_hit", 0);
    }

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = requiredSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    StagingBufferLease lease{
        .size = requiredSize,
        .memoryCategory = resolvedCategory,
    };
    vk::requireSuccess(vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &lease.buffer,
                                       &lease.allocation, &lease.allocationInfo),
                       "vmaCreateBuffer(pooled staging buffer)");
    trackVmaAllocation(vma_->allocator(), lease.allocation, lease.memoryCategory);
    if (lease.allocationInfo.pMappedData == nullptr) {
        bufferCerr() << "failed to map pooled staging buffer" << std::endl;
        exit(EXIT_FAILURE);
    }
    MCVR_PROFILE_METRIC("staging_pool.acquire_capacity_bytes", lease.size);
    return lease;
}

void vk::HostVisibleStagingBufferPool::release(StagingBufferLease lease) {
    MCVR_PROFILE_SCOPE("native.HostVisibleStagingBufferPool.release");
    if (!lease) { return; }
    MCVR_PROFILE_METRIC("staging_pool.release_capacity_bytes", lease.size);

    std::unique_lock<std::mutex> lock(mutex_);
    const bool shouldCache = lease.size <= maxCachedBytes_ && cachedBytes_ + lease.size <= maxCachedBytes_;
    if (shouldCache) {
        cachedBytes_ += lease.size;
        entries_.push_back(lease);
        MCVR_PROFILE_METRIC("staging_pool.release_cached", 1);
        MCVR_PROFILE_METRIC("staging_pool.cached_bytes_after_release", cachedBytes_);
        MCVR_PROFILE_METRIC("staging_pool.cached_count_after_release", entries_.size());
        return;
    }
    MCVR_PROFILE_METRIC("staging_pool.release_cached", 0);
    MCVR_PROFILE_METRIC("staging_pool.release_dropped_bytes", lease.size);
    lock.unlock();

    destroyLease(lease);
}

size_t vk::HostVisibleStagingBufferPool::cachedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cachedBytes_;
}

size_t vk::HostVisibleStagingBufferPool::cachedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

void vk::HostVisibleStagingBufferPool::destroyLease(StagingBufferLease &lease) {
    if (lease.buffer != VK_NULL_HANDLE || lease.allocation != VK_NULL_HANDLE) {
        untrackVmaAllocation(vma_->allocator(), lease.allocation, lease.memoryCategory);
        vmaDestroyBuffer(vma_->allocator(), lease.buffer, lease.allocation);
    }
    lease = {};
}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer)
    : DeviceLocalBuffer(vma, device, true, size, usageExceptTransfer) {}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer)
    : DeviceLocalBuffer(
          vma, device, persistStaging, size, usageExceptTransfer, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer,
                                         VmaAllocationCreateFlags vmaAllocationFlags,
                                         VmaMemoryUsage vmaUsage)
    : vma_(vma),
      device_(device),
      persistStaging_(persistStaging),
      size_(size),
      vmaAllocationFlags_(vmaAllocationFlags),
      vmaUsage_(vmaUsage) {
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usageExceptTransfer;
    uint32_t queueFamilyIndices[2] = {device_->mainQueueFamilyIndex(), device_->secondaryQueueFamilyIndex()};
    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    bufferUsage_ = bufferInfo.usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.flags = vmaAllocationFlags;
    allocationInfo.usage = vmaUsage;

    vk::requireSuccess(
        vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &allocationInfo_),
        "vmaCreateBuffer(device-local buffer)");
    trackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);

    if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer,
                                         VmaAllocationCreateFlags vmaAllocationFlags,
                                         VmaMemoryUsage vmaUsage,
                                         VkDeviceSize minAlignment)
    : vma_(vma),
      device_(device),
      persistStaging_(persistStaging),
      size_(size),
      vmaAllocationFlags_(vmaAllocationFlags),
      vmaUsage_(vmaUsage) {
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usageExceptTransfer;
    uint32_t queueFamilyIndices[2] = {device_->mainQueueFamilyIndex(), device_->secondaryQueueFamilyIndex()};
    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    bufferUsage_ = bufferInfo.usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.flags = vmaAllocationFlags;
    allocationInfo.usage = vmaUsage;

    vk::requireSuccess(vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment,
                                                    &buffer_, &allocation_, &allocationInfo_),
                       "vmaCreateBufferWithAlignment(device-local buffer)");
    trackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);

    if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::DeviceLocalBuffer::~DeviceLocalBuffer() {
    if (transientStagingRetainer_ != nullptr) {
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        stagingAllocationInfo_ = {};
        stagingBufferSize_ = 0;
        mappedPtr_ = nullptr;
        transientStagingRetainer_ = nullptr;
    }
    if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE) {
        if (persistStaging_ && stagingBufferPool_ != nullptr) {
            releaseStaging();
        } else {
            untrackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
            vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
            stagingBuffer_ = VK_NULL_HANDLE;
            stagingAllocation_ = VK_NULL_HANDLE;
            stagingAllocationInfo_ = {};
            stagingBufferSize_ = 0;
            mappedPtr_ = nullptr;
        }
    }
    untrackVmaAllocation(vma_->allocator(), allocation_, memoryCategory_);
    vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);
}

void vk::DeviceLocalBuffer::createStagingBuffer() {
    MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.createStagingBuffer");
    if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
        if (stagingBuffer_ == VK_NULL_HANDLE || stagingAllocation_ == VK_NULL_HANDLE || mappedPtr_ == nullptr) {
            bufferCerr() << "staging buffer state is inconsistent" << std::endl;
            exit(EXIT_FAILURE);
        }
        return;
    }

    if (stagingBufferPool_ != nullptr) {
        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.createStagingBufferPooled");
        StagingBufferLease lease = stagingBufferPool_->acquire(size_, memoryCategory_);
        if (!lease) {
            bufferCerr() << "failed to acquire pooled staging buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
        stagingBuffer_ = lease.buffer;
        stagingAllocation_ = lease.allocation;
        stagingAllocationInfo_ = lease.allocationInfo;
        stagingBufferSize_ = lease.size;
        mappedPtr_ = lease.allocationInfo.pMappedData;
        return;
    }

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.createStagingBufferUnpooled");
    vk::requireSuccess(vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_,
                                       &stagingAllocation_, &stagingAllocationInfo_),
                       "vmaCreateBuffer(staging buffer)");
    trackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
    stagingBufferSize_ = size_;
    mappedPtr_ = stagingAllocationInfo_.pMappedData;
    if (mappedPtr_ == nullptr) {
        bufferCerr() << "failed to map staging buffer" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest) {
    downloadFromStagingBuffer(dest, size_, 0);
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest, size_t size, size_t offset) {
    if (offset > size_ || size > size_ - offset) {
        bufferCerr() << "download range exceeds buffer size" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (size > 0 && dest == nullptr) {
        bufferCerr() << "download destination is null" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (size == 0) { return; }

    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        createStagingBuffer();
    } else {
        createStagingBuffer();
    }

    vmaInvalidateAllocation(vma_->allocator(), stagingAllocation_, offset, size);
    std::memcpy(dest, mappedPtr_, size);

    if (!persistStaging_) {
        untrackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        stagingAllocationInfo_ = {};
        stagingBufferSize_ = 0;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalBuffer::uploadToStagingBuffer(void *src) {
    uploadToStagingBuffer(src, size_, 0);
}

void vk::DeviceLocalBuffer::uploadToStagingBuffer(void *src, size_t size, size_t offset) {
    MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.uploadToStagingBuffer");
    if (offset > size_ || size > size_ - offset) {
        bufferCerr() << "upload range exceeds buffer size" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (size > 0 && src == nullptr) {
        bufferCerr() << "upload source is null" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (size == 0) { return; }

    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.createTransientStagingBuffer");
        createStagingBuffer();
        if (stagingBufferPool_ != nullptr) {
            transientStagingRetainer_ =
                TemporaryStagingBuffer::create(stagingBufferPool_, StagingBufferLease{
                                                                       .buffer = stagingBuffer_,
                                                                       .allocation = stagingAllocation_,
                                                                       .allocationInfo = stagingAllocationInfo_,
                                                                       .size = stagingBufferSize_,
                                                                       .memoryCategory = memoryCategory_,
                                                                   });
        } else {
            transientStagingRetainer_ =
                TemporaryStagingBuffer::create(vma_, stagingBuffer_, stagingAllocation_, memoryCategory_);
        }
    } else {
        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.createPersistentStagingBuffer");
        createStagingBuffer();
    }

    {
        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.writeStagingMemory");
        std::memcpy(static_cast<uint8_t *>(mappedPtr_) + offset, src, size);
    }
    {
        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.flushStagingMemory");
        vmaFlushAllocation(vma_->allocator(), stagingAllocation_, offset, size);
    }
}

void vk::DeviceLocalBuffer::flushStagingBuffer() {
    if (!persistStaging_) { return; }
    if (stagingBuffer_ == VK_NULL_HANDLE && stagingAllocation_ == VK_NULL_HANDLE && mappedPtr_ == nullptr) { return; }
    vmaFlushAllocation(vma_->allocator(), stagingAllocation_, 0, size_);
}

void vk::DeviceLocalBuffer::releaseStaging() {
    if (!persistStaging_) { return; }
    if (stagingBuffer_ == VK_NULL_HANDLE && stagingAllocation_ == VK_NULL_HANDLE) { return; }

    if (stagingBufferPool_ != nullptr) {
        stagingBufferPool_->release({
            .buffer = stagingBuffer_,
            .allocation = stagingAllocation_,
            .allocationInfo = stagingAllocationInfo_,
            .size = stagingBufferSize_ == 0 ? size_ : stagingBufferSize_,
            .memoryCategory = memoryCategory_,
        });
    } else {
        untrackVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_);
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
    }
    stagingBuffer_ = VK_NULL_HANDLE;
    stagingAllocation_ = VK_NULL_HANDLE;
    stagingAllocationInfo_ = {};
    stagingBufferSize_ = 0;
    mappedPtr_ = nullptr;
}

void vk::DeviceLocalBuffer::releaseStagingDeferred() {
    if (!persistStaging_) { return; }
    if (stagingBuffer_ == VK_NULL_HANDLE && stagingAllocation_ == VK_NULL_HANDLE) { return; }

    const VkBuffer buffer = stagingBuffer_;
    const VmaAllocation allocation = stagingAllocation_;
    const VmaAllocationInfo allocationInfo = stagingAllocationInfo_;
    const size_t bufferSize = stagingBufferSize_ == 0 ? size_ : stagingBufferSize_;
    const ResourceMemoryCategory memoryCategory = memoryCategory_;
    auto pool = stagingBufferPool_;

    stagingBuffer_ = VK_NULL_HANDLE;
    stagingAllocation_ = VK_NULL_HANDLE;
    stagingAllocationInfo_ = {};
    stagingBufferSize_ = 0;
    mappedPtr_ = nullptr;

    std::shared_ptr<TemporaryStagingBuffer> deferred;
    if (pool != nullptr) {
        deferred = TemporaryStagingBuffer::create(pool, StagingBufferLease{
                                                            .buffer = buffer,
                                                            .allocation = allocation,
                                                            .allocationInfo = allocationInfo,
                                                            .size = bufferSize,
                                                            .memoryCategory = memoryCategory,
                                                        });
    } else {
        deferred = TemporaryStagingBuffer::create(vma_, buffer, allocation, memoryCategory);
    }

    auto *renderer = Renderer::try_instance();
    auto framework = renderer == nullptr ? nullptr : renderer->framework();
    if (framework != nullptr) { framework->frameResourceRetainer().retain(std::move(deferred)); }
}

bool vk::DeviceLocalBuffer::hasStagingBuffer() const {
    return stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE;
}

void vk::DeviceLocalBuffer::setStagingBufferPool(std::shared_ptr<HostVisibleStagingBufferPool> pool) {
    if (stagingBufferPool_ == pool) { return; }
    if (persistStaging_ && hasStagingBuffer()) { releaseStaging(); }
    stagingBufferPool_ = std::move(pool);
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer) {
    downloadFromBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer,
                                               size_t size,
                                               size_t srcOffset,
                                               size_t dstOffset) {
    if (stagingBuffer_ == VK_NULL_HANDLE) { createStagingBuffer(); }
    VkBufferCopy copyRegion = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(cmdBuffer, buffer_, stagingBuffer_, 1, &copyRegion);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer) {
    uploadToBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset) {
    if (stagingBuffer_ == VK_NULL_HANDLE) {
        bufferCerr() << "cannot upload from a released staging buffer" << std::endl;
        exit(EXIT_FAILURE);
    }
    MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.recordCopyBuffer");
    VkBufferCopy copyRegion = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(cmdBuffer, stagingBuffer_, buffer_, 1, &copyRegion);
}

void vk::DeviceLocalBuffer::uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer) {
    uploadToBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer,
                                           size_t size,
                                           size_t srcOffset,
                                           size_t dstOffset) {
    cmdBuffer->retainResource(shared_from_this());
    uploadToBuffer(cmdBuffer->vkCommandBuffer(), size, srcOffset, dstOffset);

    if (!persistStaging_ && transientStagingRetainer_ != nullptr) {
        MCVR_PROFILE_SCOPE("native.DeviceLocalBuffer.retainTransientStaging");
        Renderer::instance().framework()->frameResourceRetainer().retain(transientStagingRetainer_);
        transientStagingRetainer_ = nullptr;
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        stagingAllocationInfo_ = {};
        stagingBufferSize_ = 0;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalBuffer::setMemoryCategory(ResourceMemoryCategory category) {
    retagVmaAllocation(vma_->allocator(), allocation_, memoryCategory_, category);
    retagVmaAllocation(vma_->allocator(), stagingAllocation_, memoryCategory_, category);
    memoryCategory_ = category;
}

size_t vk::DeviceLocalBuffer::size() {
    return size_;
}

VkBuffer &vk::DeviceLocalBuffer::vkStagingBuffer() {
    return stagingBuffer_;
}

VkBuffer &vk::DeviceLocalBuffer::vkBuffer() {
    return buffer_;
}

void *vk::DeviceLocalBuffer::mappedPtr() {
    return mappedPtr_;
}

VkDeviceAddress &vk::DeviceLocalBuffer::bufferAddress() {
    if (!(bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bufferCerr() << "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT not specified when try to get bufferAddress"
                     << std::endl;
        exit(EXIT_FAILURE);
    }
    return bufferAddress_;
}
