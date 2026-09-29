#include "core/vulkan/as.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/vma.hpp"

#include <iostream>
#include <stdexcept>

VkDeviceSize vk::BLASBatchBuilder::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) { return value; }
    return (value + alignment - 1) & ~(alignment - 1);
}

vk::BLAS::BLAS(std::shared_ptr<Device> device,
               VkAccelerationStructureKHR blas,
               std::shared_ptr<DeviceLocalBuffer> blasBuffer)
    : device_(device), blas_(blas), blasBuffer_(blasBuffer) {
    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    deviceAddressInfo.accelerationStructure = blas_;
    blasDeviceAddress_ = vkGetAccelerationStructureDeviceAddressKHR(device->vkDevice(), &deviceAddressInfo);
}

vk::BLAS::~BLAS() {
    vkDestroyAccelerationStructureKHR(device_->vkDevice(), blas_, nullptr);
}

std::shared_ptr<vk::DeviceLocalBuffer> vk::BLAS::blasBuffer() {
    return blasBuffer_;
}

VkAccelerationStructureKHR &vk::BLAS::blas() {
    return blas_;
}

VkDeviceAddress &vk::BLAS::blasDeviceAddress() {
    return blasDeviceAddress_;
}

vk::TLAS::TLAS(std::shared_ptr<Device> device,
               VkAccelerationStructureKHR tlas,
               std::shared_ptr<DeviceLocalBuffer> tlasBuffer)
    : device_(device), tlas_(tlas), tlasBuffer_(tlasBuffer) {
    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    deviceAddressInfo.accelerationStructure = tlas_;
    tlasDeviceAddress_ = vkGetAccelerationStructureDeviceAddressKHR(device->vkDevice(), &deviceAddressInfo);
}

vk::TLAS::~TLAS() {
    vkDestroyAccelerationStructureKHR(device_->vkDevice(), tlas_, nullptr);
}

std::shared_ptr<vk::DeviceLocalBuffer> vk::TLAS::tlasBuffer() {
    return tlasBuffer_;
}

VkAccelerationStructureKHR &vk::TLAS::tlas() {
    return tlas_;
}

VkDeviceAddress &vk::TLAS::tlasDeviceAddress() {
    return tlasDeviceAddress_;
}

vk::BLASBuilder::BLASGeometryBuilder::BLASGeometryBuilder(vk::BLASBuilder &parent) : parent(parent) {}

vk::BLASBuilder::BLASGeometryBuilder &vk::BLASBuilder::BLASGeometryBuilder::definePlaceholderGeometry() {
    VkAccelerationStructureGeometryTrianglesDataKHR trianglesData{};
    trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    trianglesData.vertexData.deviceAddress = 0;
    trianglesData.vertexStride = 0;
    trianglesData.maxVertex = 0;
    trianglesData.indexType = VK_INDEX_TYPE_NONE_KHR;
    trianglesData.indexData.deviceAddress = 0;
    trianglesData.transformData.deviceAddress = 0;

    VkAccelerationStructureGeometryKHR placeholderGeometry{};
    placeholderGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    placeholderGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    placeholderGeometry.geometry.triangles = trianglesData;

    placeholderGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    geometries.push_back(placeholderGeometry);
    primitiveCounts.push_back(0);

    return *this;
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::BLASGeometryBuilder::endGeometries() {
    return parent.shared_from_this();
}

vk::BLASBuilder::BLASBuilder() : geometryBuilder_(*this) {}

std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder> vk::BLASBuilder::beginGeometries() {
    return std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder>(shared_from_this(), &geometryBuilder_);
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    flags_ = flags;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                                       VkAccelerationStructureKHR srcBLAS) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    flags_ = flags;
    srcBLAS_ = srcBLAS;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                                       std::shared_ptr<vk::BLAS> targetBLAS) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    flags_ = flags;
    updateTargetBLAS_ = std::move(targetBLAS);
    srcBLAS_ = updateTargetBLAS_ == nullptr ? VK_NULL_HANDLE : updateTargetBLAS_->blas();
    dstBLAS_ = srcBLAS_;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::querySizeInfo(std::shared_ptr<Device> device) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    sizeInfo_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, geometryBuilder_.primitiveCounts.data(), &sizeInfo_);

    return shared_from_this();
}

const VkAccelerationStructureBuildSizesInfoKHR &vk::BLASBuilder::sizeInfo() const {
    return sizeInfo_;
}

VkDeviceSize vk::BLASBuilder::requiredStorageSize() const {
    return sizeInfo_.accelerationStructureSize;
}

VkDeviceSize vk::BLASBuilder::requiredScratchSize() const {
    return mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? sizeInfo_.buildScratchSize :
                                                                     sizeInfo_.updateScratchSize;
}

bool vk::BLASBuilder::isUpdate() const {
    return mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && updateTargetBLAS_ != nullptr;
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                     VkDeviceSize offset,
                                                                     VkDeviceSize capacity) {
    blasBuffer_ = std::move(buffer);
    blasOffset_ = offset;
    blasCapacity_ = capacity;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                     VkDeviceSize offset) {
    scratchBuffer_ = std::move(buffer);
    scratchOffset_ = offset;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                                  std::shared_ptr<Device> device,
                                                                  std::shared_ptr<VMA> vma) {
    if (mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && updateTargetBLAS_ != nullptr) {
        blasBuffer_ = updateTargetBLAS_->blasBuffer();
        dstBLAS_ = updateTargetBLAS_->blas();
    } else {
        useExternalStorage(DeviceLocalBuffer::create(vma, device, false, requiredStorageSize(),
                                                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                     0, VMA_MEMORY_USAGE_GPU_ONLY, 256),
                           0, requiredStorageSize());
    }

    useExternalScratch(
        DeviceLocalBuffer::create(
            vma, device, false, requiredScratchSize(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0,
            VMA_MEMORY_USAGE_GPU_ONLY,
            physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment),
        0);

    return shared_from_this();
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::buildAndSubmit(std::shared_ptr<Device> device,
                                                          std::shared_ptr<CommandBuffer> commandBuffer) {
    if (mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && updateTargetBLAS_ != nullptr) {
        dstBLAS_ = updateTargetBLAS_->blas();
    } else {
        if (blasBuffer_ == nullptr || scratchBuffer_ == nullptr) {
            throw std::runtime_error("BLAS external storage/scratch is not bound");
        }
        const VkDeviceSize capacity = blasCapacity_ == 0 ? blasBuffer_->size() - blasOffset_ : blasCapacity_;
        if (capacity < requiredStorageSize()) {
            throw std::runtime_error("BLAS external storage capacity is smaller than required");
        }
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = blasBuffer_->vkBuffer();
        createInfo.offset = blasOffset_;
        createInfo.size = sizeInfo_.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        if (vkCreateAccelerationStructureKHR(device->vkDevice(), &createInfo, nullptr, &dstBLAS_) != VK_SUCCESS) {
            std::cerr << "Cannot create BLAS" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress() + scratchOffset_;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);

    builtBLAS_ = updateTargetBLAS_ != nullptr ? updateTargetBLAS_ : BLAS::create(device, dstBLAS_, blasBuffer_);
    commandBuffer->retainResource(shared_from_this());
    return builtBLAS_;
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::build(std::shared_ptr<Device> device) {
    if (mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && updateTargetBLAS_ != nullptr) {
        dstBLAS_ = updateTargetBLAS_->blas();
        builtBLAS_ = updateTargetBLAS_;
        return builtBLAS_;
    }

    if (blasBuffer_ == nullptr) { throw std::runtime_error("BLAS external storage is not bound"); }
    const VkDeviceSize capacity = blasCapacity_ == 0 ? blasBuffer_->size() - blasOffset_ : blasCapacity_;
    if (capacity < requiredStorageSize()) {
        throw std::runtime_error("BLAS external storage capacity is smaller than required");
    }
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer_->vkBuffer();
    createInfo.offset = blasOffset_;
    createInfo.size = sizeInfo_.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR(device->vkDevice(), &createInfo, nullptr, &dstBLAS_) != VK_SUCCESS) {
        std::cerr << "Cannot create BLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    builtBLAS_ = BLAS::create(device, dstBLAS_, blasBuffer_);
    return builtBLAS_;
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::buildExternal(std::shared_ptr<Device> device,
                                                         std::shared_ptr<DeviceLocalBuffer> buffer,
                                                         VkDeviceSize offset) {
    if (mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && updateTargetBLAS_ != nullptr) {
        dstBLAS_ = updateTargetBLAS_->blas();
        builtBLAS_ = updateTargetBLAS_;
        return builtBLAS_;
    }

    useExternalStorage(std::move(buffer), offset, requiredStorageSize());
    return build(device);
}

void vk::BLASBuilder::submit(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    if (scratchBuffer_ == nullptr) { throw std::runtime_error("BLAS external scratch is not bound"); }
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress() + scratchOffset_;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);
    commandBuffer->retainResource(shared_from_this());
}

void vk::BLASBuilder::submitExternal(std::shared_ptr<CommandBuffer> commandBuffer,
                                     VkDeviceAddress scratchBufferAddress) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBufferAddress;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);
    commandBuffer->retainResource(shared_from_this());
}

void vk::BLASBuilder::batchSubmit(const std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                  std::shared_ptr<CommandBuffer> commandBuffer,
                                  std::shared_ptr<void> lifetimeOwner) {
    if (builders.empty()) { return; }
    if (lifetimeOwner == nullptr) { throw std::runtime_error("BLAS batch lifetime owner is not set"); }
    commandBuffer->retainResource(std::move(lifetimeOwner));

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> buildRanges;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> pbuildRanges;
    buildInfos.reserve(builders.size());
    buildRanges.reserve(builders.size());
    pbuildRanges.reserve(builders.size());

    for (int i = 0; i < builders.size(); i++) {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.mode = builders[i]->mode_;
        buildInfo.flags = builders[i]->flags_;
        buildInfo.geometryCount = static_cast<uint32_t>(builders[i]->geometryBuilder_.geometries.size());
        buildInfo.pGeometries = builders[i]->geometryBuilder_.geometries.data();

        buildInfo.srcAccelerationStructure = builders[i]->srcBLAS_;
        buildInfo.dstAccelerationStructure = builders[i]->dstBLAS_;
        if (builders[i]->scratchBuffer_ == nullptr) {
            throw std::runtime_error("BLAS batch builder scratch is not bound");
        }
        buildInfo.scratchData.deviceAddress =
            builders[i]->scratchBuffer_->bufferAddress() + builders[i]->scratchOffset_;
        buildInfos.push_back(buildInfo);

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRange;
        for (const auto &count : builders[i]->geometryBuilder_.primitiveCounts) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = count;
            range.primitiveOffset = 0;
            buildRange.push_back(range);
        }
        buildRanges.push_back(buildRange);
        pbuildRanges.push_back(buildRanges[buildRanges.size() - 1].data());
    }

    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), buildInfos.size(), buildInfos.data(),
                                        pbuildRanges.data());
}

void vk::BLASBuilder::batchSubmitExternal(const std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                          const std::vector<VkDeviceAddress> &scratchBufferAddress,
                                          std::shared_ptr<CommandBuffer> commandBuffer,
                                          std::shared_ptr<void> lifetimeOwner) {
    if (builders.empty()) { return; }
    if (lifetimeOwner == nullptr) { throw std::runtime_error("BLAS batch lifetime owner is not set"); }
    commandBuffer->retainResource(std::move(lifetimeOwner));

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> buildRanges;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> pbuildRanges;
    buildInfos.reserve(builders.size());
    buildRanges.reserve(builders.size());
    pbuildRanges.reserve(builders.size());

    for (int i = 0; i < builders.size(); i++) {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.mode = builders[i]->mode_;
        buildInfo.flags = builders[i]->flags_;
        buildInfo.geometryCount = static_cast<uint32_t>(builders[i]->geometryBuilder_.geometries.size());
        buildInfo.pGeometries = builders[i]->geometryBuilder_.geometries.data();

        buildInfo.srcAccelerationStructure = builders[i]->srcBLAS_;
        buildInfo.dstAccelerationStructure = builders[i]->dstBLAS_;
        buildInfo.scratchData.deviceAddress = scratchBufferAddress[i];
        buildInfos.push_back(buildInfo);

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRange;
        for (const auto &count : builders[i]->geometryBuilder_.primitiveCounts) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = count;
            range.primitiveOffset = 0;
            buildRange.push_back(range);
        }
        buildRanges.push_back(buildRange);
        pbuildRanges.push_back(buildRanges[buildRanges.size() - 1].data());
    }

    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), buildInfos.size(), buildInfos.data(),
                                        pbuildRanges.data());
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBatchBuilder::defineBLASBuilder() {
    auto blasBuilder = BLASBuilder::create();
    builders_.push_back(blasBuilder);
    return blasBuilder;
}

void vk::BLASBatchBuilder::calculateOffsets(VkDeviceSize blasAlignment, VkDeviceSize scratchAlignment) {
    blasOffsets_.clear();
    scratchOffsets_.clear();
    scratchAddresses_.clear();
    totalBlasSize_ = 0;
    totalScratchSize_ = 0;

    for (const auto &builder : builders_) {
        if (builder->isUpdate()) {
            blasOffsets_.push_back(0);
        } else {
            blasOffsets_.push_back(totalBlasSize_);
            totalBlasSize_ = alignUp(totalBlasSize_ + builder->requiredStorageSize(), blasAlignment);
        }

        scratchOffsets_.push_back(totalScratchSize_);
        totalScratchSize_ = alignUp(totalScratchSize_ + builder->requiredScratchSize(), scratchAlignment);
    }
}

VkDeviceSize vk::BLASBatchBuilder::totalBlasSize() const {
    return totalBlasSize_;
}

VkDeviceSize vk::BLASBatchBuilder::totalScratchSize() const {
    return totalScratchSize_;
}

const std::vector<VkDeviceSize> &vk::BLASBatchBuilder::blasOffsets() const {
    return blasOffsets_;
}

const std::vector<VkDeviceSize> &vk::BLASBatchBuilder::scratchOffsets() const {
    return scratchOffsets_;
}

const std::vector<std::shared_ptr<vk::BLASBuilder>> &vk::BLASBatchBuilder::builders() const {
    return builders_;
}

std::shared_ptr<vk::BLASBatchBuilder>
vk::BLASBatchBuilder::useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer,
                                         const std::vector<VkDeviceSize> &offsets,
                                         const std::vector<VkDeviceSize> &capacities) {
    if (offsets.size() != builders_.size()) { throw std::runtime_error("BLAS storage offset count mismatch"); }
    if (!capacities.empty() && capacities.size() != builders_.size()) {
        throw std::runtime_error("BLAS storage capacity count mismatch");
    }

    blasBuffer_ = std::move(buffer);
    for (size_t i = 0; i < builders_.size(); i++) {
        if (builders_[i]->isUpdate()) {
            builders_[i]->blasBuffer_ = builders_[i]->updateTargetBLAS_->blasBuffer();
            builders_[i]->dstBLAS_ = builders_[i]->updateTargetBLAS_->blas();
        } else {
            builders_[i]->useExternalStorage(blasBuffer_, offsets[i], capacities.empty() ? 0 : capacities[i]);
        }
    }
    return shared_from_this();
}

std::shared_ptr<vk::BLASBatchBuilder>
vk::BLASBatchBuilder::useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer,
                                         const std::vector<VkDeviceSize> &offsets) {
    if (offsets.size() != builders_.size()) { throw std::runtime_error("BLAS scratch offset count mismatch"); }

    scratchBuffer_ = std::move(buffer);
    scratchAddresses_.clear();
    scratchAddresses_.reserve(builders_.size());
    for (size_t i = 0; i < builders_.size(); i++) {
        builders_[i]->useExternalScratch(scratchBuffer_, offsets[i]);
        scratchAddresses_.push_back(scratchBuffer_->bufferAddress() + offsets[i]);
    }
    return shared_from_this();
}

std::shared_ptr<vk::BLASBatchBuilder> vk::BLASBatchBuilder::allocateBuffers(
    std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device, std::shared_ptr<VMA> vma) {
    const VkDeviceSize scratchAlignment =
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;
    const VkDeviceSize blasAlignment = 256;
    calculateOffsets(blasAlignment, scratchAlignment);

    if (totalBlasSize_ > 0) {
        blasBuffer_ = DeviceLocalBuffer::create(vma, device, false, totalBlasSize_,
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                0, VMA_MEMORY_USAGE_GPU_ONLY, 256);
        useExternalStorage(blasBuffer_, blasOffsets_);
    }

    scratchBuffer_ = DeviceLocalBuffer::create(
        vma, device, false, totalScratchSize_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY,
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);
    useExternalScratch(scratchBuffer_, scratchOffsets_);

    return shared_from_this();
}

std::vector<std::shared_ptr<vk::BLAS>>
vk::BLASBatchBuilder::build(std::shared_ptr<Device> device, const std::vector<std::shared_ptr<BLAS>> &reusedBlases) {
    if (!reusedBlases.empty() && reusedBlases.size() != builders_.size()) {
        throw std::runtime_error("BLAS reuse count mismatch");
    }
    std::vector<std::shared_ptr<vk::BLAS>> results;
    results.reserve(builders_.size());
    pendingBuilders_.clear();
    pendingBuilders_.reserve(builders_.size());
    wasSubmitted_ = false;

    for (size_t i = 0; i < builders_.size(); ++i) {
        const auto &builder = builders_[i];
        if (!reusedBlases.empty() && reusedBlases[i] != nullptr) {
            if (builder->isUpdate() || reusedBlases[i]->blasBuffer() != builder->blasBuffer_) {
                throw std::runtime_error("BLAS reuse storage mismatch");
            }
            builder->builtBLAS_ = reusedBlases[i];
            builder->dstBLAS_ = reusedBlases[i]->blas();
            results.push_back(reusedBlases[i]);
            continue;
        }
        auto result = builder->updateTargetBLAS_ == nullptr ? builder->build(device) : builder->updateTargetBLAS_;
        if (builder->updateTargetBLAS_ != nullptr) { builder->dstBLAS_ = builder->updateTargetBLAS_->blas(); }
        results.push_back(result);
        pendingBuilders_.push_back(builder);
    }

    wasSubmitted_ = pendingBuilders_.empty();
    return results;
}

size_t vk::BLASBatchBuilder::pendingBuildCount() const {
    return pendingBuilders_.size();
}

bool vk::BLASBatchBuilder::wasSubmitted() const {
    return wasSubmitted_;
}

void vk::BLASBatchBuilder::submit(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    BLASBuilder::batchSubmit(pendingBuilders_, commandBuffer, shared_from_this());
    wasSubmitted_ = true;
}

vk::TLASBuilder::TLASInstanceBuilder::TLASInstanceBuilder(TLASBuilder &parent) : parent(parent) {}

vk::TLASBuilder::TLASInstanceBuilder &
vk::TLASBuilder::TLASInstanceBuilder::defineInstance(VkTransformMatrixKHR transform,
                                                     uint32_t customIndex,
                                                     uint32_t mask,
                                                     uint32_t offset,
                                                     VkGeometryInstanceFlagsKHR flag,
                                                     std::shared_ptr<BLAS> blas) {
    instances.emplace_back(transform, customIndex, mask, offset, flag, blas);
    return *this;
}

std::vector<VkAccelerationStructureInstanceKHR> vk::TLASBuilder::TLASInstanceBuilder::buildInstances() const {
    std::vector<VkAccelerationStructureInstanceKHR> asInstances(instances.size());
    for (int i = 0; i < instances.size(); i++) {
        asInstances[i].transform = std::get<0>(instances[i]);
        asInstances[i].instanceCustomIndex = std::get<1>(instances[i]);
        asInstances[i].mask = std::get<2>(instances[i]);
        asInstances[i].instanceShaderBindingTableRecordOffset = std::get<3>(instances[i]);
        asInstances[i].flags = std::get<4>(instances[i]);
        asInstances[i].accelerationStructureReference = std::get<5>(instances[i])->blasDeviceAddress();
    }
    return asInstances;
}

vk::TLASBuilder::TLASInstanceBuilder &
vk::TLASBuilder::TLASInstanceBuilder::defineInstanceBuffer(std::shared_ptr<DeviceLocalBuffer> buffer, uint32_t count) {
    instanceBuffer = buffer;
    instanceCount = count;
    uploadInstanceBuffer = false;
    geometries.clear();

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBuffer->bufferAddress();
    geometries.push_back(geometry);

    return *this;
}

std::shared_ptr<vk::TLASBuilder>
vk::TLASBuilder::TLASInstanceBuilder::endInstanceBuilder(std::shared_ptr<Device> device, std::shared_ptr<VMA> vma) {
    std::vector<VkAccelerationStructureInstanceKHR> asInstances = buildInstances();

    instanceBuffer =
        DeviceLocalBuffer::create(vma, device, false, sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                  0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 16);
    instanceBuffer->uploadToStagingBuffer(asInstances.data());
    instanceCount = static_cast<uint32_t>(instances.size());
    uploadInstanceBuffer = true;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBuffer->bufferAddress();
    geometries.clear();
    geometries.push_back(geometry);

    return parent.shared_from_this();
}

vk::TLASBuilder::TLASBuilder() : tlasInstanceBuilder_(*this) {}

vk::TLASBuilder::TLASBuilder::TLASInstanceBuilder &vk::TLASBuilder::beginInstanceBuilder() {
    return tlasInstanceBuilder_;
}

bool vk::TLASBuilder::canUpdate(VkBuildAccelerationStructureFlagsKHR flags,
                                const std::shared_ptr<TLAS> &targetTLAS) const {
    if (targetTLAS == nullptr || (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) == 0 ||
        flags != targetTLAS->buildFlags_ || tlasInstanceBuilder_.instanceCount != targetTLAS->instanceCount_ ||
        tlasInstanceBuilder_.geometries.size() != targetTLAS->geometrySignature_.size() ||
        tlasInstanceBuilder_.instances.size() != targetTLAS->instanceBlases_.size()) {
        return false;
    }
    for (size_t i = 0; i < tlasInstanceBuilder_.geometries.size(); ++i) {
        const auto &geometry = tlasInstanceBuilder_.geometries[i];
        if (geometry.geometryType != targetTLAS->geometrySignature_[i].first ||
            geometry.flags != targetTLAS->geometrySignature_[i].second ||
            geometry.geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR || geometry.geometry.instances.arrayOfPointers) {
            return false;
        }
    }
    for (size_t i = 0; i < tlasInstanceBuilder_.instances.size(); ++i) {
        const auto &blas = std::get<5>(tlasInstanceBuilder_.instances[i]);
        if (blas == nullptr || blas != targetTLAS->instanceBlases_[i]) { return false; }
    }
    return true;
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    flags_ = flags;
    srcTLAS_ = VK_NULL_HANDLE;
    updateTargetTLAS_.reset();
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                                       std::shared_ptr<TLAS> targetTLAS) {
    if (!canUpdate(flags, targetTLAS)) { throw std::runtime_error("TLAS update dependencies or build parameters changed"); }
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    flags_ = flags;
    updateTargetTLAS_ = std::move(targetTLAS);
    srcTLAS_ = updateTargetTLAS_->tlas();
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::querySizeInfo(std::shared_ptr<Device> device) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(tlasInstanceBuilder_.geometries.size());
    buildInfo.pGeometries = tlasInstanceBuilder_.geometries.data();

    uint32_t instanceCount = tlasInstanceBuilder_.instanceCount;
    sizeInfo_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, &instanceCount, &sizeInfo_);
    return shared_from_this();
}

const VkAccelerationStructureBuildSizesInfoKHR &vk::TLASBuilder::sizeInfo() const {
    return sizeInfo_;
}

VkDeviceSize vk::TLASBuilder::requiredStorageSize() const {
    return sizeInfo_.accelerationStructureSize;
}

VkDeviceSize vk::TLASBuilder::requiredScratchSize() const {
    return mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? sizeInfo_.buildScratchSize :
                                                                     sizeInfo_.updateScratchSize;
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                     VkDeviceSize offset,
                                                                     VkDeviceSize capacity) {
    tlasBuffer_ = std::move(buffer);
    tlasOffset_ = offset;
    tlasCapacity_ = capacity;
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                     VkDeviceSize offset) {
    scratchBuffer_ = std::move(buffer);
    scratchOffset_ = offset;
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                                  std::shared_ptr<Device> device,
                                                                  std::shared_ptr<VMA> vma) {
    useExternalStorage(DeviceLocalBuffer::create(vma, device, false, requiredStorageSize(),
                                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                 0, VMA_MEMORY_USAGE_GPU_ONLY, 256),
                       0, requiredStorageSize());

    useExternalScratch(
        DeviceLocalBuffer::create(
            vma, device, false, requiredScratchSize(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0,
            VMA_MEMORY_USAGE_GPU_ONLY,
            physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment),
        0);

    return shared_from_this();
}

std::shared_ptr<vk::TLAS> vk::TLASBuilder::buildAndSubmit(std::shared_ptr<Device> device,
                                                          std::shared_ptr<CommandBuffer> commandBuffer) {
    if (updateTargetTLAS_ != nullptr) {
        if (!canUpdate(flags_, updateTargetTLAS_)) {
            throw std::runtime_error("TLAS update dependencies or build parameters changed");
        }
        tlasBuffer_ = updateTargetTLAS_->tlasBuffer();
        tlasOffset_ = updateTargetTLAS_->storageOffset_;
        tlasCapacity_ = updateTargetTLAS_->storageSize_;
        dstTLAS_ = updateTargetTLAS_->tlas();
    }
    if (tlasInstanceBuilder_.uploadInstanceBuffer) {
        tlasInstanceBuilder_.instanceBuffer->uploadToBuffer(commandBuffer);
    }
    std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers{{
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = tlasInstanceBuilder_.instanceBuffer,
    }};
    commandBuffer->barriersBufferImage(bufferBarriers, {});

    VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
    tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    if (tlasBuffer_ == nullptr || scratchBuffer_ == nullptr) {
        throw std::runtime_error("TLAS external storage/scratch is not bound");
    }
    if (tlasOffset_ > tlasBuffer_->size() || tlasOffset_ % 256 != 0) {
        throw std::runtime_error("TLAS external storage offset is invalid");
    }
    const VkDeviceSize remainingStorage = tlasBuffer_->size() - tlasOffset_;
    const VkDeviceSize capacity = tlasCapacity_ == 0 ? remainingStorage : tlasCapacity_;
    if (capacity > remainingStorage) { throw std::runtime_error("TLAS external storage range exceeds buffer size"); }
    if (capacity < requiredStorageSize()) {
        throw std::runtime_error("TLAS external storage capacity is smaller than required");
    }
    if (scratchOffset_ > scratchBuffer_->size() ||
        requiredScratchSize() > scratchBuffer_->size() - scratchOffset_) {
        throw std::runtime_error("TLAS scratch range exceeds buffer size");
    }
    tlasCreateInfo.buffer = tlasBuffer_->vkBuffer();
    tlasCreateInfo.offset = tlasOffset_;
    tlasCreateInfo.size = sizeInfo_.accelerationStructureSize;
    tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    if (updateTargetTLAS_ == nullptr &&
        vkCreateAccelerationStructureKHR(device->vkDevice(), &tlasCreateInfo, nullptr, &dstTLAS_) != VK_SUCCESS) {
        std::cerr << "Cannot create TLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(tlasInstanceBuilder_.geometries.size());
    buildInfo.pGeometries = tlasInstanceBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcTLAS_;
    buildInfo.dstAccelerationStructure = dstTLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress() + scratchOffset_;

    VkAccelerationStructureBuildRangeInfoKHR buildRanges{};
    buildRanges.primitiveCount = tlasInstanceBuilder_.instanceCount;
    buildRanges.primitiveOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = &buildRanges;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);

    builtTLAS_ = updateTargetTLAS_ != nullptr ? updateTargetTLAS_ : TLAS::create(device, dstTLAS_, tlasBuffer_);
    if (updateTargetTLAS_ == nullptr) {
        builtTLAS_->storageOffset_ = tlasOffset_;
        builtTLAS_->storageSize_ = tlasCreateInfo.size;
        builtTLAS_->buildFlags_ = flags_;
        builtTLAS_->instanceCount_ = tlasInstanceBuilder_.instanceCount;
        for (const auto &geometry : tlasInstanceBuilder_.geometries) {
            builtTLAS_->geometrySignature_.emplace_back(geometry.geometryType, geometry.flags);
        }
        builtTLAS_->instanceBlases_.reserve(tlasInstanceBuilder_.instances.size());
        for (const auto &instance : tlasInstanceBuilder_.instances) {
            builtTLAS_->instanceBlases_.push_back(std::get<5>(instance));
        }
    }
    commandBuffer->retainResource(shared_from_this());
    return builtTLAS_;
}
