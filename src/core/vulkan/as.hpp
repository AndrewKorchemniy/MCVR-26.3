#pragma once

#include "core/all_extern.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/vertex.hpp"

#include <vector>
#include <utility>

namespace vk {
class DeviceLocalBuffer;
class CommandBuffer;
class PhysicalDevice;
class Device;
class VMA;

class BLAS : public SharedObject<BLAS> {
  public:
    BLAS(std::shared_ptr<Device> device,
         VkAccelerationStructureKHR blas,
         std::shared_ptr<DeviceLocalBuffer> blasBuffer);
    ~BLAS();

    std::shared_ptr<DeviceLocalBuffer> blasBuffer();
    VkAccelerationStructureKHR &blas();
    VkDeviceAddress &blasDeviceAddress();

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;

    VkAccelerationStructureKHR blas_;
    VkDeviceAddress blasDeviceAddress_;
};

class TLAS : public SharedObject<TLAS> {
    friend class TLASBuilder;

  public:
    TLAS(std::shared_ptr<Device> device,
         VkAccelerationStructureKHR tlas,
         std::shared_ptr<DeviceLocalBuffer> tlasBuffer);
    ~TLAS();

    std::shared_ptr<DeviceLocalBuffer> tlasBuffer();
    VkAccelerationStructureKHR &tlas();
    VkDeviceAddress &tlasDeviceAddress();

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<DeviceLocalBuffer> tlasBuffer_;

    VkAccelerationStructureKHR tlas_;
    VkDeviceAddress tlasDeviceAddress_;
    VkDeviceSize storageOffset_ = 0;
    VkDeviceSize storageSize_ = 0;
    VkBuildAccelerationStructureFlagsKHR buildFlags_ = 0;
    uint32_t instanceCount_ = 0;
    std::vector<std::pair<VkGeometryTypeKHR, VkGeometryFlagsKHR>> geometrySignature_;
    std::vector<std::shared_ptr<BLAS>> instanceBlases_;
};

class BLASBatchBuilder;

class BLASBuilder : public SharedObject<BLASBuilder> {
    friend BLASBatchBuilder;

  public:
    struct BLASGeometryBuilder {
        BLASBuilder &parent;

        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        std::vector<uint32_t> primitiveCounts;

        BLASGeometryBuilder(BLASBuilder &parent);

        template <typename T>
        BLASGeometryBuilder &defineTriangleGeomrtry(std::shared_ptr<DeviceLocalBuffer> vertexBuffer,
                                                    uint32_t numVertices,
                                                    std::shared_ptr<DeviceLocalBuffer> indexBuffer,
                                                    uint32_t numIndices,
                                                    bool isOpaque);
        template <typename T>
        BLASGeometryBuilder &defineTriangleGeomrtry(VkDeviceAddress vertexBufferAddress,
                                                    uint32_t numVertices,
                                                    VkDeviceAddress indexBufferAddress,
                                                    uint32_t numIndices,
                                                    bool isOpaque);

        BLASGeometryBuilder &definePlaceholderGeometry();

        std::shared_ptr<BLASBuilder> endGeometries();
    };

  public:
    BLASBuilder();

    std::shared_ptr<BLASGeometryBuilder> beginGeometries();
    std::shared_ptr<BLASBuilder> defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags);
    std::shared_ptr<BLASBuilder> defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                      VkAccelerationStructureKHR srcBLAS);
    std::shared_ptr<BLASBuilder> defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                      std::shared_ptr<BLAS> targetBLAS);
    std::shared_ptr<BLASBuilder> querySizeInfo(std::shared_ptr<Device> device);
    const VkAccelerationStructureBuildSizesInfoKHR &sizeInfo() const;
    VkDeviceSize requiredStorageSize() const;
    VkDeviceSize requiredScratchSize() const;
    bool isUpdate() const;
    std::shared_ptr<BLASBuilder>
    useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset = 0, VkDeviceSize capacity = 0);
    std::shared_ptr<BLASBuilder> useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset = 0);
    std::shared_ptr<BLASBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                 std::shared_ptr<Device> device,
                                                 std::shared_ptr<VMA> vma);
    std::shared_ptr<BLAS> buildAndSubmit(std::shared_ptr<Device> device, std::shared_ptr<CommandBuffer> commandBuffer);
    std::shared_ptr<BLAS> build(std::shared_ptr<Device> device);
    std::shared_ptr<BLAS>
    buildExternal(std::shared_ptr<Device> device, std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset);
    void submit(std::shared_ptr<CommandBuffer> commandBuffer);
    void submitExternal(std::shared_ptr<CommandBuffer> commandBuffer, VkDeviceAddress scratchBufferAddress);
    static void batchSubmit(const std::vector<std::shared_ptr<BLASBuilder>> &builders,
                            std::shared_ptr<CommandBuffer> commandBuffer,
                            std::shared_ptr<void> lifetimeOwner);
    static void batchSubmitExternal(const std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                    const std::vector<VkDeviceAddress> &scratchBufferAddress,
                                    std::shared_ptr<CommandBuffer> commandBuffer,
                                    std::shared_ptr<void> lifetimeOwner);

  private:
    BLASGeometryBuilder geometryBuilder_;

    VkBuildAccelerationStructureModeKHR mode_;
    VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo_{};

    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;
    std::shared_ptr<BLAS> updateTargetBLAS_;
    std::shared_ptr<BLAS> builtBLAS_;
    VkDeviceSize blasOffset_ = 0;
    VkDeviceSize blasCapacity_ = 0;
    VkDeviceSize scratchOffset_ = 0;

    VkAccelerationStructureKHR srcBLAS_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR dstBLAS_ = VK_NULL_HANDLE;
};

class BLASBatchBuilder : public SharedObject<BLASBatchBuilder> {
  public:
    std::shared_ptr<BLASBuilder> defineBLASBuilder();
    void calculateOffsets(VkDeviceSize blasAlignment, VkDeviceSize scratchAlignment);
    VkDeviceSize totalBlasSize() const;
    VkDeviceSize totalScratchSize() const;
    const std::vector<VkDeviceSize> &blasOffsets() const;
    const std::vector<VkDeviceSize> &scratchOffsets() const;
    const std::vector<std::shared_ptr<BLASBuilder>> &builders() const;
    std::shared_ptr<BLASBatchBuilder> useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                         const std::vector<VkDeviceSize> &offsets,
                                                         const std::vector<VkDeviceSize> &capacities = {});
    std::shared_ptr<BLASBatchBuilder> useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                         const std::vector<VkDeviceSize> &offsets);
    std::shared_ptr<BLASBatchBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                      std::shared_ptr<Device> device,
                                                      std::shared_ptr<VMA> vma);
    std::vector<std::shared_ptr<BLAS>> build(std::shared_ptr<Device> device,
                                             const std::vector<std::shared_ptr<BLAS>> &reusedBlases = {});
    size_t pendingBuildCount() const;
    bool wasSubmitted() const;
    void submit(std::shared_ptr<CommandBuffer> commandBuffer);

  private:
    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);

    std::vector<std::shared_ptr<BLASBuilder>> builders_;
    std::vector<std::shared_ptr<BLASBuilder>> pendingBuilders_;
    bool wasSubmitted_ = false;

    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;

    std::vector<VkDeviceSize> blasOffsets_;
    std::vector<VkDeviceSize> scratchOffsets_;
    std::vector<VkDeviceAddress> scratchAddresses_;
    VkDeviceSize totalBlasSize_ = 0;
    VkDeviceSize totalScratchSize_ = 0;
};

class TLASBuilder : public SharedObject<TLASBuilder> {
  public:
    struct TLASInstanceBuilder {
        TLASBuilder &parent;

        std::vector<std::tuple<VkTransformMatrixKHR,
                               uint32_t,
                               uint32_t,
                               uint32_t,
                               VkGeometryInstanceFlagsKHR,
                               std::shared_ptr<BLAS>>>
            instances;

        std::shared_ptr<DeviceLocalBuffer> instanceBuffer;
        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        uint32_t instanceCount = 0;
        bool uploadInstanceBuffer = true;

        TLASInstanceBuilder(TLASBuilder &parent);

        TLASInstanceBuilder &defineInstance(VkTransformMatrixKHR transform,
                                            uint32_t customIndex,
                                            uint32_t mask,
                                            uint32_t offset,
                                            VkGeometryInstanceFlagsKHR flag,
                                            std::shared_ptr<BLAS> blas);
        std::vector<VkAccelerationStructureInstanceKHR> buildInstances() const;
        TLASInstanceBuilder &defineInstanceBuffer(std::shared_ptr<DeviceLocalBuffer> buffer, uint32_t count);
        std::shared_ptr<TLASBuilder> endInstanceBuilder(std::shared_ptr<Device> device, std::shared_ptr<VMA> vma);
    };

  public:
    TLASBuilder();

    TLASInstanceBuilder &beginInstanceBuilder();
    bool canUpdate(VkBuildAccelerationStructureFlagsKHR flags, const std::shared_ptr<TLAS> &targetTLAS) const;
    std::shared_ptr<TLASBuilder> defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags);
    std::shared_ptr<TLASBuilder> defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                      std::shared_ptr<TLAS> targetTLAS);
    std::shared_ptr<TLASBuilder> querySizeInfo(std::shared_ptr<Device> device);
    const VkAccelerationStructureBuildSizesInfoKHR &sizeInfo() const;
    VkDeviceSize requiredStorageSize() const;
    VkDeviceSize requiredScratchSize() const;
    std::shared_ptr<TLASBuilder>
    useExternalStorage(std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset = 0, VkDeviceSize capacity = 0);
    std::shared_ptr<TLASBuilder> useExternalScratch(std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset = 0);
    std::shared_ptr<TLASBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                 std::shared_ptr<Device> device,
                                                 std::shared_ptr<VMA> vma);
    std::shared_ptr<TLAS> buildAndSubmit(std::shared_ptr<Device> device, std::shared_ptr<CommandBuffer> commandBuffer);

  private:
    TLASInstanceBuilder tlasInstanceBuilder_;

    VkBuildAccelerationStructureModeKHR mode_;
    VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo_{};

    std::shared_ptr<DeviceLocalBuffer> tlasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;
    std::shared_ptr<TLAS> builtTLAS_;
    std::shared_ptr<TLAS> updateTargetTLAS_;
    VkDeviceSize tlasOffset_ = 0;
    VkDeviceSize tlasCapacity_ = 0;
    VkDeviceSize scratchOffset_ = 0;

    VkAccelerationStructureKHR srcTLAS_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR dstTLAS_ = VK_NULL_HANDLE;
};
};

template <typename T>
vk::BLASBuilder::BLASGeometryBuilder &
vk::BLASBuilder::BLASGeometryBuilder::defineTriangleGeomrtry(std::shared_ptr<DeviceLocalBuffer> vertexBuffer,
                                                             uint32_t numVertices,
                                                             std::shared_ptr<DeviceLocalBuffer> indexBuffer,
                                                             uint32_t numIndices,
                                                             bool isOpaque) {
    return defineTriangleGeomrtry<T>(vertexBuffer->bufferAddress(), numVertices, indexBuffer->bufferAddress(),
                                     numIndices, isOpaque);
}

template <typename T>
vk::BLASBuilder::BLASGeometryBuilder &
vk::BLASBuilder::BLASGeometryBuilder::defineTriangleGeomrtry(VkDeviceAddress vertexBufferAddress,
                                                             uint32_t numVertices,
                                                             VkDeviceAddress indexBufferAddress,
                                                             uint32_t numIndices,
                                                             bool isOpaque) {
    if (numVertices == 0 || numIndices < 3) { return definePlaceholderGeometry(); }

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

    auto &triangles = geom.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBufferAddress;
    triangles.vertexStride = sizeof(T);
    triangles.maxVertex = numVertices - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBufferAddress;

    geometries.push_back(geom);
    primitiveCounts.push_back(numIndices / 3);

    return *this;
}
