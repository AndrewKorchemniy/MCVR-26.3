#include "core/vulkan/command.hpp"
#include "core/profiler/profiler.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/descriptor.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/framebuffer.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/pipeline.hpp"
#include "core/vulkan/render_pass.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/sbt.hpp"
#include "core/vulkan/sync.hpp"

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

std::ostream &commandPoolCerr() {
    return std::cerr << "[CommandPool] ";
}

std::ostream &commandBufferCerr() {
    return std::cerr << "[CommandBuffer] ";
}

vk::CommandPool::CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device)
    : CommandPool(physicalDevice, device, physicalDevice->mainQueueIndex()) {}

vk::CommandPool::CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice,
                             std::shared_ptr<Device> device,
                             uint32_t queueIndex)
    : device_(device) {
    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = queueIndex;

    if (vkCreateCommandPool(device_->vkDevice(), &poolCreateInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        commandPoolCerr() << "failed to create command pool" << std::endl;
        exit(EXIT_FAILURE);
    }
}

vk::CommandPool::~CommandPool() {
    vkDestroyCommandPool(device_->vkDevice(), commandPool_, nullptr);
}

VkCommandPool &vk::CommandPool::vkCommandPool() {
    return commandPool_;
}

vk::CommandBuffer::CommandBuffer(std::shared_ptr<Device> device, std::shared_ptr<CommandPool> commandPool_)
    : commandPool_(commandPool_), device_(device) {
    recordedResources_.reserve(8);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_->vkCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device_->vkDevice(), &allocInfo, &commandBuffer_) != VK_SUCCESS) {
        commandBufferCerr() << "failed to allocate command buffer" << std::endl;
        exit(EXIT_FAILURE);
    }
}

vk::CommandBuffer::~CommandBuffer() {
    if (segments_.empty()) {
        if (commandBuffer_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_->vkDevice(), commandPool_->vkCommandPool(), 1, &commandBuffer_);
        }
    } else {
        vkFreeCommandBuffers(device_->vkDevice(), commandPool_->vkCommandPool(),
                             static_cast<uint32_t>(segments_.size()), segments_.data());
    }
}

VkCommandBuffer &vk::CommandBuffer::vkCommandBuffer() {
    return commandBuffer_;
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::begin(VkCommandBufferUsageFlags flags) {
    if (segments_.empty()) { segments_.push_back(commandBuffer_); }
    commandBuffer_ = segments_.front();
    segmentCount_ = 1;
    usageFlags_ = flags;
    insideRenderPass_ = false;
    VkCommandBufferBeginInfo bufferBeginInfo = {};
    bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bufferBeginInfo.flags = flags;

    vk::requireSuccess(vkBeginCommandBuffer(commandBuffer_, &bufferBeginInfo), "vkBeginCommandBuffer");
    recordedResources_.clear();
    return shared_from_this();
}

size_t vk::CommandBuffer::split() {
    if (segmentCount_ == 0 || insideRenderPass_) { throw std::runtime_error("invalid command buffer split"); }
    end();
    const size_t completedSegment = segmentCount_ - 1;
    if (segmentCount_ == segments_.size()) {
        VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocateInfo.commandPool = commandPool_->vkCommandPool();
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        VkCommandBuffer segment = VK_NULL_HANDLE;
        vk::requireSuccess(vkAllocateCommandBuffers(device_->vkDevice(), &allocateInfo, &segment),
                           "vkAllocateCommandBuffers(world segment)");
        segments_.push_back(segment);
    }
    commandBuffer_ = segments_[segmentCount_++];
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = usageFlags_;
    vk::requireSuccess(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(world segment)");
    return completedSegment;
}

std::span<const VkCommandBuffer> vk::CommandBuffer::recordedSegments() const {
    return {segments_.data(), segmentCount_};
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::copyToDeviceLocalBuffer(std::shared_ptr<DeviceLocalBuffer> deviceLocalBuffer) {
    retainResource(deviceLocalBuffer);
    deviceLocalBuffer->uploadToBuffer(commandBuffer_);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::copyToDeviceLocalImage(std::shared_ptr<DeviceLocalImage> deviceLocalImage) {
    retainResource(deviceLocalImage);
    deviceLocalImage->uploadToImage(commandBuffer_);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::barriersBufferImage(std::vector<BufferMemoryBarrier> bufferBarriers,
                                       std::vector<ImageMemoryBarrier> imageBarriers) {
    std::vector<VkBufferMemoryBarrier2> vkBufferBarriers;
    for (int i = 0; i < bufferBarriers.size(); i++) {
        retainResource(bufferBarriers[i].buffer);
        VkBufferMemoryBarrier2 vkBufferBarrier{};
        vkBufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vkBufferBarrier.srcStageMask = bufferBarriers[i].srcStageMask;
        vkBufferBarrier.srcAccessMask = bufferBarriers[i].srcAccessMask;
        vkBufferBarrier.dstStageMask = bufferBarriers[i].dstStageMask;
        vkBufferBarrier.dstAccessMask = bufferBarriers[i].dstAccessMask;
        vkBufferBarrier.srcQueueFamilyIndex = bufferBarriers[i].srcQueueFamilyIndex;
        vkBufferBarrier.dstQueueFamilyIndex = bufferBarriers[i].dstQueueFamilyIndex;
        vkBufferBarrier.buffer = bufferBarriers[i].buffer->vkBuffer();
        vkBufferBarrier.offset = bufferBarriers[i].offset;
        vkBufferBarrier.size = bufferBarriers[i].size;

        vkBufferBarriers.push_back(vkBufferBarrier);
    }

    std::vector<VkImageMemoryBarrier2> vkImageBarriers;
    for (int i = 0; i < imageBarriers.size(); i++) {
        retainResource(imageBarriers[i].image);
        VkImageMemoryBarrier2 vkImageBarrier{};
        vkImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        vkImageBarrier.srcStageMask = imageBarriers[i].srcStageMask;
        vkImageBarrier.srcAccessMask = imageBarriers[i].srcAccessMask;
        vkImageBarrier.dstStageMask = imageBarriers[i].dstStageMask;
        vkImageBarrier.dstAccessMask = imageBarriers[i].dstAccessMask;
        vkImageBarrier.oldLayout = imageBarriers[i].oldLayout;
        vkImageBarrier.newLayout = imageBarriers[i].newLayout;
        vkImageBarrier.srcQueueFamilyIndex = imageBarriers[i].srcQueueFamilyIndex;
        vkImageBarrier.dstQueueFamilyIndex = imageBarriers[i].dstQueueFamilyIndex;
        vkImageBarrier.image = imageBarriers[i].image->vkImage();
        vkImageBarrier.subresourceRange = imageBarriers[i].subresourceRange;

        vkImageBarriers.push_back(vkImageBarrier);
    }

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = vkBufferBarriers.size();
    dependencyInfo.pBufferMemoryBarriers = vkBufferBarriers.data();
    dependencyInfo.imageMemoryBarrierCount = vkImageBarriers.size();
    dependencyInfo.pImageMemoryBarriers = vkImageBarriers.data();
    dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    vkCmdPipelineBarrier2(commandBuffer_, &dependencyInfo);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::barriersMemory(std::vector<MemoryBarrier> memoryBarriers) {
    std::vector<VkMemoryBarrier2> vkMemoryBarriers;
    for (int i = 0; i < memoryBarriers.size(); i++) {
        VkMemoryBarrier2 vkMemoryBarrier{};
        vkMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        vkMemoryBarrier.srcStageMask = memoryBarriers[i].srcStageMask;
        vkMemoryBarrier.srcAccessMask = memoryBarriers[i].srcAccessMask;
        vkMemoryBarrier.dstStageMask = memoryBarriers[i].dstStageMask;
        vkMemoryBarrier.dstAccessMask = memoryBarriers[i].dstAccessMask;

        vkMemoryBarriers.push_back(vkMemoryBarrier);
    }

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = vkMemoryBarriers.size();
    dependencyInfo.pMemoryBarriers = vkMemoryBarriers.data();

    vkCmdPipelineBarrier2(commandBuffer_, &dependencyInfo);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::beginRenderPass(RenderPassBeginInfo renderPassBeginInfo) {
    retainResource(renderPassBeginInfo.renderPass);
    retainResource(renderPassBeginInfo.framebuffer);
    if (renderPassBeginInfo.framebuffer != nullptr) {
        for (const auto &attachment : renderPassBeginInfo.framebuffer->attachments()) { retainResource(attachment); }
    }
    VkRenderPassBeginInfo vkRenderPassBeginInfo{};
    vkRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    vkRenderPassBeginInfo.renderPass = renderPassBeginInfo.renderPass->vkRenderPass();
    vkRenderPassBeginInfo.framebuffer = renderPassBeginInfo.framebuffer->vkFramebuffer();
    vkRenderPassBeginInfo.renderArea.offset = renderPassBeginInfo.renderAreaOffset;
    vkRenderPassBeginInfo.renderArea.extent = renderPassBeginInfo.renderAreaExtent;
    vkRenderPassBeginInfo.clearValueCount = renderPassBeginInfo.clearValues.size();
    vkRenderPassBeginInfo.pClearValues = renderPassBeginInfo.clearValues.data();
    vkCmdBeginRenderPass(commandBuffer_, &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    insideRenderPass_ = true;
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::endRenderPass() {
    vkCmdEndRenderPass(commandBuffer_);
    insideRenderPass_ = false;
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::bindDescriptorTable(std::shared_ptr<DescriptorTable> descriptorTable,
                                       VkPipelineBindPoint bindPoint) {
    retainDescriptorTableResources(descriptorTable);
    uint32_t dynamicCount = descriptorTable->dynamicDescriptorCount();
    std::vector<uint32_t> dynamicOffsets(dynamicCount, 0);
    const auto &sets = descriptorTable->descriptorSet();
    const uint32_t setCount = static_cast<uint32_t>(sets.size());

    uint32_t rangeStart = setCount;
    for (uint32_t i = 0; i < setCount; i++) {
        if (!descriptorTable->isPushDescriptorSet(i)) {
            rangeStart = i;
            break;
        }
    }

    uint32_t i = rangeStart;
    while (i < setCount) {
        if (descriptorTable->isPushDescriptorSet(i)) {
            i++;
            continue;
        }
        uint32_t rangeEnd = i;
        while (rangeEnd < setCount && !descriptorTable->isPushDescriptorSet(rangeEnd)) { rangeEnd++; }
        vkCmdBindDescriptorSets(commandBuffer_, bindPoint, descriptorTable->vkPipelineLayout(), i, rangeEnd - i,
                                sets.data() + i, dynamicCount,
                                dynamicOffsets.empty() ? nullptr : dynamicOffsets.data());
        i = rangeEnd;
    }
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::pushDescriptorBuffers(std::shared_ptr<DescriptorTable> descriptorTable,
                                         VkPipelineBindPoint bindPoint,
                                         uint32_t set,
                                         std::vector<PushDescriptorBuffer> buffers) {
    if (!descriptorTable) { throw std::runtime_error("CommandBuffer::pushDescriptorBuffers received null table"); }
    retainDescriptorTableResources(descriptorTable);
    if (!descriptorTable->isPushDescriptorSet(set) || descriptorTable->pushDescriptorSetLayout(set) == VK_NULL_HANDLE) {
        std::ostringstream error;
        error << "CommandBuffer::pushDescriptorBuffers set " << set << " is not a valid push descriptor set";
        throw std::runtime_error(error.str());
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(buffers.size());
    for (const auto &buf : buffers) {
        if (buf.buffer == VK_NULL_HANDLE) { continue; }
        retainResource(buf.owner);
        VkDescriptorType type = descriptorTable->descriptorType(set, buf.binding, "pushDescriptorBuffers");
        if (type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            std::ostringstream error;
            error << "CommandBuffer::pushDescriptorBuffers binding " << buf.binding << " in set " << set
                  << " is not a storage buffer descriptor";
            throw std::runtime_error(error.str());
        }
        bufferInfos.push_back({.buffer = buf.buffer, .offset = 0, .range = buf.size});
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(bufferInfos.size());
    uint32_t infoIndex = 0;
    for (const auto &buf : buffers) {
        if (buf.buffer == VK_NULL_HANDLE) { continue; }
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = VK_NULL_HANDLE;
        write.dstBinding = buf.binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos[infoIndex++];
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkCmdPushDescriptorSetKHR(commandBuffer_, bindPoint, descriptorTable->vkPipelineLayout(), set,
                                  static_cast<uint32_t>(writes.size()), writes.data());
    }
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindGraphicsPipeline(std::shared_ptr<GraphicsPipeline> pipeline) {
    retainResource(pipeline);
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindRTPipeline(std::shared_ptr<RayTracingPipeline> pipeline) {
    retainResource(pipeline);
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindComputePipeline(std::shared_ptr<ComputePipeline> pipeline) {
    retainResource(pipeline);
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindVertexBuffers(std::shared_ptr<DeviceLocalBuffer> buffer) {
    retainResource(buffer);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &buffer->vkBuffer(), &offset);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer) {
    retainResource(buffer);
    vkCmdBindIndexBuffer(commandBuffer_, buffer->vkBuffer(), 0, VK_INDEX_TYPE_UINT32);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                      VkIndexType indexType) {
    retainResource(buffer);
    vkCmdBindIndexBuffer(commandBuffer_, buffer->vkBuffer(), 0, indexType);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t firstInstance) {
    vkCmdDraw(commandBuffer_, vertexCount, instanceCount, firstIndex, firstInstance);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::drawIndexed(
    uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    vkCmdDrawIndexed(commandBuffer_, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::raytracing(std::shared_ptr<SBT> sbt, uint32_t width, uint32_t height, uint32_t depth) {
    retainResource(sbt);
    vkCmdTraceRaysKHR(commandBuffer_, &sbt->raygenRegion(), &sbt->missRegion(), &sbt->hitRegion(),
                      &sbt->callableRegion(), width, height, depth);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::raytracingIndirect(std::shared_ptr<SBT> sbt,
                                                                         VkDeviceAddress indirectAddress) {
    retainResource(sbt);
    vkCmdTraceRaysIndirectKHR(commandBuffer_, &sbt->raygenRegion(), &sbt->missRegion(), &sbt->hitRegion(),
                              &sbt->callableRegion(), indirectAddress);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::beginDebugLabel(const char *name, float r, float g, float b, float a) {
    if (vkCmdBeginDebugUtilsLabelEXT == nullptr) { return shared_from_this(); }

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    vkCmdBeginDebugUtilsLabelEXT(commandBuffer_, &label);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::endDebugLabel() {
    if (vkCmdEndDebugUtilsLabelEXT != nullptr) { vkCmdEndDebugUtilsLabelEXT(commandBuffer_); }
    return shared_from_this();
}

vk::CommandBuffer::ScopedDebugLabel::ScopedDebugLabel(
    CommandBuffer &commandBuffer, const char *name, float r, float g, float b, float a)
    : commandBuffer_(&commandBuffer),
      active_(vkCmdBeginDebugUtilsLabelEXT != nullptr && vkCmdEndDebugUtilsLabelEXT != nullptr) {
    if (active_) { commandBuffer_->beginDebugLabel(name, r, g, b, a); }
}

vk::CommandBuffer::ScopedDebugLabel::~ScopedDebugLabel() {
    if (active_) { commandBuffer_->endDebugLabel(); }
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::end() {
    vk::requireSuccess(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::reset(VkCommandBufferResetFlags flags) {
    if (segments_.empty()) { segments_.push_back(commandBuffer_); }
    for (VkCommandBuffer segment : segments_) {
        vk::requireSuccess(vkResetCommandBuffer(segment, flags), "vkResetCommandBuffer");
    }
    commandBuffer_ = segments_.front();
    segmentCount_ = 0;
    insideRenderPass_ = false;
    recordedResources_.clear();
    return shared_from_this();
}

void vk::CommandBuffer::retainDescriptorTableResources(const std::shared_ptr<DescriptorTable> &descriptorTable) {
    if (descriptorTable == nullptr) { return; }
    retainResource(descriptorTable);
    for (const auto &resource : descriptorTable->boundResources()) {
        if (resource != nullptr) { recordedResources_.emplace_back(resource); }
    }
}

void vk::CommandBuffer::submitMainQueueIndividual(std::shared_ptr<Device> device) {
    submitMainQueueIndividual(device, nullptr);
}

void vk::CommandBuffer::submitMainQueueIndividual(std::shared_ptr<vk::Device> device,
                                                  std::shared_ptr<vk::Fence> fence) {
    MCVR_PROFILE_SCOPE("native.CommandBuffer.submitMainQueueIndividual");
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = static_cast<uint32_t>(segmentCount_);
    submitInfo.pCommandBuffers = segments_.data();

    std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex(), std::defer_lock);
    {
        MCVR_PROFILE_SCOPE("native.CommandBuffer.waitMainQueue");
        queueLock.lock();
    }
    {
        MCVR_PROFILE_SCOPE("native.CommandBuffer.submitIndividual");
        vk::requireSuccess(
            vkQueueSubmit(device->mainVkQueue(), 1, &submitInfo, fence == nullptr ? VK_NULL_HANDLE : fence->vkFence()),
            "vkQueueSubmit(main individual)");
    }
}

void vk::CommandBuffer::submitMainQueue(std::shared_ptr<Device> device, SubmitInfo submitInfo) {
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStageMasks;
    for (auto [semaphore, mask] : submitInfo.waitSemaphoresAndStageMasks) {
        waitSemaphores.push_back(semaphore);
        waitStageMasks.push_back(mask);
    }

    std::vector<VkSemaphore> signalSemaphores = submitInfo.signalSemaphores;

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = waitSemaphores.size();
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStageMasks.data();
    vkSubmitInfo.commandBufferCount = static_cast<uint32_t>(segmentCount_);
    vkSubmitInfo.pCommandBuffers = segments_.data();
    vkSubmitInfo.signalSemaphoreCount = signalSemaphores.size();
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.data();

    std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
    vk::requireSuccess(vkQueueSubmit(device->mainVkQueue(), 1, &vkSubmitInfo, submitInfo.signalFence),
                       "vkQueueSubmit(main)");
}
