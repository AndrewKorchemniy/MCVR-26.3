/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * MODIFICATIONS and INTEGRATION:
 *
 * Copyright (c) 2026 Radiance
 *
 * This file has been modified from its original version to be integrated into
 * Radiance Mod.
 *
 * Modifications include:
 * - Integration with Radiance Mod's vulkan rendering system.
 * - Refactoring of APIs, introducing RAII and supporting shared_ptr.
 *
 * These modifications are licensed under the GNU General Public License
 * as published by the Free Software Foundation; either version 3 of the License,
 * or (at your option) any later version.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/render/modules/world/nrd/nrd_wrapper.hpp"
#include "core/vulkan/result.hpp"

#include <NRDDescs.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <vector>

void NrdWrapper::callNrd(nrd::Result result) {
    assert(result == nrd::Result::SUCCESS);
}

VkFormat NrdWrapper::toVkFormat(nrd::Format nrdFormat) {
    static constexpr std::array formats{
        VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8_SNORM,
        VK_FORMAT_R8_UINT,
        VK_FORMAT_R8_SINT,

        VK_FORMAT_R8G8_UNORM,
        VK_FORMAT_R8G8_SNORM,
        VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R8G8_SINT,

        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_SNORM,
        VK_FORMAT_A8B8G8R8_UINT_PACK32,
        VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R8G8B8A8_SRGB,

        VK_FORMAT_R16_UNORM,
        VK_FORMAT_R16_SNORM,
        VK_FORMAT_R16_UINT,
        VK_FORMAT_R16_SINT,
        VK_FORMAT_R16_SFLOAT,

        VK_FORMAT_R16G16_UNORM,
        VK_FORMAT_R16G16_SNORM,
        VK_FORMAT_R16G16_UINT,
        VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R16G16_SFLOAT,

        VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM,
        VK_FORMAT_R16G16B16A16_UINT,
        VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R16G16B16A16_SFLOAT,

        VK_FORMAT_R32_UINT,
        VK_FORMAT_R32_SINT,
        VK_FORMAT_R32_SFLOAT,

        VK_FORMAT_R32G32_UINT,
        VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R32G32_SFLOAT,

        VK_FORMAT_R32G32B32_UINT,
        VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R32G32B32_SFLOAT,

        VK_FORMAT_R32G32B32A32_UINT,
        VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_R32G32B32A32_SFLOAT,

        VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_FORMAT_A2R10G10B10_UINT_PACK32,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
    };
    static_assert(formats.size() == size_t(nrd::Format::MAX_NUM));

    assert(size_t(nrdFormat) < formats.size());
    return formats[size_t(nrdFormat)];
}

VkDescriptorType NrdWrapper::toVkDescriptorType(nrd::DescriptorType type) {
    static constexpr std::array descriptorTypes{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    static_assert(descriptorTypes.size() == size_t(nrd::DescriptorType::MAX_NUM));

    assert(size_t(type) < descriptorTypes.size());
    return descriptorTypes[size_t(type)];
}

VkFilter NrdWrapper::toVkFilter(nrd::Sampler sampler) {
    static constexpr std::array filters{VK_FILTER_NEAREST, VK_FILTER_LINEAR};
    static_assert(filters.size() == size_t(nrd::Sampler::MAX_NUM));

    assert(size_t(sampler) < filters.size());
    return filters[size_t(sampler)];
}

uint16_t NrdWrapper::divideRoundUp(uint32_t dividend, uint16_t divisor) {
    return uint16_t((dividend + divisor - 1) / divisor);
}

const char *NrdWrapper::vkResultName(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "VK_ERROR_UNRECOGNIZED";
    }
}

void NrdWrapper::throwVkError(VkResult result, const char *operation) {
    if (result == VK_SUCCESS) return;

    std::ostringstream error;
    error << "NRD: " << operation << " failed (" << vkResultName(result) << ", " << static_cast<int32_t>(result) << ")";
    throw std::runtime_error(error.str());
}

const char *NrdWrapper::vkDescriptorTypeName(VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "SAMPLED_IMAGE";
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "STORAGE_IMAGE";
        case VK_DESCRIPTOR_TYPE_SAMPLER: return "SAMPLER";
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UNIFORM_BUFFER";
        default: return "OTHER";
    }
}

const VkDescriptorSetLayoutBinding *NrdWrapper::findBinding(const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                                                            uint32_t binding) {
    auto it = std::find_if(bindings.begin(), bindings.end(),
                           [binding](const VkDescriptorSetLayoutBinding &b) { return b.binding == binding; });
    return it == bindings.end() ? nullptr : &*it;
}

void NrdWrapper::addUniqueBinding(std::vector<VkDescriptorSetLayoutBinding> &bindings,
                                  const VkDescriptorSetLayoutBinding &binding,
                                  const char *owner) {
    if (const VkDescriptorSetLayoutBinding *existing = findBinding(bindings, binding.binding)) {
        if (existing->descriptorType == binding.descriptorType &&
            existing->descriptorCount == binding.descriptorCount) {
            return;
        }

        std::ostringstream error;
        error << "NRD: duplicate descriptor binding " << binding.binding << " in " << owner << " ("
              << vkDescriptorTypeName(existing->descriptorType) << " vs "
              << vkDescriptorTypeName(binding.descriptorType) << ")";
        throw std::runtime_error(error.str());
    }

    bindings.push_back(binding);
}

void NrdWrapper::transitionAndClearTexture(const std::shared_ptr<Framework> &framework,
                                           const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                           const std::shared_ptr<vk::DeviceLocalImage> &image,
                                           VkImageLayout layout) {
    commandBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = image->imageLayout(),
                .newLayout = layout,
                .srcQueueFamilyIndex = framework->physicalDevice()->mainQueueIndex(),
                .dstQueueFamilyIndex = framework->physicalDevice()->mainQueueIndex(),
                .image = image,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});
    image->imageLayout() = layout;

    VkClearColorValue clearValue = {{0.F, 0.F, 0.F, 0.F}};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(commandBuffer->vkCommandBuffer(), image->vkImage(), layout, &clearValue, 1, &range);
}

void NrdWrapper::addShaderImageBarrier(std::vector<VkImageMemoryBarrier> &imageBarriers,
                                       const std::shared_ptr<vk::DeviceLocalImage> &image,
                                       uint32_t queueIndex,
                                       VkAccessFlags srcAccess,
                                       VkAccessFlags dstAccess) {
    auto existingBarrier =
        std::find_if(imageBarriers.begin(), imageBarriers.end(),
                     [&](const VkImageMemoryBarrier &barrier) { return barrier.image == image->vkImage(); });
    if (existingBarrier != imageBarriers.end()) {
        existingBarrier->srcAccessMask |= srcAccess;
        existingBarrier->dstAccessMask |= dstAccess;
        return;
    }

    imageBarriers.push_back(VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .oldLayout = image->imageLayout(),
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = queueIndex,
        .dstQueueFamilyIndex = queueIndex,
        .image = image->vkImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
    });
    image->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
}

void NrdWrapper::init(std::shared_ptr<Framework> framework,
                      uint16_t width,
                      uint16_t height,
                      bool enableSh,
                      bool enableSeparateDirectDenoising,
                      bool enableIndirectDenoising,
                      bool splitIndirectLobes) {
    framework_ = framework;
    isShEnabled_ = enableSh;
    isSeparateDirectDenoisingEnabled_ = enableSeparateDirectDenoising;
    isIndirectDenoisingEnabled_ = enableIndirectDenoising;
    splitIndirectLobes_ = splitIndirectLobes;

    std::vector<nrd::DenoiserDesc> denoisers;
    if (isIndirectDenoisingEnabled_) {
        if (splitIndirectLobes_) {
            const nrd::Denoiser diffuse =
                isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SH : nrd::Denoiser::REBLUR_DIFFUSE;
            const nrd::Denoiser specular =
                isShEnabled_ ? nrd::Denoiser::REBLUR_SPECULAR_SH : nrd::Denoiser::REBLUR_SPECULAR;
            denoisers.push_back({nrd::Identifier(diffuse), diffuse});
            denoisers.push_back({nrd::Identifier(specular), specular});
        } else {
            const nrd::Denoiser reblur =
                isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH : nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;
            denoisers.push_back({nrd::Identifier(reblur), reblur});
        }
    }
    if (isSeparateDirectDenoisingEnabled_) {
        denoisers.push_back(
            {nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR});
    }

    if (denoisers.empty()) { throw std::runtime_error("NRD: no denoisers requested"); }
    nrd::InstanceCreationDesc instanceDesc{{}, denoisers.data(), uint32_t(denoisers.size())};

    callNrd(CreateInstance(instanceDesc, m_instance));
    for (const auto &denoiser : denoisers) { m_denoiserIdentifiers.push_back(denoiser.identifier); }

    const nrd::InstanceDesc *iDesc = GetInstanceDesc(*m_instance);

    for (uint32_t t = 0; t < iDesc->permanentPoolSize; ++t) {
        std::shared_ptr<vk::DeviceLocalImage> nrdTexture = createTexture(iDesc->permanentPool[t], width, height);
        m_permanentTextures.push_back(nrdTexture);
    }

    for (uint32_t t = 0; t < iDesc->transientPoolSize; ++t) {
        std::shared_ptr<vk::DeviceLocalImage> nrdTexture = createTexture(iDesc->transientPool[t], width, height);
        m_transientTextures.push_back(nrdTexture);
    }

    {
        const VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

        std::shared_ptr<vk::Fence> fence = vk::Fence::create(framework->device());
        std::shared_ptr<vk::CommandBuffer> oneTimeBuffer =
            vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());

        oneTimeBuffer->begin();

        for (auto &t : m_transientTextures) { transitionAndClearTexture(framework, oneTimeBuffer, t, layout); }
        for (auto &t : m_permanentTextures) { transitionAndClearTexture(framework, oneTimeBuffer, t, layout); }

        oneTimeBuffer->end();

        VkSubmitInfo vkSubmitInfo = {};
        vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        vkSubmitInfo.waitSemaphoreCount = 0;
        vkSubmitInfo.pWaitSemaphores = nullptr;
        vkSubmitInfo.pWaitDstStageMask = nullptr;
        vkSubmitInfo.commandBufferCount = 1;
        vkSubmitInfo.pCommandBuffers = &oneTimeBuffer->vkCommandBuffer();
        vkSubmitInfo.signalSemaphoreCount = 0;
        vkSubmitInfo.pSignalSemaphores = nullptr;
        {
            std::unique_lock<std::recursive_mutex> queueLock(framework->device()->mainQueueMutex());
            vk::requireSuccess(vkQueueSubmit(framework->device()->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence()),
                               "vkQueueSubmit(NRD texture initialization)");
        }
        vk::waitForFence(framework->device()->vkDevice(), fence->vkFence(),
                         "vkWaitForFences(NRD texture initialization)");
    }

    for (uint32_t s = 0; s < iDesc->samplersNum; ++s) {
        auto filter = toVkFilter(iDesc->samplers[s]);

        std::shared_ptr<vk::Sampler> sampler = vk::Sampler::create(
            framework->device(), filter, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        m_vkSamplers.push_back(sampler->vkSamper());
        m_samplers.push_back(sampler);
    }

    const auto limits = framework->physicalDevice()->properties().limits;
    const VkDeviceSize alignment = std::max(limits.minUniformBufferOffsetAlignment, limits.nonCoherentAtomSize);
    const VkDeviceSize stride = (iDesc->constantBufferMaxDataSize + alignment - 1) / alignment * alignment;
    m_resourceDescriptorSetsPerFrame = std::max(1U, iDesc->descriptorPoolDesc.setsMaxNum);
    const VkDeviceSize constantBufferSize =
        stride * m_resourceDescriptorSetsPerFrame * framework->swapchain()->imageCount();
    if (stride == 0 || iDesc->constantBufferMaxDataSize > limits.maxUniformBufferRange ||
        constantBufferSize > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("NRD: constant buffer arena exceeds Vulkan limits");
    }
    m_constantBufferStride = static_cast<uint32_t>(stride);
    m_constantBuffer = vk::HostVisibleBuffer::create(framework->vma(), framework->device(), constantBufferSize,
                                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    createPipelines();
}

NrdWrapper::~NrdWrapper() {
    m_permanentTextures.clear();
    m_transientTextures.clear();
    m_vkSamplers.clear();
    m_samplers.clear();
    m_constantBuffer = nullptr;

    auto device = framework_.lock()->device();
    for (auto &framePool : m_frameDescriptorPools) {
        if (framePool.resourceDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device->vkDevice(), framePool.resourceDescriptorPool, nullptr);
            framePool.resourceDescriptorPool = VK_NULL_HANDLE;
        }
    }
    m_frameDescriptorPools.clear();

    if (m_samplerConstBufferDescriptorLayout) {
        vkDestroyDescriptorSetLayout(device->vkDevice(), m_samplerConstBufferDescriptorLayout, nullptr);
        vkDestroyDescriptorPool(device->vkDevice(), m_samplerConstBufferDescriptorPool, nullptr);
    }
    for (auto &p : m_pipelines) {
        vkDestroyPipeline(device->vkDevice(), p.pipeline, nullptr);
        vkDestroyPipelineLayout(device->vkDevice(), p.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device->vkDevice(), p.resourceDescriptorLayout, nullptr);
    }
    m_pipelines.clear();

    nrd::DestroyInstance(*m_instance);
}

void NrdWrapper::setUserPoolTexture(
    std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>
        userTexturePool) {
    m_userTexturePool = userTexturePool;
}

VkFormat NrdWrapper::getNormalRoughnessFormat() {
    switch (nrd::GetLibraryDesc()->normalEncoding) {
        case nrd::NormalEncoding::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case nrd::NormalEncoding::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
        case nrd::NormalEncoding::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case nrd::NormalEncoding::RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
        case nrd::NormalEncoding::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
        default: assert(0 && "Unknown normal encoding");
    }
    return VK_FORMAT_UNDEFINED;
}

void NrdWrapper::setCommonSettings(nrd::CommonSettings &settings) {
    auto framework = framework_.lock();
    if (!framework) { throw std::runtime_error("NRD: framework expired while setting common settings"); }
    auto context = framework->safeAcquireCurrentContext();
    resetResourceDescriptorPool(context->frameIndex);
    callNrd(nrd::SetCommonSettings(*m_instance, settings));
}

void NrdWrapper::setDenoiserSettings(nrd::Identifier identifier, const void *settings) {
    callNrd(nrd::SetDenoiserSettings(*m_instance, identifier, settings));
}

void NrdWrapper::setReblurSettings(const nrd::ReblurSettings &settings) {
    if (!isIndirectDenoisingEnabled_) return;

    if (!splitIndirectLobes_) {
        setDenoiserSettings(nrd::Identifier(isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH :
                                                           nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR),
                            &settings);
        return;
    }

    nrd::ReblurSettings diffuseSettings = settings;
    nrd::ReblurSettings specularSettings = settings;

    specularSettings.checkerboardMode = settings.checkerboardMode;

    setDenoiserSettings(
        nrd::Identifier(isShEnabled_ ? nrd::Denoiser::REBLUR_DIFFUSE_SH : nrd::Denoiser::REBLUR_DIFFUSE),
        &diffuseSettings);
    setDenoiserSettings(
        nrd::Identifier(isShEnabled_ ? nrd::Denoiser::REBLUR_SPECULAR_SH : nrd::Denoiser::REBLUR_SPECULAR),
        &specularSettings);
}

void NrdWrapper::setRelaxSettings(const nrd::RelaxSettings &settings) {
    setDenoiserSettings(nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR), &settings);
}

std::shared_ptr<vk::DeviceLocalImage>
NrdWrapper::createTexture(const nrd::TextureDesc &tDesc, uint16_t width, uint16_t height) {
    uint16_t texWidth = divideRoundUp(width, tDesc.downsampleFactor);
    uint16_t texHeight = divideRoundUp(height, tDesc.downsampleFactor);

    auto framework = framework_.lock();
    return vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, texWidth, texHeight, 1,
                                        toVkFormat(tDesc.format),
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
}

void NrdWrapper::createPipelines() {
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_instance);
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();

    const uint32_t constantBufferBinding =
        lDesc->spirvBindingOffsets.constantBufferOffset + iDesc->constantBufferRegisterIndex;
    const uint32_t samplersBindingBase = lDesc->spirvBindingOffsets.samplerOffset + iDesc->samplersBaseRegisterIndex;
    const uint32_t texturesBindingBase = lDesc->spirvBindingOffsets.textureOffset + iDesc->resourcesBaseRegisterIndex;
    const uint32_t storageTextureBindingBase =
        lDesc->spirvBindingOffsets.storageTextureAndBufferOffset + iDesc->resourcesBaseRegisterIndex;

    bool samplersInSeparateSet = iDesc->constantBufferAndSamplersSpaceIndex != iDesc->resourcesSpaceIndex;
    assert(samplersInSeparateSet);

    auto device = framework_.lock()->device();
    {
        std::vector<VkDescriptorSetLayoutBinding> setBindings;
        setBindings.reserve(iDesc->samplersNum + 1);
        for (uint32_t s = 0; s < iDesc->samplersNum; ++s) {
            addUniqueBinding(setBindings,
                             VkDescriptorSetLayoutBinding{
                                 .binding = samplersBindingBase + s,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                 .descriptorCount = 1,
                                 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                 .pImmutableSamplers = &m_vkSamplers[s],
                             },
                             "sampler/constant set");
        }
        addUniqueBinding(setBindings,
                         VkDescriptorSetLayoutBinding{
                             .binding = constantBufferBinding,
                             .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                             .descriptorCount = 1,
                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                         },
                         "sampler/constant set");

        const VkDescriptorSetLayoutCreateInfo samplerLayoutInfo{.sType =
                                                                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                                .flags = 0,
                                                                .bindingCount = (uint32_t)setBindings.size(),
                                                                .pBindings = setBindings.data()};

        throwVkError(vkCreateDescriptorSetLayout(device->vkDevice(), &samplerLayoutInfo, nullptr,
                                                 &m_samplerConstBufferDescriptorLayout),
                     "create sampler/constant descriptor set layout");

        const VkDescriptorPoolSize poolSizes[] = {
            {.type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = iDesc->samplersNum},
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1}};

        const VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                  .maxSets = 1,
                                                  .poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
                                                  .pPoolSizes = poolSizes};
        throwVkError(
            vkCreateDescriptorPool(device->vkDevice(), &poolInfo, nullptr, &m_samplerConstBufferDescriptorPool),
            "create sampler/constant descriptor pool");

        const VkDescriptorSetAllocateInfo descriptorSetInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                            .descriptorPool = m_samplerConstBufferDescriptorPool,
                                                            .descriptorSetCount = 1,
                                                            .pSetLayouts = &m_samplerConstBufferDescriptorLayout};

        throwVkError(
            vkAllocateDescriptorSets(device->vkDevice(), &descriptorSetInfo, &m_samplerConstBufferDescriptorSet),
            "allocate sampler/constant descriptor set");
    }

    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_constantBuffer->vkBuffer();
        bufferInfo.offset = 0;
        bufferInfo.range = iDesc->constantBufferMaxDataSize;

        VkWriteDescriptorSet constantBufferUpdate = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                     .pNext = nullptr,
                                                     .dstSet = m_samplerConstBufferDescriptorSet,
                                                     .dstBinding = constantBufferBinding,
                                                     .descriptorCount = 1,
                                                     .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                                     .pBufferInfo = &bufferInfo};
        vkUpdateDescriptorSets(device->vkDevice(), 1, &constantBufferUpdate, 0, nullptr);
    }

    m_pipelines.resize(iDesc->pipelinesNum);

#pragma omp parallel for
    for (int p = 0; p < static_cast<int>(iDesc->pipelinesNum); ++p) {
        const nrd::PipelineDesc &pDesc = iDesc->pipelines[p];
        std::vector<VkDescriptorSetLayoutBinding> setBindings;
        uint32_t pipelineResourceBindingCount = 0;
        for (uint32_t r = 0; r < pDesc.resourceRangesNum; r++) {
            pipelineResourceBindingCount += pDesc.resourceRanges[r].descriptorsNum;
        }
        setBindings.reserve(pipelineResourceBindingCount);
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = 0, .pBindings = nullptr};

        for (uint32_t r = 0; r < pDesc.resourceRangesNum; r++) {
            const nrd::ResourceRangeDesc &resourceRange = pDesc.resourceRanges[r];
            const bool isStorage = resourceRange.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
            const uint32_t bindingBase = isStorage ? storageTextureBindingBase : texturesBindingBase;
            const VkDescriptorType descriptorType =
                isStorage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

            for (uint32_t b = 0; b < resourceRange.descriptorsNum; ++b) {
                addUniqueBinding(setBindings,
                                 VkDescriptorSetLayoutBinding{
                                     .binding = bindingBase + b,
                                     .descriptorType = descriptorType,
                                     .descriptorCount = 1,
                                     .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                 },
                                 "NRD resource set");
            }
        }
        setLayoutInfo.bindingCount = static_cast<uint32_t>(setBindings.size());
        setLayoutInfo.pBindings = setBindings.data();

        NRDPipeline &nrdPipeline = m_pipelines[p];

        throwVkError(vkCreateDescriptorSetLayout(device->vkDevice(), &setLayoutInfo, nullptr,
                                                 &nrdPipeline.resourceDescriptorLayout),
                     "create resource descriptor set layout");

        nrdPipeline.resourceBindings = setBindings;
        nrdPipeline.numBindings = setLayoutInfo.bindingCount;

        assert(iDesc->constantBufferAndSamplersSpaceIndex == 1 && iDesc->resourcesSpaceIndex == 0);

        const std::array<VkDescriptorSetLayout, 2> pipelineSetLayouts{nrdPipeline.resourceDescriptorLayout,
                                                                      m_samplerConstBufferDescriptorLayout};
        const VkPipelineLayoutCreateInfo pipelineLayoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                            .setLayoutCount = 2U,
                                                            .pSetLayouts = pipelineSetLayouts.data()};

        throwVkError(
            vkCreatePipelineLayout(device->vkDevice(), &pipelineLayoutInfo, nullptr, &nrdPipeline.pipelineLayout),
            "create pipeline layout");

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shaderModuleInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = pDesc.computeShaderSPIRV.size,
            .pCode = reinterpret_cast<const uint32_t *>(pDesc.computeShaderSPIRV.bytecode),
        };
        throwVkError(vkCreateShaderModule(device->vkDevice(), &shaderModuleInfo, nullptr, &shaderModule),
                     "create compute shader module");

        const VkPipelineShaderStageCreateInfo stageCreateInfo{.sType =
                                                                  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                              .module = shaderModule,
                                                              .pName = iDesc->shaderEntryPoint};

        const VkComputePipelineCreateInfo pipelineCreateInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                             .stage = stageCreateInfo,
                                                             .layout = nrdPipeline.pipelineLayout};

        VkResult result = vkCreateComputePipelines(device->vkDevice(), device->vkPipelineCache(), 1, &pipelineCreateInfo, nullptr,
                                                   &nrdPipeline.pipeline);
        vkDestroyShaderModule(device->vkDevice(), shaderModule, nullptr);
        throwVkError(result, "create compute pipeline");
    }

    createResourceDescriptorPools();
}

void NrdWrapper::createResourceDescriptorPools() {
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_instance);
    auto framework = framework_.lock();
    if (!framework) { throw std::runtime_error("NRD: framework expired while creating descriptor pools"); }

    const uint32_t frameCount = framework->swapchain()->imageCount();
    m_resourceDescriptorSetsPerFrame = std::max(1U, iDesc->descriptorPoolDesc.setsMaxNum);
    m_frameDescriptorPools.resize(frameCount);

    std::vector<VkDescriptorPoolSize> poolSizes;
    if (iDesc->descriptorPoolDesc.totalTexturesNum > 0) {
        poolSizes.push_back({
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = iDesc->descriptorPoolDesc.totalTexturesNum,
        });
    }
    if (iDesc->descriptorPoolDesc.totalStorageTexturesNum > 0) {
        poolSizes.push_back({
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = iDesc->descriptorPoolDesc.totalStorageTexturesNum,
        });
    }
    if (poolSizes.empty()) { throw std::runtime_error("NRD: descriptor pool requires at least one image descriptor"); }

    auto device = framework->device();
    for (NRDFrameDescriptorPool &framePool : m_frameDescriptorPools) {
        const VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                  .flags = 0,
                                                  .maxSets = m_resourceDescriptorSetsPerFrame,
                                                  .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                                  .pPoolSizes = poolSizes.data()};

        throwVkError(vkCreateDescriptorPool(device->vkDevice(), &poolInfo, nullptr, &framePool.resourceDescriptorPool),
                     "create resource descriptor pool");
        framePool.nextResourceDescriptorSet = 0;
    }
}

void NrdWrapper::resetResourceDescriptorPool(uint32_t frameIndex) {
    if (frameIndex >= m_frameDescriptorPools.size()) {
        std::ostringstream error;
        error << "NRD: frame index " << frameIndex << " is outside descriptor pool count "
              << m_frameDescriptorPools.size();
        throw std::runtime_error(error.str());
    }
    auto framework = framework_.lock();
    if (!framework) { throw std::runtime_error("NRD: framework expired while resetting descriptor pool"); }

    NRDFrameDescriptorPool &framePool = m_frameDescriptorPools[frameIndex];
    throwVkError(vkResetDescriptorPool(framework->device()->vkDevice(), framePool.resourceDescriptorPool, 0),
                 "reset resource descriptor pool");
    framePool.nextResourceDescriptorSet = 0;
    framePool.nextConstantBufferSlot = 0;
    m_activeDescriptorFrameIndex = frameIndex;
    m_imageAccess.clear();
}

VkDescriptorSet NrdWrapper::allocateResourceDescriptorSet(uint32_t frameIndex, uint32_t pipelineIndex) {
    if (frameIndex >= m_frameDescriptorPools.size()) {
        std::ostringstream error;
        error << "NRD: frame index " << frameIndex << " is outside descriptor pool count "
              << m_frameDescriptorPools.size();
        throw std::runtime_error(error.str());
    }
    if (pipelineIndex >= m_pipelines.size()) {
        std::ostringstream error;
        error << "NRD: pipeline index " << pipelineIndex << " is outside pipeline count " << m_pipelines.size();
        throw std::runtime_error(error.str());
    }

    NRDFrameDescriptorPool &framePool = m_frameDescriptorPools[frameIndex];
    if (framePool.nextResourceDescriptorSet >= m_resourceDescriptorSetsPerFrame) {
        std::ostringstream error;
        error << "NRD: descriptor pool for frame " << frameIndex << " exhausted ("
              << framePool.nextResourceDescriptorSet << " / " << m_resourceDescriptorSetsPerFrame << ")";
        throw std::runtime_error(error.str());
    }

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    const VkDescriptorSetAllocateInfo allocateInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = framePool.resourceDescriptorPool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &m_pipelines[pipelineIndex].resourceDescriptorLayout};

    auto framework = framework_.lock();
    if (!framework) { throw std::runtime_error("NRD: framework expired while allocating descriptor set"); }

    throwVkError(vkAllocateDescriptorSets(framework->device()->vkDevice(), &allocateInfo, &descriptorSet),
                 "allocate resource descriptor set");
    framePool.nextResourceDescriptorSet++;
    return descriptorSet;
}

void NrdWrapper::denoise(const nrd::Identifier *denoisers, uint32_t denoisersNum, VkCommandBuffer &commandBuffer) {
    if (denoisersNum == 0 || denoisers == nullptr) {
        throw std::invalid_argument("NRD: an explicit nonempty denoiser list is required");
    }
    for (uint32_t i = 0; i < denoisersNum; ++i) {
        if (std::find(m_denoiserIdentifiers.begin(), m_denoiserIdentifiers.end(), denoisers[i]) ==
            m_denoiserIdentifiers.end()) {
            throw std::invalid_argument("NRD: dispatch requested an identifier absent from this instance");
        }
    }
    const nrd::DispatchDesc *dispatchDescs = nullptr;
    uint32_t dispatchDescsNum = 0;
    callNrd(nrd::GetComputeDispatches(*m_instance, denoisers, denoisersNum, dispatchDescs, dispatchDescsNum));
    if (dispatchDescs == nullptr || dispatchDescsNum == 0) {
        throw std::runtime_error("NRD: no compute dispatches returned for the requested denoisers");
    }

    for (uint32_t d = 0; d < dispatchDescsNum; ++d) {
        const nrd::DispatchDesc &dDesc = dispatchDescs[d];

        const bool hasDebugLabels = vkCmdBeginDebugUtilsLabelEXT != nullptr && vkCmdEndDebugUtilsLabelEXT != nullptr;
        if (hasDebugLabels) {
            const VkDebugUtilsLabelEXT label{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                .pLabelName = dDesc.name,
                .color = {0.40f, 0.80f, 0.45f, 1.0f},
            };
            vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &label);
        }
        dispatch(commandBuffer, dDesc);
        if (hasDebugLabels) { vkCmdEndDebugUtilsLabelEXT(commandBuffer); }
    }
}

void NrdWrapper::dispatch(VkCommandBuffer commandBuffer, const nrd::DispatchDesc &dispatchDesc) {
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_instance);

    if (dispatchDesc.pipelineIndex >= m_pipelines.size() || dispatchDesc.pipelineIndex >= iDesc->pipelinesNum) {
        std::ostringstream error;
        error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
              << "' references invalid pipeline index " << dispatchDesc.pipelineIndex;
        throw std::runtime_error(error.str());
    }

    const nrd::PipelineDesc &pDesc = iDesc->pipelines[dispatchDesc.pipelineIndex];

    const uint32_t texturesBindingBase = lDesc->spirvBindingOffsets.textureOffset + iDesc->resourcesBaseRegisterIndex;
    const uint32_t storageTextureBindingBase =
        lDesc->spirvBindingOffsets.storageTextureAndBufferOffset + iDesc->resourcesBaseRegisterIndex;

    NRDPipeline &pipeline = m_pipelines[dispatchDesc.pipelineIndex];

    if (pipeline.pipeline == VK_NULL_HANDLE || pipeline.pipelineLayout == VK_NULL_HANDLE ||
        pipeline.resourceDescriptorLayout == VK_NULL_HANDLE) {
        std::ostringstream error;
        error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
              << "' uses an incomplete Vulkan pipeline";
        throw std::runtime_error(error.str());
    }

    uint32_t expectedResourcesNum = 0;
    for (uint32_t r = 0; r < pDesc.resourceRangesNum; ++r) {
        expectedResourcesNum += pDesc.resourceRanges[r].descriptorsNum;
    }
    if (expectedResourcesNum != dispatchDesc.resourcesNum) {
        std::ostringstream error;
        error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
              << "' resource count mismatch: pipeline expects " << expectedResourcesNum << ", dispatch provides "
              << dispatchDesc.resourcesNum;
        throw std::runtime_error(error.str());
    }

    std::vector<VkWriteDescriptorSet> descriptorUpdates(
        dispatchDesc.resourcesNum, VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr});
    std::vector<VkDescriptorImageInfo> descriptorImageInfos(dispatchDesc.resourcesNum);

    auto framework = framework_.lock();
    if (!framework) { throw std::runtime_error("NRD: framework expired before dispatch"); }
    auto context = framework->safeAcquireCurrentContext();
    const uint32_t descriptorFrameIndex = context->frameIndex;
    if (m_activeDescriptorFrameIndex != descriptorFrameIndex) { resetResourceDescriptorPool(descriptorFrameIndex); }
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    std::vector<VkImageMemoryBarrier> imageBarriers;

    uint32_t numResourceUpdates = 0;
    for (uint32_t r = 0; r < pDesc.resourceRangesNum; ++r) {
        const nrd::ResourceRangeDesc &resourceRange = pDesc.resourceRanges[r];
        const bool isStorage = resourceRange.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;

        uint32_t rangeBaseBindingIndex = isStorage ? storageTextureBindingBase : texturesBindingBase;

        for (uint32_t d = 0; d < resourceRange.descriptorsNum; ++d) {
            if (numResourceUpdates >= dispatchDesc.resourcesNum) {
                std::ostringstream error;
                error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
                      << "' overran its resource list";
                throw std::runtime_error(error.str());
            }

            const nrd::ResourceDesc &nrdResource = dispatchDesc.resources[numResourceUpdates];
            const uint32_t binding = rangeBaseBindingIndex + d;
            const VkDescriptorType descriptorType = toVkDescriptorType(nrdResource.descriptorType);

            VkWriteDescriptorSet &update = descriptorUpdates[numResourceUpdates];
            update.dstBinding = binding;
            update.descriptorCount = 1;
            update.descriptorType = descriptorType;
            update.pImageInfo = &descriptorImageInfos[numResourceUpdates];

            if (nrdResource.descriptorType != resourceRange.descriptorType) {
                std::ostringstream error;
                error << "NRD dispatch resource order mismatch in pass '"
                      << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>") << "'";
                throw std::runtime_error(error.str());
            }

            const VkDescriptorSetLayoutBinding *layoutBinding = findBinding(pipeline.resourceBindings, binding);
            if (layoutBinding == nullptr) {
                std::ostringstream error;
                error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
                      << "' writes binding " << binding << ", but pipeline " << dispatchDesc.pipelineIndex
                      << " layout does not contain it";
                throw std::runtime_error(error.str());
            }
            if (layoutBinding->descriptorType != descriptorType) {
                std::ostringstream error;
                error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
                      << "' writes binding " << binding << " as " << vkDescriptorTypeName(descriptorType)
                      << ", but the pipeline layout expects " << vkDescriptorTypeName(layoutBinding->descriptorType);
                throw std::runtime_error(error.str());
            }

            std::shared_ptr<vk::DeviceLocalImage> texture = nullptr;
            if (nrdResource.type == nrd::ResourceType::TRANSIENT_POOL) {
                if (nrdResource.indexInPool >= m_transientTextures.size()) {
                    std::ostringstream error;
                    error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
                          << "' references transient texture " << nrdResource.indexInPool << " of "
                          << m_transientTextures.size();
                    throw std::runtime_error(error.str());
                }
                texture = m_transientTextures[nrdResource.indexInPool];
            } else if (nrdResource.type == nrd::ResourceType::PERMANENT_POOL) {
                if (nrdResource.indexInPool >= m_permanentTextures.size()) {
                    std::ostringstream error;
                    error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>")
                          << "' references permanent texture " << nrdResource.indexInPool << " of "
                          << m_permanentTextures.size();
                    throw std::runtime_error(error.str());
                }
                texture = m_permanentTextures[nrdResource.indexInPool];
            } else {
                if (!m_userTexturePool) {
                    throw std::runtime_error("NRD user texture pool is not bound before denoise dispatch");
                }
                texture = m_userTexturePool->at((uint32_t)nrdResource.type);
            }

            if (!texture) {
                std::ostringstream error;
                const char *resourceName = nrd::GetResourceTypeString(nrdResource.type);
                error << "NRD missing texture for resource " << (resourceName != nullptr ? resourceName : "UNKNOWN")
                      << " in pass '" << dispatchDesc.name << "'";
                throw std::runtime_error(error.str());
            }

            if (texture->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
                VkImageMemoryBarrier initialization{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                initialization.srcAccessMask = 0;
                initialization.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                initialization.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                initialization.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                initialization.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                initialization.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                initialization.image = texture->vkImage();
                initialization.subresourceRange = vk::wholeColorSubresourceRange;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &initialization);
                const VkClearColorValue zero{};
                vkCmdClearColorImage(commandBuffer, texture->vkImage(), VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                                     &initialization.subresourceRange);
                initialization.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                initialization.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                initialization.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &initialization);
                texture->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            }

            constexpr VkAccessFlags previousShaderAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            const VkAccessFlags nextShaderAccess =
                isStorage ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            addShaderImageBarrier(imageBarriers, texture, queueIndex, previousShaderAccess, nextShaderAccess);

            descriptorImageInfos[numResourceUpdates] = {
                .imageView = texture->vkImageView(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            ++numResourceUpdates;
        }
    }
    if (numResourceUpdates != dispatchDesc.resourcesNum) {
        std::ostringstream error;
        error << "NRD dispatch '" << (dispatchDesc.name != nullptr ? dispatchDesc.name : "<unnamed>") << "' consumed "
              << numResourceUpdates << " resources, but dispatch provides " << dispatchDesc.resourcesNum;
        throw std::runtime_error(error.str());
    }

    imageBarriers.erase(std::remove_if(imageBarriers.begin(), imageBarriers.end(),
                                       [&](VkImageMemoryBarrier &barrier) {
                                           auto [access, inserted] = m_imageAccess.try_emplace(
                                               barrier.image, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                                           barrier.srcAccessMask = access->second;
                                           access->second = barrier.dstAccessMask;
                                           return barrier.oldLayout == barrier.newLayout &&
                                                  ((barrier.srcAccessMask | barrier.dstAccessMask) &
                                                   VK_ACCESS_SHADER_WRITE_BIT) == 0;
                                       }),
                        imageBarriers.end());
    if (!imageBarriers.empty()) {
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, (uint32_t)imageBarriers.size(), imageBarriers.data());
    }

    VkDescriptorSet resourceDescriptorSet = VK_NULL_HANDLE;
    if (numResourceUpdates > 0) {
        resourceDescriptorSet = allocateResourceDescriptorSet(descriptorFrameIndex, dispatchDesc.pipelineIndex);
        for (uint32_t i = 0; i < numResourceUpdates; i++) { descriptorUpdates[i].dstSet = resourceDescriptorSet; }
        vkUpdateDescriptorSets(framework->device()->vkDevice(), numResourceUpdates, descriptorUpdates.data(), 0,
                               nullptr);
    }

    const bool samplersInSeparateSet = iDesc->constantBufferAndSamplersSpaceIndex != iDesc->resourcesSpaceIndex;
    assert(samplersInSeparateSet);

    uint32_t constantBufferOffset = descriptorFrameIndex * m_resourceDescriptorSetsPerFrame * m_constantBufferStride;
    if (pDesc.hasConstantData) {
        auto &framePool = m_frameDescriptorPools[descriptorFrameIndex];
        if (framePool.nextConstantBufferSlot >= m_resourceDescriptorSetsPerFrame ||
            dispatchDesc.constantBufferData == nullptr || dispatchDesc.constantBufferDataSize == 0 ||
            dispatchDesc.constantBufferDataSize > iDesc->constantBufferMaxDataSize) {
            throw std::runtime_error("NRD: invalid constant data or exhausted frame constant buffer arena");
        }
        constantBufferOffset += framePool.nextConstantBufferSlot++ * m_constantBufferStride;
        std::memcpy(static_cast<uint8_t *>(m_constantBuffer->mappedPtr()) + constantBufferOffset,
                    dispatchDesc.constantBufferData, dispatchDesc.constantBufferDataSize);
        m_constantBuffer->flush(dispatchDesc.constantBufferDataSize, constantBufferOffset);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout,
                            iDesc->constantBufferAndSamplersSpaceIndex, 1, &m_samplerConstBufferDescriptorSet, 1,
                            &constantBufferOffset);

    if (resourceDescriptorSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout,
                                iDesc->resourcesSpaceIndex, 1, &resourceDescriptorSet, 0, nullptr);
    }

    vkCmdDispatch(commandBuffer, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
}
