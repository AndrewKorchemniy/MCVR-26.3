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

#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"

#include <NRD.h>

#include "core/render/render_framework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class NrdWrapper : public SharedObject<NrdWrapper> {
  public:
    NrdWrapper() = default;
    ~NrdWrapper();

    void init(std::shared_ptr<Framework> framework,
              uint16_t width,
              uint16_t height,
              bool enableSh,
              bool enableSeparateDirectDenoising,
              bool enableIndirectDenoising,
              bool splitIndirectLobes);

    void setUserPoolTexture(
        std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>
            userTexturePool);

    void setCommonSettings(nrd::CommonSettings &settings);

    void setReblurSettings(const nrd::ReblurSettings &settings);
    void setRelaxSettings(const nrd::RelaxSettings &settings);

    void denoise(const nrd::Identifier *denoisers, uint32_t denoisersNum, VkCommandBuffer &commandBuffer);

    static VkFormat getNormalRoughnessFormat();

  private:
    struct NRDPipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout resourceDescriptorLayout = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayoutBinding> resourceBindings;
        uint32_t numBindings = 0;
    };

    struct NRDFrameDescriptorPool {
        VkDescriptorPool resourceDescriptorPool = VK_NULL_HANDLE;
        uint32_t nextResourceDescriptorSet = 0;
        uint32_t nextConstantBufferSlot = 0;
    };

    NrdWrapper(const NrdWrapper &) = delete;
    NrdWrapper &operator=(const NrdWrapper) = delete;

    static void callNrd(nrd::Result result);
    static VkFormat toVkFormat(nrd::Format nrdFormat);
    static VkDescriptorType toVkDescriptorType(nrd::DescriptorType type);
    static VkFilter toVkFilter(nrd::Sampler sampler);
    static uint16_t divideRoundUp(uint32_t dividend, uint16_t divisor);
    static const char *vkResultName(VkResult result);
    static void throwVkError(VkResult result, const char *operation);
    static const char *vkDescriptorTypeName(VkDescriptorType type);
    static const VkDescriptorSetLayoutBinding *findBinding(const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                                                           uint32_t binding);
    static void addUniqueBinding(std::vector<VkDescriptorSetLayoutBinding> &bindings,
                                 const VkDescriptorSetLayoutBinding &binding,
                                 const char *owner);
    static void transitionAndClearTexture(const std::shared_ptr<Framework> &framework,
                                          const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                          const std::shared_ptr<vk::DeviceLocalImage> &image,
                                          VkImageLayout layout);
    static void addShaderImageBarrier(std::vector<VkImageMemoryBarrier> &imageBarriers,
                                      const std::shared_ptr<vk::DeviceLocalImage> &image,
                                      uint32_t queueIndex,
                                      VkAccessFlags srcAccess,
                                      VkAccessFlags dstAccess);

    std::shared_ptr<vk::DeviceLocalImage> createTexture(const nrd::TextureDesc &tDesc, uint16_t width, uint16_t height);
    void createPipelines();
    void setDenoiserSettings(nrd::Identifier identifier, const void *settings);
    void dispatch(VkCommandBuffer commandBuffer, const nrd::DispatchDesc &dispatchDesc);
    void createResourceDescriptorPools();
    void resetResourceDescriptorPool(uint32_t frameIndex);
    VkDescriptorSet allocateResourceDescriptorSet(uint32_t frameIndex, uint32_t pipelineIndex);

    std::weak_ptr<Framework> framework_;
    nrd::Instance *m_instance = nullptr;
    std::vector<nrd::Identifier> m_denoiserIdentifiers;
    bool isShEnabled_ = false;
    bool isSeparateDirectDenoisingEnabled_ = true;
    bool isIndirectDenoisingEnabled_ = true;
    bool splitIndirectLobes_ = false;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_permanentTextures;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_transientTextures;
    std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>
        m_userTexturePool;
    std::vector<VkSampler> m_vkSamplers;
    std::vector<std::shared_ptr<vk::Sampler>> m_samplers;
    std::shared_ptr<vk::HostVisibleBuffer> m_constantBuffer;
    uint32_t m_constantBufferStride = 0;

    std::vector<NRDPipeline> m_pipelines;
    std::vector<NRDFrameDescriptorPool> m_frameDescriptorPools;
    uint32_t m_resourceDescriptorSetsPerFrame = 0;
    uint32_t m_activeDescriptorFrameIndex = UINT32_MAX;
    std::unordered_map<VkImage, VkAccessFlags> m_imageAccess;

    VkDescriptorSetLayout m_samplerConstBufferDescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_samplerConstBufferDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_samplerConstBufferDescriptorSet = VK_NULL_HANDLE;
};
