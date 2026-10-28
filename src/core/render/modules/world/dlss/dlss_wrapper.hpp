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

#include "core/all_extern.hpp"

#include "nvsdk_ngx_defs_dlssd.h"
#include "nvsdk_ngx_defs_dlssg.h"
#include "nvsdk_ngx_vk.h"

#include <glm/glm.hpp>

#include <array>
#include <string>
#include <vector>

namespace vk {
class Instance;
class Device;
class PhysicalDevice;
class CommandPool;
class CommandBuffer;
class DeviceLocalImage;
};

std::string getNGXResultString(NVSDK_NGX_Result result);

NVSDK_NGX_Result checkNgxResult(NVSDK_NGX_Result result, const char *func, int line);

class DlssRR;
class DlssSR;

class NgxContext : public SharedObject<NgxContext> {
  public:
    NgxContext() = default;
    ~NgxContext();

    struct NgxInitInfo {
        std::shared_ptr<vk::Instance> instance;
        std::shared_ptr<vk::PhysicalDevice> physicalDevice;
        std::shared_ptr<vk::Device> device;
        std::string applicationPath;
    };

    NVSDK_NGX_Result init(const NgxInitInfo &initInfo);

    static std::shared_ptr<NgxContext> acquireShared(const NgxInitInfo &initInfo);
    static void shutdownShared();

    void deinit();

    NVSDK_NGX_Parameter *parameters();

    NVSDK_NGX_Result queryDlssRRAvailable();
    NVSDK_NGX_Result queryDlssSRAvailable();
    NVSDK_NGX_Result queryDlssFrameGenerationAvailable();
    uint32_t queryDlssFrameGenerationMaxFrames();

    struct SupportedSizes {
        VkExtent2D minSize = {};
        VkExtent2D maxSize = {};
        VkExtent2D optimalSize = {};
    };
    struct QuerySizeInfo {
        VkExtent2D outputSize;
        NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_Balanced;
    };
    NVSDK_NGX_Result querySupportedDlssRRInputSizes(const QuerySizeInfo &outputSize, SupportedSizes &renderSizes);
    NVSDK_NGX_Result querySupportedDlssSRInputSizes(const QuerySizeInfo &outputSize, SupportedSizes &renderSizes);

    struct DlssRRInitInfo {
        VkExtent2D inputSize = {};
        VkExtent2D outputSize = {};
        NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_Balanced;
        NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E;
    };
    NVSDK_NGX_Result initDlssRR(const DlssRRInitInfo &initInfo,
                                std::shared_ptr<vk::CommandPool> cmdPool,
                                std::shared_ptr<DlssRR> dlssrr);

    struct DlssSRInitInfo {
        VkExtent2D inputSize = {};
        VkExtent2D outputSize = {};
        NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_Balanced;
        int featureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                 NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        NVSDK_NGX_DLSS_Hint_Render_Preset preset = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
    };
    NVSDK_NGX_Result initDlssSR(const DlssSRInitInfo &initInfo,
                                std::shared_ptr<vk::CommandPool> cmdPool,
                                std::shared_ptr<DlssSR> dlsssr);

    static NVSDK_NGX_Result getDlssRRRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result getDlssSRRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result
    getDlssFrameGenerationRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions);

    static NVSDK_NGX_Result getDlssRRRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                              std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                              std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result getDlssSRRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                              std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                              std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result
    getDlssFrameGenerationRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                   std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                   std::vector<VkExtensionProperties> &extensions);

    // The same queries on raw handles, for asking before Minecraft creates its device.
    static NVSDK_NGX_Result getDlssRRRequiredDeviceExtensions(VkInstance instance,
                                                              VkPhysicalDevice physicalDevice,
                                                              std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result getDlssSRRequiredDeviceExtensions(VkInstance instance,
                                                              VkPhysicalDevice physicalDevice,
                                                              std::vector<VkExtensionProperties> &extensions);
    static NVSDK_NGX_Result getDlssFrameGenerationRequiredDeviceExtensions(
        VkInstance instance, VkPhysicalDevice physicalDevice, std::vector<VkExtensionProperties> &extensions);

  private:
    NgxContext(const NgxContext &) = delete;
    NgxContext(const NgxContext &&) = delete;
    NgxContext &operator=(const NgxContext &) = delete;
    NgxContext &operator=(const NgxContext &&) = delete;

    std::shared_ptr<vk::Device> device_;
    NVSDK_NGX_Parameter *ngxParams_ = nullptr;
    std::wstring applicationPath_;

    static std::shared_ptr<NgxContext> sharedContext_;
};

class DlssRR : public SharedObject<DlssRR> {
  public:
    DlssRR() = default;
    ~DlssRR();

    void deinit();

    enum DlssResource {
        RESOURCE_COLOR_IN = 0,
        RESOURCE_COLOR_OUT,
        RESOURCE_DIFFUSE_ALBEDO,
        RESOURCE_SPECULAR_ALBEDO,
        RESOURCE_NORMALROUGHNESS,
        RESOURCE_MOTIONVECTOR,
        RESOURCE_LINEARDEPTH,
        RESOURCE_SPECULAR_HITDISTANCE,

        RESOURCE_NUM
    };

    void setResource(DlssResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image);
    void resetResource(DlssResource resourceId);

    NVSDK_NGX_Result denoise(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                             glm::uvec2 renderSize,
                             glm::vec2 jitter,
                             const glm::mat4 &modelView,
                             const glm::mat4 &projection,
                             bool reset = false);

  private:
    friend class NgxContext;
    NVSDK_NGX_Result init(std::shared_ptr<vk::Device> device,
                          std::shared_ptr<vk::CommandPool> cmdPool,
                          NVSDK_NGX_Parameter *ngxParams,
                          const NgxContext::DlssRRInitInfo &info);

    DlssRR(const DlssRR &) = delete;
    DlssRR(const DlssRR &&) = delete;
    DlssRR &operator=(const DlssRR &) = delete;
    DlssRR &operator=(const DlssRR &&) = delete;

    std::shared_ptr<vk::Device> m_device = VK_NULL_HANDLE;
    NVSDK_NGX_Parameter *m_ngxParams = nullptr;
    NVSDK_NGX_Handle *m_dlssdHandle = nullptr;
    VkExtent2D m_inputSize;
    VkExtent2D m_outputSize;
    std::array<NVSDK_NGX_Resource_VK, RESOURCE_NUM> m_resources;
};

class DlssSR : public SharedObject<DlssSR> {
  public:
    DlssSR() = default;
    ~DlssSR();

    void deinit();

    enum DlssResource {
        RESOURCE_COLOR_IN = 0,
        RESOURCE_COLOR_OUT,
        RESOURCE_MOTIONVECTOR,
        RESOURCE_DEPTH,

        RESOURCE_NUM
    };

    void setResource(DlssResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image);
    void resetResource(DlssResource resourceId);

    NVSDK_NGX_Result evaluate(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                              glm::uvec2 renderSize,
                              glm::vec2 jitter,
                              glm::vec2 motionVectorScale = glm::vec2(1.0f),
                              bool reset = false,
                              float frameTimeMilliseconds = 1000.0f / 60.0f);

  private:
    friend class NgxContext;
    NVSDK_NGX_Result init(std::shared_ptr<vk::Device> device,
                          std::shared_ptr<vk::CommandPool> cmdPool,
                          NVSDK_NGX_Parameter *ngxParams,
                          const NgxContext::DlssSRInitInfo &info);

    DlssSR(const DlssSR &) = delete;
    DlssSR(const DlssSR &&) = delete;
    DlssSR &operator=(const DlssSR &) = delete;
    DlssSR &operator=(const DlssSR &&) = delete;

    std::shared_ptr<vk::Device> m_device = VK_NULL_HANDLE;
    NVSDK_NGX_Parameter *m_ngxParams = nullptr;
    NVSDK_NGX_Handle *m_dlssHandle = nullptr;
    VkExtent2D m_inputSize;
    VkExtent2D m_outputSize;
    std::array<NVSDK_NGX_Resource_VK, RESOURCE_NUM> m_resources;
};
