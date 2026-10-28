#include "core/vulkan/device.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"
#include "core/render/renderer.hpp"

#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif

std::ostream &deviceCerr() {
    return std::cerr << "[Device] ";
}

vk::Device::Device(std::shared_ptr<Instance> instance,
                   std::shared_ptr<PhysicalDevice> physicalDevice,
                   VkDevice device,
                   const std::vector<std::string> &enabledExtensions)
    : instance_(instance), physicalDevice_(physicalDevice), device_(device) {
    if (device_ == VK_NULL_HANDLE) { throw std::invalid_argument("Vulkan device handle is null"); }
    enabledExtensions_.insert(enabledExtensions.begin(), enabledExtensions.end());

    volkLoadDevice(device_);

    // The queues come from vanilla. The compute queue is separate from the graphics queue only if vanilla created it
    // as a different VkQueue; vanilla never submits to it, so it is free for asynchronous work.
    const AdoptedQueues &queues = physicalDevice_->queues();
    mainQueue_ = queues.graphics;
    presentQueue_ = mainQueue_;
    independentSecondaryQueue_ = physicalDevice_->hasIndependentSecondaryQueue();
    secondaryQueue_ = independentSecondaryQueue_ ? queues.compute : mainQueue_;
    asyncComputeQueue_ = secondaryQueue_;

    for (const char *required : {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                 VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}) {
        if (!hasEnabledExtension(required)) {
            throw std::runtime_error(std::string("Vulkan device was created without required extension ") + required);
        }
    }
    if (!hasEnabledExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        deviceCerr() << "VK_KHR_push_descriptor was not enabled on the device; the main render module needs it."
                     << std::endl;
    }

    memoryBudgetEnabled_ = hasEnabledExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    // Vanilla only reports the extension list, so a feature is assumed on when the GPU supports it.
    multiDrawIndirect_ = physicalDevice_->supportedFeatures().multiDrawIndirect == VK_TRUE;
    drawIndirectFirstInstance_ = physicalDevice_->supportedFeatures().drawIndirectFirstInstance == VK_TRUE;
    extendedDynamicState2LogicOp_ = false;

    // A FeatureSet only enables an extension when its feature is supported too, so the extension alone is enough.
#if defined(VK_EXT_ray_tracing_invocation_reorder)
    shaderInvocationReorderEXT_ = hasEnabledExtension(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
#endif
#if defined(VK_NV_ray_tracing_invocation_reorder)
    shaderInvocationReorderNV_ =
        !shaderInvocationReorderEXT_ && hasEnabledExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
#endif

    auto areRequiredExtensionsEnabled = [&](const std::vector<std::string> &requiredExtensions) {
        for (const auto &requiredExtension : requiredExtensions) {
            if (requiredExtension == "VK_EXT_buffer_device_address") { continue; }
            if (!hasEnabledExtension(requiredExtension.c_str())) { return false; }
        }
        return true;
    };

    std::vector<VkExtensionProperties> dlssRRExtensions;
    NVSDK_NGX_Result dlssRRResult =
        NgxContext::getDlssRRRequiredDeviceExtensions(instance_, physicalDevice_, dlssRRExtensions);
    if (NVSDK_NGX_FAILED(dlssRRResult)) {
        deviceCerr() << "dlss ray reconstruction device extensions unavailable; skipping." << std::endl;
    } else {
        std::vector<std::string> required;
        for (const auto &extension : dlssRRExtensions) { required.emplace_back(extension.extensionName); }
        dlssRRDeviceExtensionsCompatible_ =
            instance_->isDlssRRInstanceExtensionsCompatible() && areRequiredExtensionsEnabled(required);
    }
    if (!dlssRRDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss ray reconstruction device extension requirements are not fully satisfied." << std::endl;
    }

    std::vector<VkExtensionProperties> dlssSRExtensions;
    NVSDK_NGX_Result dlssSRResult =
        NgxContext::getDlssSRRequiredDeviceExtensions(instance_, physicalDevice_, dlssSRExtensions);
    if (NVSDK_NGX_FAILED(dlssSRResult)) {
        deviceCerr() << "dlss super resolution device extensions unavailable; skipping." << std::endl;
    } else {
        std::vector<std::string> required;
        for (const auto &extension : dlssSRExtensions) { required.emplace_back(extension.extensionName); }
        dlssSRDeviceExtensionsCompatible_ =
            instance_->isDlssSRInstanceExtensionsCompatible() && areRequiredExtensionsEnabled(required);
    }
    if (!dlssSRDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss super resolution device extension requirements are not fully satisfied." << std::endl;
    }

    std::vector<VkExtensionProperties> dlssFrameGenerationExtensions;
    NVSDK_NGX_Result dlssFrameGenerationResult = NgxContext::getDlssFrameGenerationRequiredDeviceExtensions(
        instance_, physicalDevice_, dlssFrameGenerationExtensions);
    if (NVSDK_NGX_FAILED(dlssFrameGenerationResult)) {
        deviceCerr() << "dlss frame generation device extensions unavailable; skipping." << std::endl;
    } else {
        std::vector<std::string> required;
        for (const auto &extension : dlssFrameGenerationExtensions) { required.emplace_back(extension.extensionName); }
        dlssFrameGenerationDeviceExtensionsCompatible_ =
            instance_->isDlssFrameGenerationInstanceExtensionsCompatible() && areRequiredExtensionsEnabled(required);
    }
    if (!dlssFrameGenerationDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss frame generation device extension requirements are not fully satisfied." << std::endl;
    }

#ifdef MCVR_ENABLE_FFX_UPSCALER
    const std::vector<std::string> fsrFrameGenerationRequiredExtensions = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,          VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
    };
    // synchronization2, image_format_list and descriptor_indexing are core in the 1.3 device vanilla creates.
    fsrFrameGenerationDeviceExtensionsCompatible_ = hasEnabledExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) &&
                                                    hasEnabledExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    (void)fsrFrameGenerationRequiredExtensions;
    if (!fsrFrameGenerationDeviceExtensionsCompatible_) {
        deviceCerr() << "fsr frame generation device extension requirements are not fully satisfied." << std::endl;
    }
#endif

#ifdef MCVR_ENABLE_XESS
    std::vector<const char *> xessExtensions;
    if (mcvr::XeSSWrapper::getRequiredDeviceExtensions(instance_->vkInstance(), physicalDevice_->vkPhysicalDevice(),
                                                       xessExtensions)) {
        std::vector<std::string> required(xessExtensions.begin(), xessExtensions.end());
        xessDeviceExtensionsCompatible_ =
            instance_->isXessInstanceExtensionsCompatible() && areRequiredExtensionsEnabled(required);
    } else {
        deviceCerr() << "xess device extensions unavailable; skipping." << std::endl;
    }
    if (!xessDeviceExtensionsCompatible_) {
        deviceCerr() << "xess device extension requirements are not fully satisfied." << std::endl;
    }
#endif

    initPipelineCache();
}

vk::Device::~Device() {
    // The device belongs to Minecraft, which destroys it; only what MCVR created on it is released here.
    if (device_ != VK_NULL_HANDLE) {
        savePipelineCache();
        if (pipelineCache_ != VK_NULL_HANDLE) { vkDestroyPipelineCache(device_, pipelineCache_, nullptr); }
        pipelineCache_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }
}

VkDevice &vk::Device::vkDevice() {
    return device_;
}

bool vk::Device::hasEnabledExtension(const char *name) const {
    return enabledExtensions_.find(name) != enabledExtensions_.end();
}

void vk::Device::initPipelineCache() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_->vkPhysicalDevice(), &properties);
    std::ostringstream name;
    name << std::hex << properties.vendorID << '-' << properties.deviceID << '-' << properties.driverVersion << '-';
    for (uint8_t byte : properties.pipelineCacheUUID) {
        name << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    }
    if (!Renderer::folderPath.empty()) {
        pipelineCachePath_ = Renderer::folderPath / "cache/pipelines" / (name.str() + ".bin");
    }
    std::vector<char> data;
    std::ifstream input(pipelineCachePath_, std::ios::binary | std::ios::ate);
    if (input) {
        const auto size = input.tellg();
        if (size >= static_cast<std::streamoff>(sizeof(VkPipelineCacheHeaderVersionOne)) && size <= 256 * 1024 * 1024) {
            data.resize(static_cast<size_t>(size));
            input.seekg(0);
            if (!input.read(data.data(), data.size())) { data.clear(); }
        }
    }
    if (!data.empty()) {
        VkPipelineCacheHeaderVersionOne header{};
        std::memcpy(&header, data.data(), sizeof(header));
        if (header.headerSize != sizeof(header) || header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
            header.vendorID != properties.vendorID || header.deviceID != properties.deviceID ||
            std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
            data.clear();
        }
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    VkResult result = vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_);
    if (result != VK_SUCCESS && !data.empty()) {
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        result = vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_);
    }
    if (result != VK_SUCCESS) {
        pipelineCache_ = VK_NULL_HANDLE;
        deviceCerr() << "Pipeline cache unavailable: " << vk::resultName(result) << std::endl;
    }
}

VkPipelineCache vk::Device::vkPipelineCache() const {
    return pipelineCache_;
}

void vk::Device::savePipelineCache() {
    if (pipelineCache_ == VK_NULL_HANDLE || pipelineCachePath_.empty()) { return; }
    size_t size = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr) != VK_SUCCESS || size == 0 ||
        size > 256 * 1024 * 1024) {
        return;
    }
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data()) != VK_SUCCESS) { return; }
    std::error_code error;
    std::filesystem::create_directories(pipelineCachePath_.parent_path(), error);
    if (error) { return; }
    auto temporaryPath = pipelineCachePath_;
    temporaryPath += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
    {
        std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
        output.write(data.data(), size);
        output.close();
        if (!output) {
            std::filesystem::remove(temporaryPath, error);
            return;
        }
    }
#ifdef _WIN32
    if (MoveFileExW(temporaryPath.c_str(), pipelineCachePath_.c_str(), MOVEFILE_REPLACE_EXISTING) != 0) { return; }
#else
    std::filesystem::rename(temporaryPath, pipelineCachePath_, error);
    if (!error) { return; }
#endif
    std::filesystem::remove(temporaryPath, error);
}

VkQueue &vk::Device::mainVkQueue() {
    return mainQueue_;
}

VkQueue &vk::Device::presentQueue() {
    return presentQueue_;
}

VkQueue &vk::Device::secondaryQueue() {
    return secondaryQueue_;
}

VkQueue &vk::Device::asyncComputeQueue() {
    return asyncComputeQueue_;
}

uint32_t vk::Device::mainQueueFamilyIndex() const {
    return physicalDevice_->mainQueueIndex();
}

uint32_t vk::Device::secondaryQueueFamilyIndex() const {
    return physicalDevice_->secondaryQueueIndex();
}

bool vk::Device::hasIndependentSecondaryQueue() const {
    return independentSecondaryQueue_;
}

std::recursive_mutex &vk::Device::mainQueueMutex() {
    return mainQueueMutex_;
}

std::recursive_mutex &vk::Device::presentQueueMutex() {
    return presentQueue_ == mainQueue_ ? mainQueueMutex_ : presentQueueMutex_;
}

std::recursive_mutex &vk::Device::secondaryQueueMutex() {
    return independentSecondaryQueue_ ? secondaryQueueMutex_ : mainQueueMutex_;
}

std::recursive_mutex &vk::Device::asyncComputeQueueMutex() {
    if (asyncComputeQueue_ == mainQueue_) { return mainQueueMutex_; }
    if (asyncComputeQueue_ == secondaryQueue_) { return secondaryQueueMutex(); }
    return asyncComputeQueueMutex_;
}

bool vk::Device::hasExtendedDynamicState2LogicOp() const {
    return extendedDynamicState2LogicOp_;
}

bool vk::Device::hasMultiDrawIndirect() const {
    return multiDrawIndirect_;
}

bool vk::Device::hasDrawIndirectFirstInstance() const {
    return drawIndirectFirstInstance_;
}

bool vk::Device::hasShaderInvocationReorderEXT() const {
    return shaderInvocationReorderEXT_;
}

bool vk::Device::hasShaderInvocationReorderNV() const {
    return shaderInvocationReorderNV_;
}

bool vk::Device::isDlssRRDeviceExtensionsCompatible() const {
    return dlssRRDeviceExtensionsCompatible_;
}

bool vk::Device::isDlssSRDeviceExtensionsCompatible() const {
    return dlssSRDeviceExtensionsCompatible_;
}

bool vk::Device::isDlssFrameGenerationDeviceExtensionsCompatible() const {
    return dlssFrameGenerationDeviceExtensionsCompatible_;
}

bool vk::Device::isFsrFrameGenerationDeviceExtensionsCompatible() const {
    return fsrFrameGenerationDeviceExtensionsCompatible_;
}

bool vk::Device::isXessDeviceExtensionsCompatible() const {
    return xessDeviceExtensionsCompatible_;
}

bool vk::Device::hasMemoryBudget() const {
    return memoryBudgetEnabled_;
}
