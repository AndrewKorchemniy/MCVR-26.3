#include "core/vulkan/instance.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"

#include <iostream>
#include <stdexcept>

std::ostream &instanceCerr() {
    return std::cerr << "[Instance] ";
}

vk::Instance::Instance(VkInstance instance, const std::vector<std::string> &enabledExtensions) : instance_(instance) {
    if (instance_ == VK_NULL_HANDLE) { throw std::invalid_argument("Vulkan instance handle is null"); }
    enabledExtensions_.insert(enabledExtensions.begin(), enabledExtensions.end());

    if (volkInitialize() != VK_SUCCESS) { throw std::runtime_error("volkInitialize failed"); }
    volkLoadInstance(instance_);

    // The instance extensions each upscaler needs are queried from its SDK, then checked against what vanilla enabled.
    auto areRequiredExtensionsEnabled = [&](const std::vector<std::string> &requiredExtensions) {
        for (const auto &requiredExtension : requiredExtensions) {
            if (!hasExtension(requiredExtension.c_str())) { return false; }
        }
        return true;
    };
    auto collectNames = [](const std::vector<VkExtensionProperties> &extensions) {
        std::vector<std::string> names;
        for (const auto &extension : extensions) { names.emplace_back(extension.extensionName); }
        return names;
    };

    std::vector<VkExtensionProperties> dlssRRExtensions;
    if (NVSDK_NGX_SUCCEED(NgxContext::getDlssRRRequiredInstanceExtensions(dlssRRExtensions))) {
        dlssRRInstanceExtensionsCompatible_ = areRequiredExtensionsEnabled(collectNames(dlssRRExtensions));
    } else {
        instanceCerr() << "failed to query dlss ray reconstruction instance extensions; skipping." << std::endl;
    }
    if (!dlssRRInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss ray reconstruction instance extension requirements are not fully satisfied."
                       << std::endl;
    }

    std::vector<VkExtensionProperties> dlssSRExtensions;
    if (NVSDK_NGX_SUCCEED(NgxContext::getDlssSRRequiredInstanceExtensions(dlssSRExtensions))) {
        dlssSRInstanceExtensionsCompatible_ = areRequiredExtensionsEnabled(collectNames(dlssSRExtensions));
    } else {
        instanceCerr() << "failed to query dlss super resolution instance extensions; skipping." << std::endl;
    }
    if (!dlssSRInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss super resolution instance extension requirements are not fully satisfied." << std::endl;
    }

    std::vector<VkExtensionProperties> dlssFrameGenerationExtensions;
    if (NVSDK_NGX_SUCCEED(NgxContext::getDlssFrameGenerationRequiredInstanceExtensions(dlssFrameGenerationExtensions))) {
        dlssFrameGenerationInstanceExtensionsCompatible_ =
            areRequiredExtensionsEnabled(collectNames(dlssFrameGenerationExtensions));
    } else {
        instanceCerr() << "failed to query dlss frame generation instance extensions; skipping." << std::endl;
    }
    if (!dlssFrameGenerationInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss frame generation instance extension requirements are not fully satisfied." << std::endl;
    }

#ifdef MCVR_ENABLE_XESS
    std::vector<const char *> xessExtensions;
    uint32_t xessMinApiVersion = 0;
    if (mcvr::XeSSWrapper::getRequiredInstanceExtensions(xessExtensions, &xessMinApiVersion)) {
        std::vector<std::string> xessRequired(xessExtensions.begin(), xessExtensions.end());
        xessInstanceExtensionsCompatible_ = areRequiredExtensionsEnabled(xessRequired);
    } else {
        instanceCerr() << "xess instance extensions unavailable; skipping." << std::endl;
    }
    if (!xessInstanceExtensionsCompatible_) {
        instanceCerr() << "xess instance extension requirements are not fully satisfied." << std::endl;
    }
#endif
}

vk::Instance::~Instance() {
    // The instance belongs to Minecraft, which destroys it.
}

VkInstance &vk::Instance::vkInstance() {
    return instance_;
}

bool vk::Instance::hasExtension(const char *name) const {
    return enabledExtensions_.find(name) != enabledExtensions_.end();
}

bool vk::Instance::isDlssRRInstanceExtensionsCompatible() const {
    return dlssRRInstanceExtensionsCompatible_;
}

bool vk::Instance::isDlssSRInstanceExtensionsCompatible() const {
    return dlssSRInstanceExtensionsCompatible_;
}

bool vk::Instance::isDlssFrameGenerationInstanceExtensionsCompatible() const {
    return dlssFrameGenerationInstanceExtensionsCompatible_;
}

bool vk::Instance::isXessInstanceExtensionsCompatible() const {
    return xessInstanceExtensionsCompatible_;
}
