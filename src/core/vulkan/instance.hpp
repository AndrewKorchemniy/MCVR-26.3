#pragma once

#include "core/all_extern.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace vk {
// Non-owning wrapper around the VkInstance vanilla created. Never creates or destroys the instance.
class Instance : public SharedObject<Instance> {
  public:
    Instance(VkInstance instance, const std::vector<std::string> &enabledExtensions);
    ~Instance();

    VkInstance &vkInstance();
    bool hasExtension(const char *name) const;
    bool isDlssRRInstanceExtensionsCompatible() const;
    bool isDlssSRInstanceExtensionsCompatible() const;
    bool isDlssFrameGenerationInstanceExtensionsCompatible() const;
    bool isXessInstanceExtensionsCompatible() const;

  private:
    VkInstance instance_;
    std::unordered_set<std::string> enabledExtensions_;
    bool dlssRRInstanceExtensionsCompatible_ = false;
    bool dlssSRInstanceExtensionsCompatible_ = false;
    bool dlssFrameGenerationInstanceExtensionsCompatible_ = false;
    bool xessInstanceExtensionsCompatible_ = false;
};
}
