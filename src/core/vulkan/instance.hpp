#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/window_surface.hpp"

namespace vk {
class Instance : public SharedObject<Instance> {
  public:
    explicit Instance(WindowSurfaceBackend surfaceBackend = WindowSurfaceBackend::Glfw);
    ~Instance();

    VkInstance &vkInstance();
    bool isDlssRRInstanceExtensionsCompatible() const;
    bool isDlssSRInstanceExtensionsCompatible() const;
    bool isDlssFrameGenerationInstanceExtensionsCompatible() const;
    bool isXessInstanceExtensionsCompatible() const;

  private:
    VkInstance instance_;
    bool dlssRRInstanceExtensionsCompatible_ = false;
    bool dlssSRInstanceExtensionsCompatible_ = false;
    bool dlssFrameGenerationInstanceExtensionsCompatible_ = false;
    bool xessInstanceExtensionsCompatible_ = false;
};
}
