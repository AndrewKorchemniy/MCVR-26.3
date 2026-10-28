#pragma once

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#endif
// Radiance never creates surfaces or windows (Minecraft does), so no platform-specific Vulkan WSI is needed.
#include "volk.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/string_cast.hpp>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vk_mem_alloc.h"

#include <memory>
template <typename T, typename... Args>
concept TwoStepInit =
    std::is_default_constructible_v<T> && requires(T t, Args &&...args) { t.init(std::forward<Args>(args)...); };

template <typename Derived>
class SharedObject : public std::enable_shared_from_this<Derived> {
    friend Derived;

  public:
    template <typename... Args>
    static std::shared_ptr<Derived> create(Args &&...args) {
        if constexpr (TwoStepInit<Derived, Args...>) {
            auto ptr = std::make_shared<Derived>();
            ptr->init(std::forward<Args>(args)...);
            return ptr;
        } else {
            return std::make_shared<Derived>(std::forward<Args>(args)...);
        }
    }
};
#define sharedDecl(funName, retName) std::shared_ptr<retName> funName();
#define sharedImpl(namespace, funName, varName, retName)                                                               \
    std::shared_ptr<retName> namespace ::funName() {                                                                   \
        return std::shared_ptr<retName>(shared_from_this(), varName##_);                                               \
    }


#include "stb_image.h"
