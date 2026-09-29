#pragma once

#include "core/all_extern.hpp"

#include <cstdint>

namespace vk {

const char *resultName(VkResult result);

void requireSuccess(VkResult result, const char *operation);

void waitForFence(VkDevice device,
                  VkFence fence,
                  const char *operation,
                  uint64_t timeoutNanoseconds = 10'000'000'000ull);

}
