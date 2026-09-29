#pragma once

#include "core/all_extern.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace vk {

enum class ResourceMemoryCategory : uint8_t {
    Chunk = 0,
    Entity = 1,
    Texture = 2,
    Other = 3,
    Count = 4,
};

#if defined(MCVR_ENABLE_PROFILER)

struct ResourceMemoryCounters {
    int64_t allocationBytes = 0;
    int64_t deviceLocalBytes = 0;
    int64_t hostVisibleBytes = 0;
    int64_t allocationCount = 0;
};

struct ResourceMemorySnapshot {
    std::array<ResourceMemoryCounters, static_cast<size_t>(ResourceMemoryCategory::Count)> categories{};

    ResourceMemoryCounters total() const;
};

const char *resourceMemoryCategoryName(ResourceMemoryCategory category);

void trackVmaAllocation(VmaAllocator allocator, VmaAllocation allocation, ResourceMemoryCategory category);
void untrackVmaAllocation(VmaAllocator allocator, VmaAllocation allocation, ResourceMemoryCategory category);
void retagVmaAllocation(VmaAllocator allocator,
                        VmaAllocation allocation,
                        ResourceMemoryCategory oldCategory,
                        ResourceMemoryCategory newCategory);

ResourceMemorySnapshot getResourceMemorySnapshot();
void recordResourceMemoryProfilerMetrics(const char *phase);

class ResourceMemoryTagScope {
  public:
    explicit ResourceMemoryTagScope(std::string tag);
    ~ResourceMemoryTagScope();

    ResourceMemoryTagScope(const ResourceMemoryTagScope &) = delete;
    ResourceMemoryTagScope &operator=(const ResourceMemoryTagScope &) = delete;

  private:
    bool active_ = true;
};

#else

inline void trackVmaAllocation(VmaAllocator, VmaAllocation, ResourceMemoryCategory) {}
inline void untrackVmaAllocation(VmaAllocator, VmaAllocation, ResourceMemoryCategory) {}
inline void retagVmaAllocation(VmaAllocator, VmaAllocation, ResourceMemoryCategory, ResourceMemoryCategory) {}

class ResourceMemoryTagScope {
  public:
    template <typename Tag>
    explicit ResourceMemoryTagScope(Tag &&) {}

    ResourceMemoryTagScope(const ResourceMemoryTagScope &) = delete;
    ResourceMemoryTagScope &operator=(const ResourceMemoryTagScope &) = delete;
};

#endif

}
