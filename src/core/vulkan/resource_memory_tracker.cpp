#include "core/vulkan/resource_memory_tracker.hpp"

#if defined(MCVR_ENABLE_PROFILER)

#    include "core/profiler/profiler.hpp"

#    include <algorithm>
#    include <limits>
#    include <mutex>
#    include <string>
#    include <unordered_map>
#    include <utility>
#    include <vector>

namespace vk {
class ResourceMemoryTrackerState {
  public:
    struct AllocationDescription {
        int64_t allocationBytes = 0;
        int64_t deviceLocalBytes = 0;
        int64_t hostVisibleBytes = 0;
    };

    static constexpr size_t categoryCount = static_cast<size_t>(ResourceMemoryCategory::Count);

    static std::mutex &mutex() {
        static std::mutex value;
        return value;
    }

    static std::array<ResourceMemoryCounters, categoryCount> &counters() {
        static std::array<ResourceMemoryCounters, categoryCount> value{};
        return value;
    }

    static std::unordered_map<std::string, ResourceMemoryCounters> &tagCounters() {
        static std::unordered_map<std::string, ResourceMemoryCounters> value{};
        return value;
    }

    static std::unordered_map<VmaAllocation, std::pair<std::string, AllocationDescription>> &allocationTags() {
        static std::unordered_map<VmaAllocation, std::pair<std::string, AllocationDescription>> value{};
        return value;
    }

    static std::vector<std::string> &tagStack() {
        thread_local std::vector<std::string> value;
        return value;
    }

    static size_t categoryIndex(ResourceMemoryCategory category) {
        const size_t index = static_cast<size_t>(category);
        return index < categoryCount ? index : static_cast<size_t>(ResourceMemoryCategory::Other);
    }

    static int64_t clampToInt64(VkDeviceSize value) {
        constexpr auto maxValue = static_cast<VkDeviceSize>(std::numeric_limits<int64_t>::max());
        return value > maxValue ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(value);
    }

    static AllocationDescription describeAllocation(VmaAllocator allocator, VmaAllocation allocation) {
        AllocationDescription description{};
        if (allocator == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) { return description; }

        VmaAllocationInfo allocationInfo{};
        vmaGetAllocationInfo(allocator, allocation, &allocationInfo);

        VkMemoryPropertyFlags memoryPropertyFlags = 0;
        vmaGetAllocationMemoryProperties(allocator, allocation, &memoryPropertyFlags);

        description.allocationBytes = clampToInt64(allocationInfo.size);
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            description.deviceLocalBytes = description.allocationBytes;
        }
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            description.hostVisibleBytes = description.allocationBytes;
        }
        return description;
    }

    static void
    applyDescription(ResourceMemoryCounters &counter, const AllocationDescription &description, int64_t sign) {
        counter.allocationBytes += sign * description.allocationBytes;
        counter.deviceLocalBytes += sign * description.deviceLocalBytes;
        counter.hostVisibleBytes += sign * description.hostVisibleBytes;
        counter.allocationCount += sign;
    }

    static void clampCounter(ResourceMemoryCounters &counter) {
        counter.allocationBytes = std::max<int64_t>(0, counter.allocationBytes);
        counter.deviceLocalBytes = std::max<int64_t>(0, counter.deviceLocalBytes);
        counter.hostVisibleBytes = std::max<int64_t>(0, counter.hostVisibleBytes);
        counter.allocationCount = std::max<int64_t>(0, counter.allocationCount);
    }

    static std::string metricName(const char *prefix, const char *category, const char *suffix, const char *phase) {
        std::string name(prefix);
        name += category;
        name += suffix;
        name += phase;
        return name;
    }

    static std::string currentTag() {
        auto &stack = tagStack();
        return stack.empty() ? std::string("untagged") : stack.back();
    }

    static std::string sanitizedMetricToken(std::string value) {
        if (value.empty()) { return "empty"; }
        for (char &ch : value) {
            const bool valid = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
            if (!valid) { ch = '_'; }
        }
        return value;
    }
};

ResourceMemoryCounters ResourceMemorySnapshot::total() const {
    ResourceMemoryCounters total{};
    for (const auto &counter : categories) {
        total.allocationBytes += counter.allocationBytes;
        total.deviceLocalBytes += counter.deviceLocalBytes;
        total.hostVisibleBytes += counter.hostVisibleBytes;
        total.allocationCount += counter.allocationCount;
    }
    return total;
}

const char *resourceMemoryCategoryName(ResourceMemoryCategory category) {
    switch (category) {
        case ResourceMemoryCategory::Chunk: return "chunk";
        case ResourceMemoryCategory::Entity: return "entity";
        case ResourceMemoryCategory::Texture: return "texture";
        case ResourceMemoryCategory::Other: return "other";
        default: return "other";
    }
}

void trackVmaAllocation(VmaAllocator allocator, VmaAllocation allocation, ResourceMemoryCategory category) {
    if (allocator == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) { return; }

    const auto description = ResourceMemoryTrackerState::describeAllocation(allocator, allocation);
    const std::string tag = ResourceMemoryTrackerState::currentTag();
    std::lock_guard<std::mutex> lock(ResourceMemoryTrackerState::mutex());
    ResourceMemoryTrackerState::applyDescription(
        ResourceMemoryTrackerState::counters()[ResourceMemoryTrackerState::categoryIndex(category)], description, 1);
    ResourceMemoryTrackerState::applyDescription(ResourceMemoryTrackerState::tagCounters()[tag], description, 1);
    ResourceMemoryTrackerState::allocationTags()[allocation] = std::make_pair(tag, description);
}

void untrackVmaAllocation(VmaAllocator allocator, VmaAllocation allocation, ResourceMemoryCategory category) {
    if (allocator == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) { return; }

    const auto description = ResourceMemoryTrackerState::describeAllocation(allocator, allocation);
    std::lock_guard<std::mutex> lock(ResourceMemoryTrackerState::mutex());
    auto &counter = ResourceMemoryTrackerState::counters()[ResourceMemoryTrackerState::categoryIndex(category)];
    ResourceMemoryTrackerState::applyDescription(counter, description, -1);
    ResourceMemoryTrackerState::clampCounter(counter);

    auto &allocationTags = ResourceMemoryTrackerState::allocationTags();
    auto tagIt = allocationTags.find(allocation);
    if (tagIt != allocationTags.end()) {
        auto &tagCounters = ResourceMemoryTrackerState::tagCounters();
        auto detailIt = tagCounters.find(tagIt->second.first);
        if (detailIt != tagCounters.end()) {
            ResourceMemoryTrackerState::applyDescription(detailIt->second, tagIt->second.second, -1);
            ResourceMemoryTrackerState::clampCounter(detailIt->second);
        }
        allocationTags.erase(tagIt);
    }
}

void retagVmaAllocation(VmaAllocator allocator,
                        VmaAllocation allocation,
                        ResourceMemoryCategory oldCategory,
                        ResourceMemoryCategory newCategory) {
    if (oldCategory == newCategory || allocator == VK_NULL_HANDLE || allocation == VK_NULL_HANDLE) { return; }

    const auto description = ResourceMemoryTrackerState::describeAllocation(allocator, allocation);
    std::lock_guard<std::mutex> lock(ResourceMemoryTrackerState::mutex());
    auto &oldCounter = ResourceMemoryTrackerState::counters()[ResourceMemoryTrackerState::categoryIndex(oldCategory)];
    auto &newCounter = ResourceMemoryTrackerState::counters()[ResourceMemoryTrackerState::categoryIndex(newCategory)];
    ResourceMemoryTrackerState::applyDescription(oldCounter, description, -1);
    ResourceMemoryTrackerState::applyDescription(newCounter, description, 1);
    ResourceMemoryTrackerState::clampCounter(oldCounter);
}

ResourceMemorySnapshot getResourceMemorySnapshot() {
    std::lock_guard<std::mutex> lock(ResourceMemoryTrackerState::mutex());
    return ResourceMemorySnapshot{.categories = ResourceMemoryTrackerState::counters()};
}

void recordResourceMemoryProfilerMetrics(const char *phase) {
    const char *resolvedPhase = phase == nullptr ? "sample" : phase;
    const ResourceMemorySnapshot snapshot = getResourceMemorySnapshot();

    for (size_t i = 0; i < ResourceMemoryTrackerState::categoryCount; ++i) {
        const auto category = static_cast<ResourceMemoryCategory>(i);
        const char *categoryName = resourceMemoryCategoryName(category);
        const ResourceMemoryCounters &counter = snapshot.categories[i];

        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vram.", categoryName, "_bytes.", resolvedPhase),
            counter.deviceLocalBytes);
        MCVR_PROFILE_METRIC(ResourceMemoryTrackerState::metricName("memory.vulkan.host_visible.", categoryName,
                                                                   "_bytes.", resolvedPhase),
                            counter.hostVisibleBytes);
        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vulkan.allocation.", categoryName, "_bytes.", resolvedPhase),
            counter.allocationBytes);
        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vulkan.allocation.", categoryName, "_count.", resolvedPhase),
            counter.allocationCount);
    }

    const ResourceMemoryCounters total = snapshot.total();
    MCVR_PROFILE_METRIC(ResourceMemoryTrackerState::metricName("memory.vram.", "total", "_bytes.", resolvedPhase),
                        total.deviceLocalBytes);
    MCVR_PROFILE_METRIC(
        ResourceMemoryTrackerState::metricName("memory.vulkan.host_visible.", "total", "_bytes.", resolvedPhase),
        total.hostVisibleBytes);
    MCVR_PROFILE_METRIC(
        ResourceMemoryTrackerState::metricName("memory.vulkan.allocation.", "total", "_bytes.", resolvedPhase),
        total.allocationBytes);
    MCVR_PROFILE_METRIC(
        ResourceMemoryTrackerState::metricName("memory.vulkan.allocation.", "total", "_count.", resolvedPhase),
        total.allocationCount);

    for (const auto &[tag, counter] : ResourceMemoryTrackerState::tagCounters()) {
        const std::string safeTag = ResourceMemoryTrackerState::sanitizedMetricToken(tag);
        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vulkan.detail.", safeTag.c_str(), "_bytes.", resolvedPhase),
            counter.allocationBytes);
        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vram.detail.", safeTag.c_str(), "_bytes.", resolvedPhase),
            counter.deviceLocalBytes);
        MCVR_PROFILE_METRIC(ResourceMemoryTrackerState::metricName("memory.vulkan.detail.", safeTag.c_str(),
                                                                   "_host_visible_bytes.", resolvedPhase),
                            counter.hostVisibleBytes);
        MCVR_PROFILE_METRIC(
            ResourceMemoryTrackerState::metricName("memory.vulkan.detail.", safeTag.c_str(), "_count.", resolvedPhase),
            counter.allocationCount);
    }
}

ResourceMemoryTagScope::ResourceMemoryTagScope(std::string tag) {
    ResourceMemoryTrackerState::tagStack().emplace_back(std::move(tag));
}

ResourceMemoryTagScope::~ResourceMemoryTagScope() {
    auto &stack = ResourceMemoryTrackerState::tagStack();
    if (!active_ || stack.empty()) { return; }
    stack.pop_back();
}

}

#endif
