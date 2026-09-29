#pragma once

#include <cstdint>

#if defined(MCVR_ENABLE_PROFILER)

#    include <cstddef>
#    include <filesystem>
#    include <iosfwd>
#    include <string>

namespace mcvr::profiler {

class Profiler {
  public:
    static void configureOutputDirectory(const std::filesystem::path &folderPath);
    static void beginFrame(uint32_t swapchainFrameIndex);
    static void setFrameSwapchainIndex(uint32_t swapchainFrameIndex);
    static void endFrame();
    static void enter(const std::string &name);
    static void exit();
    static void metric(const std::string &name, int64_t value);
    static void flush();

  private:
    struct Node;
    struct Frame;
    struct ThreadState;
    struct State;

    static State &state();
    static ThreadState &threadState();
    static uint64_t nowNs();
    static uint64_t threadId();
    static void writeEscaped(std::ostream &out, const std::string &value);
    static void openStreamLocked(State &profilerState);
    static void ensureThreadFrameLocked(State &profilerState, ThreadState &currentThreadState);
    static void closeOpenNodesLocked(Frame &frame, uint64_t endNs);
    static uint64_t nodeEndNs(const Frame &frame, const Node &node);
    static uint64_t nodeDurationNs(const Frame &frame, const Node &node);
    static uint64_t childDurationNs(const Frame &frame, const Node &node);
    static void writeNodeJson(std::ostream &out, const Frame &frame, size_t nodeIndex, uint32_t depth);
    static void writeFrameLocked(State &profilerState, Frame &frame);
};

class ScopedScope {
  public:
    explicit ScopedScope(const std::string &name);
    ~ScopedScope();

    ScopedScope(const ScopedScope &) = delete;
    ScopedScope &operator=(const ScopedScope &) = delete;

  private:
    bool active_ = true;
};

}

#    define MCVR_PROFILE_CONCAT_IMPL(a, b) a##b
#    define MCVR_PROFILE_CONCAT(a, b) MCVR_PROFILE_CONCAT_IMPL(a, b)
#    define MCVR_PROFILE_SCOPE(name)                                                                                   \
        ::mcvr::profiler::ScopedScope MCVR_PROFILE_CONCAT(mcvrProfileScope_, __LINE__)(name)
#    define MCVR_PROFILE_BEGIN_FRAME(swapchainFrameIndex) ::mcvr::profiler::Profiler::beginFrame(swapchainFrameIndex)
#    define MCVR_PROFILE_SET_FRAME_SWAPCHAIN_INDEX(swapchainFrameIndex)                                                \
        ::mcvr::profiler::Profiler::setFrameSwapchainIndex(swapchainFrameIndex)
#    define MCVR_PROFILE_END_FRAME() ::mcvr::profiler::Profiler::endFrame()
#    define MCVR_PROFILE_METRIC(name, value) ::mcvr::profiler::Profiler::metric(name, static_cast<int64_t>(value))
#    define MCVR_PROFILE_CONFIGURE_OUTPUT(folderPath) ::mcvr::profiler::Profiler::configureOutputDirectory(folderPath)
#    define MCVR_PROFILE_FLUSH() ::mcvr::profiler::Profiler::flush()

#else

#    define MCVR_PROFILE_SCOPE(name) ((void)0)
#    define MCVR_PROFILE_BEGIN_FRAME(swapchainFrameIndex) ((void)0)
#    define MCVR_PROFILE_SET_FRAME_SWAPCHAIN_INDEX(swapchainFrameIndex) ((void)0)
#    define MCVR_PROFILE_END_FRAME() ((void)0)
#    define MCVR_PROFILE_METRIC(name, value) ((void)0)
#    define MCVR_PROFILE_CONFIGURE_OUTPUT(folderPath) ((void)0)
#    define MCVR_PROFILE_FLUSH() ((void)0)

#endif
