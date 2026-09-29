#include "core/profiler/profiler.hpp"

#if defined(MCVR_ENABLE_PROFILER)

#    include <chrono>
#    include <fstream>
#    include <iomanip>
#    include <iostream>
#    include <map>
#    include <memory>
#    include <mutex>
#    include <thread>
#    include <utility>
#    include <vector>

namespace mcvr::profiler {

struct Profiler::Node {
    uint64_t id = 0;
    uint64_t parent = 0;
    size_t parentIndex = SIZE_MAX;
    std::string name;
    uint64_t thread = 0;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    std::vector<size_t> children;
    std::vector<std::pair<std::string, int64_t>> metrics;
};

struct Profiler::Frame {
    uint64_t index = 0;
    uint32_t swapchainFrameIndex = UINT32_MAX;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    std::vector<Node> nodes;
};

struct Profiler::ThreadState {
    uint64_t frameIndex = UINT64_MAX;
    std::vector<size_t> stack;
};

struct Profiler::State {
    std::mutex mutex;
    std::filesystem::path outputPath;
    std::ofstream stream;
    bool metadataWritten = false;
    uint64_t nextFrameIndex = 0;
    uint64_t nextNodeId = 1;
    std::unique_ptr<Frame> frame;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

Profiler::State &Profiler::state() {
    static State profilerState;
    return profilerState;
}

Profiler::ThreadState &Profiler::threadState() {
    thread_local ThreadState currentThreadState;
    return currentThreadState;
}

uint64_t Profiler::nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - state().start).count());
}

uint64_t Profiler::threadId() {
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

void Profiler::writeEscaped(std::ostream &out, const std::string &value) {
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec
                        << std::setfill(' ');
                } else {
                    out << ch;
                }
                break;
        }
    }
    out << '"';
}

void Profiler::openStreamLocked(State &profilerState) {
    if (profilerState.stream.is_open() || profilerState.outputPath.empty()) { return; }

    std::filesystem::create_directories(profilerState.outputPath.parent_path());
    profilerState.stream.open(profilerState.outputPath, std::ios::out | std::ios::trunc);
    if (!profilerState.stream.is_open()) {
        std::cerr << "[MCVR Profiler] Failed to open trace file: " << profilerState.outputPath << std::endl;
        return;
    }

    if (!profilerState.metadataWritten) {
        profilerState.stream
            << "{\"type\":\"metadata\",\"version\":1,\"time_unit\":\"ns\",\"format\":\"mcvr-profiler-jsonl\"}\n";
        profilerState.metadataWritten = true;
    }
}

void Profiler::ensureThreadFrameLocked(State &profilerState, ThreadState &currentThreadState) {
    if (profilerState.frame == nullptr) {
        currentThreadState.stack.clear();
        currentThreadState.frameIndex = UINT64_MAX;
        return;
    }

    if (currentThreadState.frameIndex == profilerState.frame->index && !currentThreadState.stack.empty()) { return; }

    currentThreadState.frameIndex = profilerState.frame->index;
    currentThreadState.stack.clear();
    currentThreadState.stack.push_back(0);
}

void Profiler::closeOpenNodesLocked(Frame &frame, uint64_t endNs) {
    for (Node &node : frame.nodes) {
        if (node.endNs == 0 || node.endNs < node.startNs) { node.endNs = endNs; }
    }
}

uint64_t Profiler::nodeEndNs(const Frame &frame, const Node &node) {
    return node.endNs == 0 ? frame.endNs : node.endNs;
}

uint64_t Profiler::nodeDurationNs(const Frame &frame, const Node &node) {
    const uint64_t endNs = nodeEndNs(frame, node);
    return endNs > node.startNs ? endNs - node.startNs : 0;
}

uint64_t Profiler::childDurationNs(const Frame &frame, const Node &node) {
    uint64_t durationNs = 0;
    for (size_t childIndex : node.children) { durationNs += nodeDurationNs(frame, frame.nodes[childIndex]); }
    return durationNs;
}

void Profiler::writeNodeJson(std::ostream &out, const Frame &frame, size_t nodeIndex, uint32_t depth) {
    const Node &node = frame.nodes[nodeIndex];
    const uint64_t endNs = nodeEndNs(frame, node);
    const uint64_t durationNs = nodeDurationNs(frame, node);
    const uint64_t childrenDurationNs = childDurationNs(frame, node);
    const uint64_t exclusiveDurationNs = durationNs > childrenDurationNs ? durationNs - childrenDurationNs : 0;

    out << "{\"id\":" << node.id << ",\"parent\":" << node.parent << ",\"name\":";
    writeEscaped(out, node.name);
    out << ",\"parent_name\":";
    if (nodeIndex == 0 || node.parent == 0 || node.parentIndex >= frame.nodes.size()) {
        out << "null";
    } else {
        writeEscaped(out, frame.nodes[node.parentIndex].name);
    }
    out << ",\"thread\":" << node.thread << ",\"start_ns\":" << node.startNs << ",\"end_ns\":" << endNs
        << ",\"duration_ns\":" << durationNs << ",\"child_duration_ns\":" << childrenDurationNs
        << ",\"exclusive_duration_ns\":" << exclusiveDurationNs << ",\"depth\":" << depth
        << ",\"child_count\":" << node.children.size();
    if (!node.metrics.empty()) {
        out << ",\"metrics\":{";
        for (size_t i = 0; i < node.metrics.size(); ++i) {
            if (i > 0) { out << ','; }
            writeEscaped(out, node.metrics[i].first);
            out << ':' << node.metrics[i].second;
        }
        out << '}';
    }
    out << ",\"children\":[";
    for (size_t i = 0; i < node.children.size(); ++i) {
        if (i > 0) { out << ','; }
        writeNodeJson(out, frame, node.children[i], depth + 1);
    }
    out << "]}";
}

void Profiler::writeFrameLocked(State &profilerState, Frame &frame) {
    openStreamLocked(profilerState);
    if (!profilerState.stream.is_open() || frame.nodes.empty()) { return; }

    std::map<std::string, uint64_t> jniCallCounts;
    uint64_t totalJniCalls = 0;
    for (const Node &node : frame.nodes) {
        if (!node.name.starts_with("jni.")) { continue; }

        ++totalJniCalls;
        ++jniCallCounts[node.name];
    }
    if (totalJniCalls != 0) {
        Node &root = frame.nodes.front();
        root.metrics.emplace_back("jni.call_count.total", static_cast<int64_t>(totalJniCalls));
        root.metrics.emplace_back("jni.call_count.unique_methods", static_cast<int64_t>(jniCallCounts.size()));
        for (const auto &[scopeName, callCount] : jniCallCounts) {
            root.metrics.emplace_back("jni.call_count.by_method." + scopeName.substr(4),
                                      static_cast<int64_t>(callCount));
        }
    }

    const uint64_t durationNs = frame.endNs > frame.startNs ? frame.endNs - frame.startNs : 0;
    profilerState.stream << "{\"type\":\"frame\",\"frame\":" << frame.index << ",\"swapchain_frame_index\":";
    if (frame.swapchainFrameIndex == UINT32_MAX) {
        profilerState.stream << "null";
    } else {
        profilerState.stream << frame.swapchainFrameIndex;
    }
    profilerState.stream << ",\"start_ns\":" << frame.startNs << ",\"end_ns\":" << frame.endNs
                         << ",\"duration_ns\":" << durationNs << ",\"root\":";
    writeNodeJson(profilerState.stream, frame, 0, 0);
    profilerState.stream << "}\n";
    profilerState.stream.flush();
}

void Profiler::configureOutputDirectory(const std::filesystem::path &folderPath) {
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    profilerState.outputPath = folderPath / "profiler" / "mcvr_profiler_trace.jsonl";
    openStreamLocked(profilerState);
}

void Profiler::beginFrame(uint32_t swapchainFrameIndex) {
    const uint64_t startNs = nowNs();
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);

    if (profilerState.frame != nullptr) {
        profilerState.frame->endNs = startNs;
        closeOpenNodesLocked(*profilerState.frame, startNs);
        writeFrameLocked(profilerState, *profilerState.frame);
    }

    auto frame = std::make_unique<Frame>();
    frame->index = profilerState.nextFrameIndex++;
    frame->swapchainFrameIndex = swapchainFrameIndex;
    frame->startNs = startNs;
    frame->nodes.push_back(Node{
        .id = profilerState.nextNodeId++,
        .parent = 0,
        .parentIndex = SIZE_MAX,
        .name = "frame",
        .thread = threadId(),
        .startNs = startNs,
        .endNs = 0,
        .children = {},
    });
    profilerState.frame = std::move(frame);

    ThreadState &currentThreadState = threadState();
    currentThreadState.frameIndex = profilerState.frame->index;
    currentThreadState.stack.clear();
    currentThreadState.stack.push_back(0);
}

void Profiler::setFrameSwapchainIndex(uint32_t swapchainFrameIndex) {
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    if (profilerState.frame != nullptr) { profilerState.frame->swapchainFrameIndex = swapchainFrameIndex; }
}

void Profiler::endFrame() {
    const uint64_t endNs = nowNs();
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    if (profilerState.frame == nullptr) { return; }

    profilerState.frame->endNs = endNs;
    closeOpenNodesLocked(*profilerState.frame, endNs);
    writeFrameLocked(profilerState, *profilerState.frame);
    profilerState.frame.reset();

    ThreadState &currentThreadState = threadState();
    currentThreadState.stack.clear();
    currentThreadState.frameIndex = UINT64_MAX;
}

void Profiler::enter(const std::string &name) {
    const uint64_t startNs = nowNs();
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    if (profilerState.frame == nullptr) { return; }

    ThreadState &currentThreadState = threadState();
    ensureThreadFrameLocked(profilerState, currentThreadState);
    const size_t parentIndex = currentThreadState.stack.empty() ? 0 : currentThreadState.stack.back();
    const uint64_t parentId = profilerState.frame->nodes[parentIndex].id;
    const size_t nodeIndex = profilerState.frame->nodes.size();
    profilerState.frame->nodes.push_back(Node{
        .id = profilerState.nextNodeId++,
        .parent = parentId,
        .parentIndex = parentIndex,
        .name = name,
        .thread = threadId(),
        .startNs = startNs,
        .endNs = 0,
        .children = {},
    });
    profilerState.frame->nodes[parentIndex].children.push_back(nodeIndex);
    currentThreadState.stack.push_back(nodeIndex);
}

void Profiler::exit() {
    const uint64_t endNs = nowNs();
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    ThreadState &currentThreadState = threadState();
    if (profilerState.frame == nullptr || currentThreadState.frameIndex != profilerState.frame->index ||
        currentThreadState.stack.size() <= 1) {
        currentThreadState.stack.clear();
        currentThreadState.frameIndex = UINT64_MAX;
        return;
    }

    const size_t nodeIndex = currentThreadState.stack.back();
    currentThreadState.stack.pop_back();
    if (nodeIndex < profilerState.frame->nodes.size() && profilerState.frame->nodes[nodeIndex].endNs == 0) {
        profilerState.frame->nodes[nodeIndex].endNs = endNs;
    }
}

void Profiler::metric(const std::string &name, int64_t value) {
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    if (profilerState.frame == nullptr) { return; }

    ThreadState &currentThreadState = threadState();
    ensureThreadFrameLocked(profilerState, currentThreadState);
    const size_t nodeIndex = currentThreadState.stack.empty() ? 0 : currentThreadState.stack.back();
    if (nodeIndex < profilerState.frame->nodes.size()) {
        profilerState.frame->nodes[nodeIndex].metrics.emplace_back(name, value);
    }
}

void Profiler::flush() {
    State &profilerState = state();
    std::lock_guard<std::mutex> lock(profilerState.mutex);
    if (profilerState.stream.is_open()) { profilerState.stream.flush(); }
}

ScopedScope::ScopedScope(const std::string &name) {
    Profiler::enter(name);
}

ScopedScope::~ScopedScope() {
    if (active_) { Profiler::exit(); }
}

}

#endif
