#pragma once

#include "core/render/build_geometry.hpp"

#include <jni.h>

#include <cstdint>
#include <memory>
#include <vector>

class JavaBuildBufferRef;

class JavaBuildBufferLease : public BuildBufferLease, public SharedObject<JavaBuildBufferLease> {
  public:
    JavaBuildBufferLease(JNIEnv *env, jobjectArray transferredBuffers);
    ~JavaBuildBufferLease() = default;

    bool empty() const override;
    bool valid(uint32_t index) const override;
    std::shared_ptr<void> retain(uint32_t index) const override;

  private:
    JavaVM *vm_ = nullptr;
    jmethodID closeMethod_ = nullptr;
    std::vector<std::shared_ptr<JavaBuildBufferRef>> buffers_;
};
