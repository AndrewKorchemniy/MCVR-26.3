#include "core/middleware/java_build_buffer_lease.hpp"

#include "core/profiler/profiler.hpp"

#include <stdexcept>

class JavaBuildBufferRef {
  public:
    JavaBuildBufferRef(JavaVM *vm, jobject buffer, jmethodID closeMethod)
        : vm_(vm), buffer_(buffer), closeMethod_(closeMethod) {}

    void discardGlobalRef(JNIEnv *env) {
        MCVR_PROFILE_SCOPE("native.JavaBuildBufferRef.discardGlobalRef");
        if (env != nullptr && buffer_ != nullptr) {
            env->DeleteGlobalRef(buffer_);
            buffer_ = nullptr;
        }
    }

    ~JavaBuildBufferRef() {
        MCVR_PROFILE_SCOPE("native.JavaBuildBufferRef.destroy");
        if (vm_ == nullptr || buffer_ == nullptr) { return; }

        JNIEnv *env = nullptr;
        bool attached = false;
        jint envResult = vm_->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
        if (envResult == JNI_EDETACHED) {
#ifdef __ANDROID__
            if (vm_->AttachCurrentThread(&env, nullptr) != JNI_OK) { return; }
#else
            if (vm_->AttachCurrentThread(reinterpret_cast<void **>(&env), nullptr) != JNI_OK) { return; }
#endif
            attached = true;
        } else if (envResult != JNI_OK || env == nullptr) {
            return;
        }

        if (closeMethod_ != nullptr) {
            env->CallVoidMethod(buffer_, closeMethod_);
            if (env->ExceptionCheck()) { env->ExceptionClear(); }
        }
        env->DeleteGlobalRef(buffer_);
        buffer_ = nullptr;

        if (attached) { vm_->DetachCurrentThread(); }
    }

  private:
    JavaVM *vm_ = nullptr;
    jobject buffer_ = nullptr;
    jmethodID closeMethod_ = nullptr;
};

JavaBuildBufferLease::JavaBuildBufferLease(JNIEnv *env, jobjectArray transferredBuffers) {
    MCVR_PROFILE_SCOPE("native.JavaBuildBufferLease.init");
    if (env == nullptr || transferredBuffers == nullptr) { return; }
    try {
        if (env->GetJavaVM(&vm_) != JNI_OK || vm_ == nullptr) { throw std::runtime_error("Unable to capture JavaVM"); }

        jclass localTransferClass = env->FindClass("com/radiance/client/vertex/TransferredBuildBuffer");
        if (localTransferClass == nullptr) { throw std::runtime_error("Unable to find TransferredBuildBuffer class"); }

        closeMethod_ = env->GetMethodID(localTransferClass, "close", "()V");
        env->DeleteLocalRef(localTransferClass);
        if (closeMethod_ == nullptr) { throw std::runtime_error("Unable to find TransferredBuildBuffer.close"); }

        const jsize count = env->GetArrayLength(transferredBuffers);
        MCVR_PROFILE_METRIC("java_build_buffer_lease.count", count);
        buffers_.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; ++i) {
            jobject local = env->GetObjectArrayElement(transferredBuffers, i);
            if (local == nullptr) {
                buffers_.push_back(nullptr);
                continue;
            }

            jobject global = env->NewGlobalRef(local);
            env->DeleteLocalRef(local);
            if (global == nullptr) { throw std::runtime_error("Unable to retain transferred build buffer"); }
            buffers_.push_back(std::make_shared<JavaBuildBufferRef>(vm_, global, closeMethod_));
        }
    } catch (...) {
        for (const auto &buffer : buffers_) {
            if (buffer != nullptr) { buffer->discardGlobalRef(env); }
        }
        buffers_.clear();
        closeMethod_ = nullptr;
        throw;
    }
}

bool JavaBuildBufferLease::empty() const {
    MCVR_PROFILE_SCOPE("native.JavaBuildBufferLease.empty");
    for (const auto &buffer : buffers_) {
        if (buffer != nullptr) { return false; }
    }
    return true;
}

bool JavaBuildBufferLease::valid(uint32_t index) const {
    MCVR_PROFILE_SCOPE("native.JavaBuildBufferLease.valid");
    return index < buffers_.size() && buffers_[index] != nullptr;
}

std::shared_ptr<void> JavaBuildBufferLease::retain(uint32_t index) const {
    MCVR_PROFILE_SCOPE("native.JavaBuildBufferLease.retain");
    if (!valid(index)) { return nullptr; }
    return buffers_[index];
}
