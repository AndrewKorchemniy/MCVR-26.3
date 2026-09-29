#include "com_radiance_client_proxy_world_VistaProxy.h"
#include "core/middleware/jni_export.hpp"
#include "core/render/renderer.hpp"
#include "core/render/vista.hpp"
#include "core/render/world.hpp"

#include <cmath>
#include <stdexcept>

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_VistaProxy_reset(JNIEnv *, jclass, jlong version) {
    auto world = Renderer::instance().world();
    if (world != nullptr) { world->vista()->reset(version); }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_VistaProxy_rebuildSingleNative(
    JNIEnv *env, jclass, jlong worldVersion, jlong id, jint x, jint y, jint z, jint width, jint count, jlong address,
    jlong terrainId, jint terrainCount, jlong terrainAddress) {
    try {
        if (width < 0 || (width != 0 && (width < 16 || width % 16 != 0 || x % 16 != 0 || z % 16 != 0)) ||
            count <= 0 || count > 1048576 || count % 4 != 0 || address == 0) {
            throw std::invalid_argument("Invalid Vista quad buffer");
        }
        auto world = Renderer::instance().world();
        if (world == nullptr) { return false; }
        auto vertices = reinterpret_cast<const vk::VertexFormat::ChunkPBRVertex *>(address);
        VistaBuildTask task{
            .worldVersion = worldVersion, .id = id, .x = x, .y = y, .z = z, .width = width,
            .vertices = {vertices, vertices + count},
            .terrain = world->vista()->terrain(worldVersion, terrainId,
                vistaTerrainView(terrainCount, reinterpret_cast<const VistaSpan *>(terrainAddress)))};
        for (const auto &vertex : task.vertices) {
            if (!std::isfinite(vertex.posX) || !std::isfinite(vertex.posY) || !std::isfinite(vertex.posZ)) {
                throw std::invalid_argument("Non-finite Vista vertex");
            }
        }
        return world->vista()->queueVistaBuild(std::move(task));
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) {
            auto type = env->FindClass("java/lang/IllegalArgumentException");
            if (type != nullptr) { env->ThrowNew(type, exception.what()); }
        }
        return false;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_VistaProxy_updateObjectListNative(
    JNIEnv *env, jclass, jlong worldVersion, jlongArray ids) {
    try {
        if (ids == nullptr) { return; }
        auto world = Renderer::instance().world();
        if (world == nullptr) { return; }
        const jsize count = env->GetArrayLength(ids);
        std::vector<int64_t> copy(count);
        env->GetLongArrayRegion(ids, 0, count, reinterpret_cast<jlong *>(copy.data()));
        if (!env->ExceptionCheck()) { world->vista()->updateObjectList(worldVersion, std::move(copy)); }
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) {
            auto type = env->FindClass("java/lang/RuntimeException");
            if (type != nullptr) { env->ThrowNew(type, exception.what()); }
        }
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_VistaProxy_moveObject(
    JNIEnv *, jclass, jlong worldVersion, jlong id, jdouble x, jdouble y, jdouble z) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) { return; }
    auto world = Renderer::instance().world();
    if (world != nullptr) { world->vista()->moveObject(worldVersion, id, {x, y, z}); }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_VistaProxy_isReady(
    JNIEnv *, jclass, jlong worldVersion, jlong id) {
    auto world = Renderer::instance().world();
    return world != nullptr && world->vista()->isReady(worldVersion, id);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_VistaProxy_updateRenderListNative(JNIEnv *env,
                                                                                              jclass,
                                                                                              jlong worldVersion,
                                                                                              jlongArray ids) {
    try {
        if (ids == nullptr) { return; }
        auto world = Renderer::instance().world();
        if (world == nullptr) { return; }
        const jsize count = env->GetArrayLength(ids);
        std::vector<int64_t> copy(count);
        env->GetLongArrayRegion(ids, 0, count, reinterpret_cast<jlong *>(copy.data()));
        if (!env->ExceptionCheck()) { world->vista()->updateRenderList(worldVersion, std::move(copy)); }
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) {
            auto type = env->FindClass("java/lang/RuntimeException");
            if (type != nullptr) { env->ThrowNew(type, exception.what()); }
        }
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_VistaProxy_invalidateSingle(JNIEnv *,
                                                                                        jclass,
                                                                                        jlong worldVersion,
                                                                                        jlong id) {
    auto world = Renderer::instance().world();
    if (world != nullptr) { world->vista()->invalidateVista(worldVersion, id); }
}
