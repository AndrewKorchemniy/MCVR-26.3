#include "com_radiance_client_proxy_world_ChunkProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/middleware/java_build_buffer_lease.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"
#include "core/render/vista.hpp"

#include <exception>
#include <iostream>

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_initNative(
    JNIEnv *, jclass, jint chunkNum, jint sizeX, jint sizeY, jint sizeZ, jint bottomSectionCoord) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.initNative");
    Renderer::instance().world()->chunks()->reset(chunkNum, sizeX, sizeY, sizeZ, bottomSectionCoord);
    Renderer::instance().buffers()->resetWorldHistory();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_updateSectionPosNative(
    JNIEnv *, jclass, jint sectionX, jint sectionY, jint sectionZ) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.updateSectionPosNative");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->setChunkStorageSectionPos(glm::ivec3(sectionX, sectionY, sectionZ));
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingleNative(JNIEnv *env,
                                                                                           jclass,
                                                                                           jint originX,
                                                                                           jint originY,
                                                                                           jint originZ,
                                                                                           jlong index,
                                                                                           jint geometryCount,
                                                                                           jlong geometryTypes,
                                                                                           jlong geometryGroupNames,
                                                                                           jlong geometryTextures,
                                                                                           jlong vertexFormats,
                                                                                           jlong indexTypes,
                                                                                           jlong vertexCounts,
                                                                                           jlong vertexAddrs,
                                                                                           jlong indexAddrs,
                                                                                           jobjectArray buffers,
                                                                                           jboolean important,
                                                                                           jlong generation,
                                                                                           jint terrainCount,
                                                                                           jlong terrainAddress) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.rebuildSingle");
    try {
        auto lease = JavaBuildBufferLease::create(env, buffers);
        if (lease->empty()) { lease.reset(); }

        auto world = Renderer::instance().world();
        if (world == nullptr) return;
        world->chunks()->queueChunkBuild(ChunkBuildTask{
            .x = originX,
            .y = originY,
            .z = originZ,
            .generation = generation,
            .terrain = copyVistaTerrain(
                vistaTerrainView(terrainCount, reinterpret_cast<const VistaSpan *>(terrainAddress))),
            .id = index,
            .geometryCount = geometryCount,
            .geometryTypes = reinterpret_cast<int *>(geometryTypes),
            .geometryGroupNames = reinterpret_cast<const char **>(geometryGroupNames),
            .geometryTextures = reinterpret_cast<int *>(geometryTextures),
            .vertexFormats = reinterpret_cast<int *>(vertexFormats),
            .indexTypes = reinterpret_cast<int *>(indexTypes),
            .vertexCounts = reinterpret_cast<int *>(vertexCounts),
            .vertices = reinterpret_cast<vk::VertexFormat::ChunkPBRVertex **>(vertexAddrs),
            .indices = reinterpret_cast<void **>(indexAddrs),
            .transferredBufferLease = lease,
            .isImportant = static_cast<bool>(important),
        });
    } catch (const std::exception &exception) {
        if (env->ExceptionCheck()) { return; }
        jclass runtimeExceptionClass = env->FindClass("java/lang/RuntimeException");
        if (runtimeExceptionClass != nullptr) { env->ThrowNew(runtimeExceptionClass, exception.what()); }
    }
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_isChunkReady(JNIEnv *, jclass, jlong id) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.isChunkReady");
    auto world = Renderer::instance().world();
    if (world == nullptr)
        return false;
    else
        return world->chunks()->isChunkReady(id);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_relocateSingle(
    JNIEnv *, jclass, jlong index, jint originX, jint originY, jint originZ, jlong generation) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.relocateSingle");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->relocateChunk(index, originX, originY, originZ, generation);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_invalidateSingle(
    JNIEnv *env, jclass, jlong index, jint originX, jint originY, jint originZ, jlong generation,
    jint terrainCount, jlong terrainAddress) {
    MCVR_PROFILE_SCOPE("jni.ChunkProxy.invalidateSingle");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    try {
        world->chunks()->invalidateChunk(index, originX, originY, originZ, generation,
            copyVistaTerrain(vistaTerrainView(terrainCount, reinterpret_cast<const VistaSpan *>(terrainAddress))));
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) {
            auto type = env->FindClass("java/lang/IllegalArgumentException");
            if (type != nullptr) { env->ThrowNew(type, exception.what()); }
        }
    }
}
