#include "com_radiance_client_proxy_world_EntityProxy.h"
#include "core/middleware/jni_export.hpp"

#include "core/middleware/java_build_buffer_lease.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/entities.hpp"
#include "core/render/renderer.hpp"

#include <exception>
#include <cstring>
#include "core/vulkan/vertex.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_EntityProxy_queueBuildNative(JNIEnv *env,
                                                                                         jclass,
                                                                                         jfloat lineWidth,
                                                                                         jint coordinate,
                                                                                         jboolean normalOffset,
                                                                                         jint size,
                                                                                         jlong entityKeys,
                                                                                         jlong entityPosXs,
                                                                                         jlong entityPosYs,
                                                                                         jlong entityPosZs,
                                                                                         jlong entityRayTracingFlags,
                                                                                         jlong entityPostRenderFlags,
                                                                                         jlong entityPrebuiltBLASs,
                                                                                         jlong entityTransforms,
                                                                                         jlong entityMaterials,
                                                                                         jlong entityPosts,
                                                                                         jlong entityLayerCounts,
                                                                                         jlong geometryTypes,
                                                                                         jlong geometryGroupNames,
                                                                                         jlong geometryContentNames,
                                                                                         jlong geometryTextures,
                                                                                         jlong geometryNormalOffsets,
                                                                                         jlong vertexFormats,
                                                                                         jlong indexFormats,
                                                                                         jlong vertexCounts,
                                                                                         jlong vertices,
                                                                                         jobjectArray buffers) {
    MCVR_PROFILE_SCOPE("jni.EntityProxy.queueBuild");
    try {
        auto lease = JavaBuildBufferLease::create(env, buffers);
        if (lease->empty()) { lease.reset(); }

        auto world = Renderer::instance().world();
        if (world == nullptr) return;
        world->entities()->queueBuild(EntitiesBuildTask{
            .lineWidth = lineWidth,
            .coordinate = static_cast<World::Coordinates>(coordinate),
            .normalOffset = static_cast<bool>(normalOffset),
            .entityCount = size,
            .entityKeys = reinterpret_cast<int64_t *>(entityKeys),
            .entityXs = reinterpret_cast<double *>(entityPosXs),
            .entityYs = reinterpret_cast<double *>(entityPosYs),
            .entityZs = reinterpret_cast<double *>(entityPosZs),
            .entityRayTracingFlags = reinterpret_cast<int *>(entityRayTracingFlags),
            .entityPostRenderFlags = reinterpret_cast<int *>(entityPostRenderFlags),
            .entityPrebuiltBLASs = reinterpret_cast<int *>(entityPrebuiltBLASs),
            .entityTransforms = reinterpret_cast<const float *>(entityTransforms),
            .entityMaterials = reinterpret_cast<const vk::VertexFormat::EntityInstanceMaterial *>(entityMaterials),
            .entityPosts = reinterpret_cast<int *>(entityPosts),
            .entityGeometryCounts = reinterpret_cast<int *>(entityLayerCounts),
            .geometryTypes = reinterpret_cast<int *>(geometryTypes),
            .geometryGroupNames = reinterpret_cast<const char **>(geometryGroupNames),
            .geometryContentNames = reinterpret_cast<const char **>(geometryContentNames),
            .geometryTextures = reinterpret_cast<int *>(geometryTextures),
            .geometryNormalOffsets = reinterpret_cast<float *>(geometryNormalOffsets),
            .vertexFormats = reinterpret_cast<int *>(vertexFormats),
            .indexFormats = reinterpret_cast<int *>(indexFormats),
            .vertexCounts = reinterpret_cast<int *>(vertexCounts),
            .vertices = reinterpret_cast<void **>(vertices),
            .transferredBufferLease = lease,
        });
    } catch (const std::exception &exception) {
        if (env->ExceptionCheck()) { return; }
        jclass runtimeExceptionClass = env->FindClass("java/lang/RuntimeException");
        if (runtimeExceptionClass != nullptr) { env->ThrowNew(runtimeExceptionClass, exception.what()); }
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_EntityProxy_buildNative(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.EntityProxy.build");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->entities()->build();
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_world_EntityProxy_registerPrebuiltGeometryNative(
    JNIEnv *env, jclass, jobject data) {
    try {
        auto *cursor = static_cast<const uint8_t *>(env->GetDirectBufferAddress(data));
        const jlong capacity = env->GetDirectBufferCapacity(data);
        if (cursor == nullptr || capacity < 4) { throw std::runtime_error("invalid entity geometry buffer"); }
        const auto *end = cursor + capacity;
        auto read = [&](void *output, size_t size) {
            if (size > static_cast<size_t>(end - cursor)) { throw std::runtime_error("truncated entity geometry"); }
            std::memcpy(output, cursor, size);
            cursor += size;
        };
        uint32_t count;
        read(&count, sizeof(count));
        if (count == 0 || count > 4096) { throw std::runtime_error("invalid entity geometry count"); }
        std::vector<BuildGeometry> geometries;
        std::vector<std::array<float, 16>> programVertices;
        std::vector<World::GeometryTypes> types;
        std::vector<std::string> groups, names;
        for (uint32_t g = 0; g < count; ++g) {
            uint32_t header[4];
            read(header, sizeof(header));
            const auto [vertexCount, indexCount, texture, flags] = std::tuple{header[0], header[1], header[2], header[3]};
            if (vertexCount == 0 || indexCount == 0 || indexCount % 3 != 0 ||
                static_cast<uint64_t>(vertexCount) * 36 + static_cast<uint64_t>(indexCount) * 4 > end - cursor) {
                throw std::runtime_error("invalid entity mesh size");
            }
            std::vector<vk::VertexFormat::EntityPBRVertex> vertices(vertexCount);
            for (auto &vertex : vertices) {
                vk::VertexFormat::PositionColorTexOverlayLightNormal source;
                static_assert(sizeof(source) == 36);
                read(&source, sizeof(source));
                const glm::vec4 color = vk::Vertex::unpackColorRGBA8(source.color);
                const glm::ivec2 overlay = vk::Vertex::unpackU16x2(source.uv1);
                const glm::ivec2 light = vk::Vertex::unpackU16x2(source.uv2);
                programVertices.push_back({source.position.x, source.position.y, source.position.z,
                    color.r, color.g, color.b, color.a, source.uv0.x, source.uv0.y,
                    float(overlay.x), float(overlay.y), light.x / 256.0f, light.y / 256.0f,
                    std::max(-1.0f, int8_t(source.normal & 255) / 127.0f),
                    std::max(-1.0f, int8_t((source.normal >> 8) & 255) / 127.0f),
                    std::max(-1.0f, int8_t((source.normal >> 16) & 255) / 127.0f)});
                vertex.pos = source.position;
                vertex.textureID = texture;
                vk::Vertex::setColorLayer(vertex, source.color);
                vk::Vertex::setTexture(vertex, source.uv0);
                vk::Vertex::setNormalBytes(vertex, source.normal);
                if (flags & 4u) { vk::Vertex::setOverlay(vertex, source.uv1); }
                if (flags & 8u) { vk::Vertex::setLight(vertex, source.uv2); }
                vertex.packedData = vk::Vertex::setAlphaMode(vertex.packedData, (flags >> 16) & 15u);
            }
            std::vector<uint32_t> indices(indexCount);
            read(indices.data(), indices.size() * sizeof(uint32_t));
            for (uint32_t index : indices) {
                if (index >= vertexCount) { throw std::runtime_error("entity mesh index out of range"); }
            }
            geometries.push_back(BuildGeometry::materializedPBR(std::move(vertices), std::move(indices)));
            types.push_back(((flags >> 16) & 15u) == 0 ? World::WORLD_SOLID : World::WORLD_TRANSPARENT);
            groups.emplace_back(((flags >> 16) & 15u) == 0 ? "solid" : (((flags >> 16) & 15u) == 2 || ((flags >> 16) & 15u) >= 7) ? "translucent" : "cutout_mipped");
            names.push_back(std::to_string(g));
        }
        auto world = Renderer::instance().world();
        if (world == nullptr) { throw std::runtime_error("no world for entity geometry"); }
        auto input = EntityBuildInput::create(0, 0, 0, 0, 1, 0, -1,
            World::WORLD, count, std::move(types), std::move(groups), std::move(names), std::move(geometries));
        input->programVertices = std::move(programVertices);
        return world->entities()->registerPrebuiltGeometry(input);
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) { env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), exception.what()); }
        return -1;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_EntityProxy_releasePrebuiltGeometryNative(
    JNIEnv *, jclass, jint handle) {
    auto world = Renderer::instance().world();
    if (world != nullptr) { world->entities()->releasePrebuiltGeometry(handle); }
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_world_EntityProxy_updatePrebuiltGeometryNative(
    JNIEnv *env, jclass, jint previous, jint format, jintArray geometryTypes, jobjectArray buffers) {
    try {
        if (buffers == nullptr || geometryTypes == nullptr) throw std::runtime_error("missing entity mesh data");
        const jsize count = env->GetArrayLength(buffers);
        if (count != env->GetArrayLength(geometryTypes) || count > 4096)
            throw std::runtime_error("invalid entity mesh count");
        const size_t stride = format == World::CHUNK_PBR_TRIANGLE ? sizeof(vk::VertexFormat::ChunkPBRVertex) :
            format == World::ENTITY_PBR_TRIANGLE ? sizeof(vk::VertexFormat::EntityPBRVertex) : 0;
        if (!stride) throw std::runtime_error("unsupported prebuilt vertex format");
        std::vector<jint> rawTypes(count);
        env->GetIntArrayRegion(geometryTypes, 0, count, rawTypes.data());
        std::vector<BuildGeometry> geometries;
        std::vector<World::GeometryTypes> types;
        std::vector<std::string> groups, names;
        for (jsize g = 0; g < count; ++g) {
            jobject buffer = env->GetObjectArrayElement(buffers, g);
            auto *data = static_cast<const uint8_t *>(env->GetDirectBufferAddress(buffer));
            const jlong bytes = env->GetDirectBufferCapacity(buffer);
            env->DeleteLocalRef(buffer);
            if (!data || bytes <= 0 || bytes % (stride * 4) != 0 || bytes / stride > UINT32_MAX)
                throw std::runtime_error("invalid prebuilt quad buffer");
            if (rawTypes[g] != World::WORLD_SOLID && rawTypes[g] != World::WORLD_TRANSPARENT)
                throw std::runtime_error("invalid prebuilt geometry type");
            std::vector<vk::VertexFormat::EntityPBRVertex> vertices(bytes / stride);
            for (size_t i = 0; i < vertices.size(); ++i) {
                auto &vertex = vertices[i];
                if (format == World::ENTITY_PBR_TRIANGLE) {
                    std::memcpy(&vertex, data + i * stride, stride);
                } else {
                    vk::VertexFormat::ChunkPBRVertex source;
                    std::memcpy(&source, data + i * stride, stride);
                    vertex.pos = vk::Vertex::position(source);
                    vertex.normalOct = source.normalOct;
                    vertex.textureUV = source.textureUV;
                    vertex.colorRGBA8 = source.colorRGBA8;
                    vertex.lightUV = source.lightUV;
                    vertex.textureID = source.textureID;
                    vertex.packedData = source.packedData;
                }
                vk::Vertex::setCoordinate(vertex, World::WORLD);
                if (!std::isfinite(vertex.pos.x) || !std::isfinite(vertex.pos.y) || !std::isfinite(vertex.pos.z))
                    throw std::runtime_error("non-finite prebuilt vertex");
            }
            std::vector<uint32_t> indices(vertices.size() / 4 * 6);
            for (uint32_t i = 0; i < indices.size(); ++i) indices[i] = BuildGeometry::generatedQuadIndex(i);
            geometries.push_back(BuildGeometry::materializedPBR(std::move(vertices), std::move(indices)));
            types.push_back(static_cast<World::GeometryTypes>(rawTypes[g]));
            groups.push_back(rawTypes[g] == World::WORLD_SOLID ? "solid" : "translucent");
            names.push_back(std::to_string(g));
        }
        auto world = Renderer::instance().world();
        if (!world) throw std::runtime_error("no world for entity geometry");
        if (geometries.empty()) {
            world->entities()->releasePrebuiltGeometry(previous);
            return -1;
        }
        auto input = EntityBuildInput::create(0, 0, 0, 0, 1, 0, -1, World::WORLD, count,
            std::move(types), std::move(groups), std::move(names), std::move(geometries));
        return world->entities()->registerPrebuiltGeometry(input, previous);
    } catch (const std::exception &exception) {
        if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), exception.what());
        return -1;
    }
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_world_EntityProxy_registerVertexShaderNative(
    JNIEnv *env, jclass, jstring source) {
    const char *text = env->GetStringUTFChars(source, nullptr);
    if (!text) return -1;
    try {
        std::string code(text);
        env->ReleaseStringUTFChars(source, text);
        text = nullptr;
        auto world = Renderer::instance().world();
        if (!world) throw std::runtime_error("no world for entity program");
        return world->entities()->registerVertexShader(code);
    } catch (const std::exception &error) {
        if (text) env->ReleaseStringUTFChars(source, text);
        if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), error.what());
        return -1;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_EntityProxy_releaseVertexShaderNative(
    JNIEnv *, jclass, jint handle) {
    auto world = Renderer::instance().world();
    if (world) world->entities()->releaseVertexShader(handle);
}

extern "C" JNIEXPORT jobjectArray JNICALL Java_com_radiance_client_proxy_world_EntityProxy_queueVertexShaderInstancesNative(
    JNIEnv *env, jclass, jobject data, jobjectArray uniforms) {
    MCVR_PROFILE_SCOPE("jni.EntityProxy.queueVertexShaderInstances");
    try {
        auto world = Renderer::instance().world();
        if (!world) throw std::runtime_error("no world for entity program instances");
        auto *cursor = static_cast<const uint8_t *>(env->GetDirectBufferAddress(data));
        jlong size = env->GetDirectBufferCapacity(data);
        if (!cursor || size < 4) throw std::runtime_error("invalid entity instance buffer");
        const uint8_t *end = cursor + size;
        auto read = [&](void *target, size_t bytes) {
            if (bytes > size_t(end - cursor)) throw std::runtime_error("truncated entity instance buffer");
            std::memcpy(target, cursor, bytes); cursor += bytes;
        };
        uint32_t count;
        read(&count, 4);
        if (count > size_t(end - cursor) / 152) throw std::runtime_error("invalid entity instance count");
        thread_local std::vector<Entities::VertexShaderInstance> instances;
        instances.resize(count);
        for (auto &instance : instances) {
            read(&instance.key, 8); read(&instance.owner, 8);
            read(&instance.geometry, 4); read(&instance.program, 4);
            read(&instance.x, 8); read(&instance.y, 8); read(&instance.z, 8);
            read(&instance.pose, 64); read(&instance.normal, 36);
            uint32_t bytes;
            read(&bytes, 4);
            if (bytes > size_t(end - cursor)) throw std::runtime_error("truncated entity program input");
            instance.data = {cursor, bytes}; cursor += bytes;
        }
        if (cursor != end) throw std::runtime_error("trailing entity instance data");
        Entities::VertexShaderEvaluator::Bindings bindings{};
        jsize uniformCount = env->GetArrayLength(uniforms);
        if (uniformCount != 5) throw std::runtime_error("invalid entity uniform count");
        for (jsize i = 0; i < uniformCount; ++i) {
            jobject buffer = env->GetObjectArrayElement(uniforms, i);
            auto *pointer = static_cast<const uint8_t *>(env->GetDirectBufferAddress(buffer));
            jlong bytes = env->GetDirectBufferCapacity(buffer);
            if (!pointer || bytes < 0) throw std::runtime_error("invalid entity uniform buffer");
            bindings[8 + i] = {pointer, size_t(bytes)};
            env->DeleteLocalRef(buffer);
        }
        auto failures = world->entities()->queueVertexShaderInstances(instances, bindings);
        jobjectArray output = env->NewObjectArray(failures.size(), env->FindClass("java/lang/String"), nullptr);
        for (size_t i = 0; i < failures.size(); ++i) {
            std::string message = std::to_string(failures[i].first) + "\t" + failures[i].second;
            jstring string = env->NewStringUTF(message.c_str());
            env->SetObjectArrayElement(output, i, string);
            env->DeleteLocalRef(string);
        }
        return output;
    } catch (const std::exception &error) {
        if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), error.what());
        return nullptr;
    }
}
