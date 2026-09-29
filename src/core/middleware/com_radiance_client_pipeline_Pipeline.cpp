#include "com_radiance_client_pipeline_Pipeline.h"
#include "core/middleware/jni_export.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/world/main_render/main_render_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <vector>

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_buildNative(JNIEnv *, jclass, jlong paramsLongPtr) {
    MCVR_PROFILE_SCOPE("jni.Pipeline.buildNative");
    WorldPipelineBuildParams *params = reinterpret_cast<WorldPipelineBuildParams *>(paramsLongPtr);
    auto pipeline = Renderer::instance().framework()->pipeline();
    if (pipeline != nullptr) Renderer::instance().framework()->pipeline()->buildWorldPipelineBlueprint(params);
}

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_collectNativeModules(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.Pipeline.collectNativeModules");
    Pipeline::collectWorldModules();
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_pipeline_Pipeline_isNativeRebuildActive(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.Pipeline.isNativeRebuildActive");
    return Pipeline::nativeRebuildActive() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_pipeline_Pipeline_isNativeModuleAvailable(JNIEnv *env,
                                                                                              jclass,
                                                                                              jstring name) {
    MCVR_PROFILE_SCOPE("jni.Pipeline.isNativeModuleAvailable");
    if (name == nullptr) return JNI_FALSE;
    const char *nativeString = env->GetStringUTFChars(name, nullptr);
    if (nativeString == nullptr) return JNI_FALSE;
    bool available = Pipeline::worldModuleConstructors.find(nativeString) != Pipeline::worldModuleConstructors.end();
    env->ReleaseStringUTFChars(name, nativeString);
    return available ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_radiance_client_pipeline_Pipeline_getAttributes(
    JNIEnv *env, jclass, jstring name, jobjectArray attributes, jstring languageObject) {
    MCVR_PROFILE_SCOPE("jni.Pipeline.getAttributes");
    if (name == nullptr) { return env->NewStringUTF("{}"); }

    const char *moduleNameChars = env->GetStringUTFChars(name, nullptr);
    if (moduleNameChars == nullptr) { return env->NewStringUTF("{}"); }

    std::vector<std::string> attributeList;
    if (attributes != nullptr) {
        jsize count = env->GetArrayLength(attributes);
        attributeList.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; i++) {
            auto *entry = static_cast<jstring>(env->GetObjectArrayElement(attributes, i));
            if (entry == nullptr) {
                attributeList.emplace_back();
                continue;
            }

            const char *entryChars = env->GetStringUTFChars(entry, nullptr);
            attributeList.emplace_back(entryChars != nullptr ? entryChars : "");
            if (entryChars != nullptr) { env->ReleaseStringUTFChars(entry, entryChars); }
            env->DeleteLocalRef(entry);
        }
    }

    std::string language = "en_us";
    if (languageObject != nullptr) {
        const char *languageChars = env->GetStringUTFChars(languageObject, nullptr);
        if (languageChars != nullptr) {
            language = languageChars;
            env->ReleaseStringUTFChars(languageObject, languageChars);
        }
    }

    std::string moduleName(moduleNameChars);
    env->ReleaseStringUTFChars(name, moduleNameChars);

    std::string result = "{}";
    if (moduleName == std::string(MainRenderModule::NAME)) {
        MainRenderModule mainRenderModule;
        result = mainRenderModule.getAttributes(attributeList, language);
    }

    return env->NewStringUTF(result.c_str());
}
