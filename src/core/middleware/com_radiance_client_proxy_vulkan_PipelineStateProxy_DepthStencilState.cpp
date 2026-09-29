#include "com_radiance_client_proxy_vulkan_PipelineStateProxy_DepthStencilState.h"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {
JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_setDepthTestEnableNative(
    JNIEnv *, jclass, jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.setDepthTestEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayDepthTestEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_setDepthWriteEnable(JNIEnv *,
                                                                                                    jclass,
                                                                                                    jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.setDepthWriteEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayDepthWriteEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_setStencilTestEnable(JNIEnv *,
                                                                                                     jclass,
                                                                                                     jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.setStencilTestEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilTestEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetDepthCompareOpNative(
    JNIEnv *, jclass, jint depthCompareOp) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetDepthCompareOp");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayDepthCompareOp(depthCompareOp);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilFrontFunc(
    JNIEnv *, jclass, jint compareOp, jint reference, jint compareMask) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilFrontFunc");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilFrontFunc(compareOp, reference, compareMask);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilBackFunc(
    JNIEnv *, jclass, jint compareOp, jint reference, jint compareMask) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilBackFunc");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilBackFunc(compareOp, reference, compareMask);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilFrontOp(
    JNIEnv *, jclass, jint failOp, jint depthFailOp, jint passOp) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilFrontOp");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilFrontOp(failOp, depthFailOp, passOp);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilBackOp(
    JNIEnv *, jclass, jint failOp, jint depthFailOp, jint passOp) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilBackOp");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilBackOp(failOp, depthFailOp, passOp);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilFrontWriteMask(
    JNIEnv *, jclass, jint writeMask) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilFrontWriteMask");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilFrontWriteMask(writeMask);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024DepthStencilState_vkSetStencilBackWriteMask(
    JNIEnv *, jclass, jint writeMask) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.DepthStencil.vkSetStencilBackWriteMask");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayStencilBackWriteMask(writeMask);
}
}
