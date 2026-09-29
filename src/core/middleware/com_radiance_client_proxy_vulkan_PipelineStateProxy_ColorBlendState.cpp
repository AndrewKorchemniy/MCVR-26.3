#include "com_radiance_client_proxy_vulkan_PipelineStateProxy_ColorBlendState.h"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setBlendEnableNative(JNIEnv *,
                                                                                                   jclass,
                                                                                                   jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.setBlendEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setColorBlendConstants(
    JNIEnv *, jclass, jfloat const1, jfloat const2, jfloat const3, jfloat const4) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.setColorBlendConstants");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorBlendConstants(const1, const2, const3, const4);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setColorLogicOpEnableNative(
    JNIEnv *, jclass, jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.setColorLogicOpEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorLogicOpEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetBlendFuncSeparateNative(
    JNIEnv *,
    jclass,
    jint srcColorBlendFactor,
    jint srcAlphaBlendFactor,
    jint dstColorBlendFactor,
    jint dstAlphaBlendFactor) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.vkSetBlendFuncSeparate");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendFuncSeparate(srcColorBlendFactor, srcAlphaBlendFactor,
                                                                  dstColorBlendFactor, dstAlphaBlendFactor);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetBlendOpSeparate(JNIEnv *,
                                                                                                   jclass,
                                                                                                   jint colorBlendOp,
                                                                                                   jint alphaBlendOp) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.vkSetBlendOpSeparate");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendOpSeparate(colorBlendOp, alphaBlendOp);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetColorWriteMask(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jint colorWriteMask) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.vkSetColorWriteMask");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorWriteMask(colorWriteMask);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetColorLogicOp(
    JNIEnv *, jclass, jint colorLogicOp) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.ColorBlend.vkSetColorLogicOp");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorLogicOp(colorLogicOp);
}
}
