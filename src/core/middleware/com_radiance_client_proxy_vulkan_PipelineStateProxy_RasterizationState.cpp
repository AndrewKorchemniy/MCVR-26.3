#include "com_radiance_client_proxy_vulkan_PipelineStateProxy_RasterizationState.h"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {
JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_setLineWidth(
    JNIEnv *, jclass, jfloat lineWidth) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.setLineWidth");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayLineWidth(lineWidth);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_vkSetPolygonMode(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jint polygonMode) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.vkSetPolygonMode");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayPolygonMode(polygonMode);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_vkSetCullMode(
    JNIEnv *, jclass, jint cullMode) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.vkSetCullMode");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayCullMode(cullMode);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_vkSetFrontFace(
    JNIEnv *, jclass, jint frontFace) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.vkSetFrontFace");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayFrontFace(frontFace);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_vkSetDepthBiasEnable(JNIEnv *,
                                                                                                      jclass,
                                                                                                      jint polygonMode,
                                                                                                      jboolean enable) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.vkSetDepthBiasEnable");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayDepthBiasEnable(polygonMode, enable);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024RasterizationState_vkSetDepthBias(
    JNIEnv *, jclass, jfloat depthBiasSlopeFactor, jfloat depthBiasConstantFactor) {
    MCVR_PROFILE_SCOPE("jni.PipelineState.Rasterization.vkSetDepthBias");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayDepthBias(depthBiasSlopeFactor, depthBiasConstantFactor);
}
}
