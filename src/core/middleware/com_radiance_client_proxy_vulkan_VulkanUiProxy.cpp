#include <jni.h>

#include "core/middleware/jni_export.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

extern "C" {
JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VulkanUiProxy_fuseWorld(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.VulkanUiProxy.fuseWorld");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->fuseWorld();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VulkanUiProxy_postBlur(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.VulkanUiProxy.postBlur");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto world = Renderer::instance().world();
    if (world != nullptr && world->shouldRender()) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->postBlur(6);
}
}
