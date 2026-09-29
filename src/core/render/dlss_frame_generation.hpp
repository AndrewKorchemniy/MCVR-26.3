#pragma once

#include "core/render/frame_generation.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"

namespace mcvr {

class DlssFrameGenerationProvider : public IFrameGenerationProvider, public SharedObject<DlssFrameGenerationProvider> {
  public:
    DlssFrameGenerationProvider() = default;
    ~DlssFrameGenerationProvider() override;

    bool init(std::shared_ptr<Framework> framework) override;
    void close() override;
    bool isAvailable() const override;
    FrameGenerationCapabilities capabilities() const override;
    bool evaluate(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                  const FrameGenerationInputs &inputs,
                  std::shared_ptr<vk::DeviceLocalImage> outputInterpolated,
                  uint32_t multiFrameCount,
                  uint32_t multiFrameIndex) override;

  private:
    bool ensureFeature(std::shared_ptr<vk::CommandBuffer> commandBuffer, const FrameGenerationInputs &inputs);
    void releaseFeature();

    std::weak_ptr<Framework> framework_;
    std::shared_ptr<NgxContext> ngxContext_;
    NVSDK_NGX_Handle *featureHandle_ = nullptr;
    FrameGenerationCapabilities capabilities_{};
    VkExtent2D displayExtent_{};
    VkExtent2D renderExtent_{};
    VkFormat backbufferFormat_ = VK_FORMAT_UNDEFINED;
    bool available_ = false;
};

}
