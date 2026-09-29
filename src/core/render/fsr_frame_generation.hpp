#pragma once

#include "core/render/frame_generation.hpp"

#ifdef MCVR_ENABLE_FFX_UPSCALER
#    include <ffx_api/ffx_api.hpp>
#endif

namespace mcvr {

class FsrFrameGenerationProvider : public IFrameGenerationProvider, public SharedObject<FsrFrameGenerationProvider> {
  public:
    FsrFrameGenerationProvider() = default;
    ~FsrFrameGenerationProvider() override;

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
    bool ensureContext(const FrameGenerationInputs &inputs);
    void releaseContext();

    std::weak_ptr<Framework> framework_;
    FrameGenerationCapabilities capabilities_{};
    VkExtent2D displayExtent_{};
    VkExtent2D renderExtent_{};
    VkFormat backbufferFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat hudlessBackbufferFormat_ = VK_FORMAT_UNDEFINED;
    uint64_t frameId_ = 0;
    bool available_ = false;

#ifdef MCVR_ENABLE_FFX_UPSCALER
    ffx::Context frameGenerationContext_ = nullptr;
#endif
};

}
