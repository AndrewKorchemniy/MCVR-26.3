#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <vector>

class Framework;
struct FrameworkContext;

// A non-owning vk::Image over the image Minecraft drew its UI into (its main color target). It is re-pointed at the
// current frame's image by setPresentSource(), since Minecraft recreates the image on resize. MCVR never frees it and
// never changes its layout: it is always VK_IMAGE_LAYOUT_GENERAL.
class UiSurfaceImage : public vk::Image {
  public:
    void setTarget(VkImage image, VkImageView imageView, uint32_t width, uint32_t height);
    void clearTarget();
    bool valid() const;

    uint32_t width() override;
    uint32_t height() override;
    uint32_t depth() override;
    uint32_t layer() override;
    VkFormat &vkFormat() override;
    VkImage &vkImage() override;
    VkImageView &vkImageView(int index = 0) override;
    VkImageLayout &imageLayout() override;
    VkImageSubresourceRange fullSubresourceRange() const override;

  private:
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_GENERAL;
};

// Builds the final frame: the ray-traced world with Minecraft's UI surface composited over it. This plays the role
// the OpenGL UI surface played on 1.21.4. The result is the image frame generation reads and the one copied to the
// swapchain.
class UiComposite : public SharedObject<UiComposite> {
  public:
    UiComposite();
    ~UiComposite();

    void init(std::shared_ptr<Framework> framework);

    std::shared_ptr<UiSurfaceImage> uiSurface();
    std::shared_ptr<vk::DeviceLocalImage> finalColor(uint32_t frameIndex);

    // Records the world copy, the UI copy and the composite into the frame's overlay command buffer.
    void record(std::shared_ptr<FrameworkContext> context, bool worldRendered);

  private:
    std::weak_ptr<Framework> framework_;
    std::shared_ptr<UiSurfaceImage> uiSurface_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> finalColorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> uiCopyImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::shared_ptr<vk::Sampler> sampler_;
    std::shared_ptr<vk::ComputePipeline> pipeline_;
};
