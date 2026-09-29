#include "core/vulkan/render_pass.hpp"

#include "core/vulkan/device.hpp"

#include <iostream>

std::ostream &renderPassCerr() {
    return std::cerr << "[RenderPass] ";
}

vk::RenderPass::RenderPass(std::shared_ptr<Device> device, VkRenderPass renderpass)
    : device_(device), renderPass_(renderpass) {}

VkRenderPass &vk::RenderPass::vkRenderPass() {
    return renderPass_;
}

vk::RenderPass::~RenderPass() {
    vkDestroyRenderPass(device_->vkDevice(), renderPass_, nullptr);
}

vk::RenderPassBuilder::AttachmentDescriptionBuilder::AttachmentDescriptionBuilder(vk::RenderPassBuilder &parent)
    : parent(parent) {}

vk::RenderPassBuilder::AttachmentDescriptionBuilder &
vk::RenderPassBuilder::AttachmentDescriptionBuilder::defineAttachmentDescription(
    VkAttachmentDescription attachmentDescription) {
    attachmentDescriptions.push_back(attachmentDescription);
    return *this;
}

vk::RenderPassBuilder &vk::RenderPassBuilder::AttachmentDescriptionBuilder::endAttachmentDescription() {
    return parent;
}

vk::RenderPassBuilder::AttachmentReferenceBuilder::AttachmentReferenceBuilder(RenderPassBuilder &parent)
    : parent(parent) {}

vk::RenderPassBuilder::AttachmentReferenceBuilder &
vk::RenderPassBuilder::AttachmentReferenceBuilder::defineAttachmentReference(
    VkAttachmentReference attachmentReference) {
    attachmentReferences.push_back(attachmentReference);
    return *this;
}

vk::RenderPassBuilder &vk::RenderPassBuilder::AttachmentReferenceBuilder::endAttachmentReference() {
    return parent;
}

vk::RenderPassBuilder::SubpassDescriptionBuilder::SubpassDescriptionBuilder(vk::RenderPassBuilder &parent)
    : parent(parent) {}

vk::RenderPassBuilder::SubpassDescriptionBuilder &
vk::RenderPassBuilder::SubpassDescriptionBuilder::defineSubpassDescription(SubpassDescription subpassDescription) {
    VkSubpassDescription &vkDescription = subpassDescriptions.emplace_back();
    vkDescription = {};
    vkDescription.flags = subpassDescription.flags;
    vkDescription.pipelineBindPoint = subpassDescription.pipelineBindPoint;

    std::vector<VkAttachmentReference> &inputAttachmentReference = inputAttachmentReferences.emplace_back();
    for (uint32_t i : subpassDescription.inputAttachmentIndices) {
        inputAttachmentReference.push_back(parent.attachmentReferenceBuilder_.attachmentReferences[i]);
    }
    vkDescription.inputAttachmentCount = inputAttachmentReference.size();
    vkDescription.pInputAttachments = inputAttachmentReference.data();

    std::vector<VkAttachmentReference> &colorAttachmentReference = colorAttachmentReferences.emplace_back();
    for (uint32_t i : subpassDescription.colorAttachmentIndices) {
        colorAttachmentReference.push_back(parent.attachmentReferenceBuilder_.attachmentReferences[i]);
    }
    vkDescription.colorAttachmentCount = colorAttachmentReference.size();
    vkDescription.pColorAttachments = colorAttachmentReference.data();

    if (subpassDescription.resolveAttachmentIndex != -1) {
        vkDescription.pResolveAttachments =
            &parent.attachmentReferenceBuilder_.attachmentReferences[subpassDescription.resolveAttachmentIndex];
    } else {
        vkDescription.pResolveAttachments = nullptr;
    }

    if (subpassDescription.depthStencilAttachmentIndex != -1) {
        vkDescription.pDepthStencilAttachment =
            &parent.attachmentReferenceBuilder_.attachmentReferences[subpassDescription.depthStencilAttachmentIndex];
    } else {
        vkDescription.pDepthStencilAttachment = nullptr;
    }

    std::vector<uint32_t> &preserveAttachmentIndices_ =
        preserveAttachmentIndices.emplace_back(subpassDescription.preserveAttachmentIndices);
    vkDescription.preserveAttachmentCount = preserveAttachmentIndices_.size();
    vkDescription.pPreserveAttachments = preserveAttachmentIndices_.data();

    return *this;
}

vk::RenderPassBuilder &vk::RenderPassBuilder::SubpassDescriptionBuilder::endSubpassDescription() {
    return parent;
}

vk::RenderPassBuilder::SubpassDependencyBuilder::SubpassDependencyBuilder(vk::RenderPassBuilder &parent)
    : parent(parent) {}

vk::RenderPassBuilder::SubpassDependencyBuilder &
vk::RenderPassBuilder::SubpassDependencyBuilder::defineSubpassDependency(VkSubpassDependency subpassDependency) {
    subpassDependencies.push_back(subpassDependency);
    return *this;
}

vk::RenderPassBuilder &vk::RenderPassBuilder::SubpassDependencyBuilder::endSubpassDependency() {
    return parent;
}

vk::RenderPassBuilder::RenderPassBuilder()
    : attachmentDescriptionBuilder_(*this),
      attachmentReferenceBuilder_(*this),
      subpassDescriptionBuilder_(*this),
      subpassDependencyBuilder_(*this) {}

vk::RenderPassBuilder::AttachmentDescriptionBuilder &
vk::RenderPassBuilder::RenderPassBuilder::beginAttachmentDescription() {
    return attachmentDescriptionBuilder_;
}

vk::RenderPassBuilder::AttachmentReferenceBuilder &
vk::RenderPassBuilder::RenderPassBuilder::beginAttachmentReference() {
    return attachmentReferenceBuilder_;
}

vk::RenderPassBuilder::SubpassDescriptionBuilder &vk::RenderPassBuilder::RenderPassBuilder::beginSubpassDescription() {
    return subpassDescriptionBuilder_;
}

vk::RenderPassBuilder::SubpassDependencyBuilder &vk::RenderPassBuilder::RenderPassBuilder::beginSubpassDependency() {
    return subpassDependencyBuilder_;
}

std::shared_ptr<vk::RenderPass> vk::RenderPassBuilder::RenderPassBuilder::build(std::shared_ptr<Device> device) {
    VkRenderPassCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = attachmentDescriptionBuilder_.attachmentDescriptions.size();
    createInfo.pAttachments = attachmentDescriptionBuilder_.attachmentDescriptions.data();
    createInfo.subpassCount = subpassDescriptionBuilder_.subpassDescriptions.size();
    createInfo.pSubpasses = subpassDescriptionBuilder_.subpassDescriptions.data();
    createInfo.dependencyCount = subpassDependencyBuilder_.subpassDependencies.size();
    createInfo.pDependencies = subpassDependencyBuilder_.subpassDependencies.data();

    VkRenderPass renderPass;
    if (vkCreateRenderPass(device->vkDevice(), &createInfo, nullptr, &renderPass) != VK_SUCCESS) {
        renderPassCerr() << "failed to create render pass" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_shared<vk::RenderPass>(device, renderPass);
}