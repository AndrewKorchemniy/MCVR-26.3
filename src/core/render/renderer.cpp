#include "core/render/renderer.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

template <typename T, typename... Args>
std::shared_ptr<T> Renderer::createWithMemoryTag(const std::string &tag, Args &&...args) {
    vk::ResourceMemoryTagScope memoryTag(tag);
    return T::create(std::forward<Args>(args)...);
}

std::filesystem::path Renderer::folderPath{};
Options Renderer::options{};

Renderer::Renderer(GLFWwindow *window)
    : framework_(createWithMemoryTag<Framework>("framework", window)),
      textures_(createWithMemoryTag<Textures>("textures", framework_)),
      buffers_(createWithMemoryTag<Buffers>("buffers", framework_)),
      world_(createWithMemoryTag<World>("world", framework_)) {}

Renderer::Renderer(std::shared_ptr<vk::PresentationTarget> presentationTarget,
                   vk::OpenGLDeviceIdentity openGLDeviceIdentity)
    : framework_(
          createWithMemoryTag<Framework>("framework", std::move(presentationTarget), std::move(openGLDeviceIdentity))),
      textures_(createWithMemoryTag<Textures>("textures", framework_)),
      buffers_(createWithMemoryTag<Buffers>("buffers", framework_)),
      world_(createWithMemoryTag<World>("world", framework_)) {}

Renderer::~Renderer() {
    try {
        close();
    } catch (const std::exception &error) {
        std::cerr << "Renderer shutdown failed: " << error.what() << std::endl;
    } catch (...) { std::cerr << "Renderer shutdown failed" << std::endl; }
}

std::shared_ptr<Framework> Renderer::framework() {
    std::scoped_lock lck(closeMtx_);
    return framework_;
}

std::shared_ptr<Textures> Renderer::textures() {
    std::scoped_lock lck(closeMtx_);
    return textures_;
}

std::shared_ptr<Buffers> Renderer::buffers() {
    std::scoped_lock lck(closeMtx_);
    return buffers_;
}

std::shared_ptr<World> Renderer::world() {
    std::scoped_lock lck(closeMtx_);
    return world_;
}

void Renderer::close() {
    std::shared_ptr<Framework> framework;
    std::shared_ptr<Textures> textures;
    std::shared_ptr<Buffers> buffers;
    std::shared_ptr<World> world;
    {
        std::unique_lock closeLock(closeMtx_);
        closeCv_.wait(closeLock, [this] { return !closing_; });
        if (closed_) return;
        closing_ = true;
        framework = framework_;
        textures = textures_;
        buffers = buffers_;
        world = world_;
    }

    try {
        if (framework != nullptr) {
            framework->waitDeviceIdle();
        }
        if (world != nullptr) world->close();
        if (textures != nullptr) textures->shutdown();
        if (framework != nullptr) framework->close();
    } catch (...) {
        {
            std::scoped_lock closeLock(closeMtx_);
            closing_ = false;
        }
        closeCv_.notify_all();
        throw;
    }

    {
        std::scoped_lock closeLock(closeMtx_);
        textures_.reset();
        buffers_.reset();
        world_.reset();
        framework_.reset();
        closing_ = false;
        closed_ = true;
    }
    closeCv_.notify_all();

    world.reset();
    buffers.reset();
    textures.reset();
    framework.reset();
}
