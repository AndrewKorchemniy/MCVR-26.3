#include "core/render/world.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/vista.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

World::World(std::shared_ptr<Framework> framework)
    : chunks_(Chunks::create(framework)), entities_(Entities::create(framework)), vista_(Vista::create(framework)) {
    MCVR_PROFILE_SCOPE("native.World.init");
}

void World::resetFrame() {
    MCVR_PROFILE_SCOPE("native.World.resetFrame");
}

bool &World::shouldRender() {
    return shouldRenderWorld_;
}

std::shared_ptr<Chunks> World::chunks() {
    return chunks_;
}

std::shared_ptr<Entities> World::entities() {
    return entities_;
}

std::shared_ptr<Vista> World::vista() { return vista_; }

void World::setCameraPos(glm::dvec3 cameraPos) {
    MCVR_PROFILE_SCOPE("native.World.setCameraPos");
    cameraPos_ = cameraPos;
}

glm::dvec3 World::getCameraPos() {
    return cameraPos_;
}

void World::close() {
    MCVR_PROFILE_SCOPE("native.World.close");
    shouldRenderWorld_ = false;
    chunks_->close();
    entities_->close();
    vista_->close();
}
