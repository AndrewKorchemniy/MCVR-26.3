#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <condition_variable>
#include <future>
#include <thread>

class Framework;
class Chunks;
class Entities;
class Vista;

class World : public SharedObject<World> {
  public:
    enum VertexFormats {
        POSITION_COLOR_TEXTURE_LIGHT_NORMAL,
        POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL,
        POSITION_TEXTURE_COLOR_LIGHT,
        POSITION,
        POSITION_COLOR,
        LINES,
        POSITION_COLOR_LIGHT,
        POSITION_TEXTURE,
        POSITION_TEXTURE_COLOR,
        POSITION_COLOR_TEXTURE_LIGHT,
        POSITION_TEXTURE_LIGHT_COLOR,
        POSITION_TEXTURE_COLOR_NORMAL,
        ENTITY_PBR_TRIANGLE,
        CHUNK_PBR_TRIANGLE,
        NUM_VERTEX_FORMATS,
    };

    enum GeometryTypes {
        SHADOW,
        WORLD_SOLID,
        WORLD_TRANSPARENT,
        WORLD_NO_REFLECT,
        WORLD_CLOUD,
        BOAT_WATER_MASK,
        END_PORTAL,
        END_GATE_WAY,
        NUM_GEOMETRY_TYPES,
    };

    enum Coordinates {
        WORLD,
        CAMERA,
        CAMERA_SHIFT,
    };

    // Values match Java's Constants.DrawModes. 26.3 dropped LINE_STRIP, so value 1 is never sent but stays reserved.
    enum class DrawMode {
        LINES,
        UNUSED_LINE_STRIP,
        DEBUG_LINES,
        DEBUG_LINE_STRIP,
        TRIANGLES,
        TRIANGLE_STRIP,
        TRIANGLE_FAN,
        QUADS,
        POINTS,
    };

    World(std::shared_ptr<Framework> framework);

    void resetFrame();
    bool &shouldRender();

    std::shared_ptr<Chunks> chunks();
    std::shared_ptr<Entities> entities();
    std::shared_ptr<Vista> vista();

    void setCameraPos(glm::dvec3 cameraPos);
    glm::dvec3 getCameraPos();

    void close();

  private:
    std::shared_ptr<Chunks> chunks_;
    std::shared_ptr<Entities> entities_;
    std::shared_ptr<Vista> vista_;

    glm::dvec3 cameraPos_ = {0, 0, 0};

    bool shouldRenderWorld_;
};
