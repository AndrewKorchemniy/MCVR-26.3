#pragma once

#include <chrono>
#include <optional>

namespace mcvr::fsr {
class FrameClock {
  public:
    using Clock = std::chrono::steady_clock;

    float advance(Clock::time_point now = Clock::now()) {
        float milliseconds = previous_ ? std::chrono::duration<float, std::milli>(now - *previous_).count() :
                                        1000.0f / 60.0f;
        previous_ = now;
        return milliseconds > 0.0f ? milliseconds : 1000.0f / 60.0f;
    }

    void reset() { previous_.reset(); }

  private:
    std::optional<Clock::time_point> previous_;
};
}
