#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Device;

class Semaphore : public SharedObject<Semaphore> {
  public:
    Semaphore(std::shared_ptr<Device> device);
    Semaphore(std::shared_ptr<Device> device, bool exportable);
    ~Semaphore();

    VkSemaphore &vkSemaphore();
    uint64_t exportHandle();

  private:
    std::shared_ptr<Device> device_;

    VkSemaphore semaphore_;
};

class Fence : public SharedObject<Fence> {
  public:
    Fence(std::shared_ptr<Device> device);
    Fence(std::shared_ptr<Device> device, bool signaled);
    ~Fence();

    VkFence &vkFence();

  private:
    std::shared_ptr<Device> device_;

    VkFence fence_;
};
};
