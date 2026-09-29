#include "blue_noise.hpp"


#include "../../../extern/FidelityFX-SDK/sdk/src/components/sssr/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp"

BlueNoise::BlueNoise(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma) {
    m_sobolBuffer =
        vk::DeviceLocalBuffer::create(vma, device, SOBOL_SIZE * sizeof(uint32_t),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint32_t> sobolData(SOBOL_SIZE);
    for (size_t i = 0; i < SOBOL_SIZE; i++) { sobolData[i] = static_cast<uint32_t>(sobol_256spp_256d[i]); }
    m_sobolBuffer->uploadToStagingBuffer(sobolData.data());

    m_scramblingBuffer =
        vk::DeviceLocalBuffer::create(vma, device, SCRAMBLING_SIZE * sizeof(uint32_t),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    std::vector<uint32_t> scramblingData(SCRAMBLING_SIZE);
    for (size_t i = 0; i < SCRAMBLING_SIZE; i++) { scramblingData[i] = static_cast<uint32_t>(scramblingTile[i]); }
    m_scramblingBuffer->uploadToStagingBuffer(scramblingData.data());
}

std::shared_ptr<vk::DeviceLocalBuffer> BlueNoise::sobolBuffer() {
    return m_sobolBuffer;
}

std::shared_ptr<vk::DeviceLocalBuffer> BlueNoise::scramblingBuffer() {
    return m_scramblingBuffer;
}

void BlueNoise::uploadToBuffer(std::shared_ptr<vk::CommandBuffer> cmdBuffer) {
    m_sobolBuffer->uploadToBuffer(cmdBuffer);
    m_scramblingBuffer->uploadToBuffer(cmdBuffer);
}
