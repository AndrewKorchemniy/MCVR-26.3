#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class TextureUploadOverlapTracker {
  public:
    TextureUploadOverlapTracker(uint32_t width, uint32_t height) : width_(width), height_(height) {}

    bool append(uint32_t mip, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        assert(mip < mipMasks_.size() && width > 0 && height > 0);
        auto &mask = mipMasks_[mip];
        if (mask.words.empty()) {
            const uint32_t mipWidth = std::max(1u, width_ >> mip);
            const uint32_t mipHeight = std::max(1u, height_ >> mip);
            while (((uint64_t(std::max(mipWidth, mipHeight)) + ((uint64_t(1) << mask.shift) - 1)) >>
                    mask.shift) > 512) {
                ++mask.shift;
            }
            const uint64_t cellSize = uint64_t(1) << mask.shift;
            const uint32_t columns = static_cast<uint32_t>((mipWidth + cellSize - 1) >> mask.shift);
            const uint32_t rows = static_cast<uint32_t>((mipHeight + cellSize - 1) >> mask.shift);
            mask.rowWords = (columns + 63u) / 64u;
            mask.words.resize(size_t(rows) * mask.rowWords, 0);
        }
        const uint32_t x0 = x >> mask.shift;
        const uint32_t x1 = static_cast<uint32_t>((uint64_t(x) + width - 1) >> mask.shift);
        const uint32_t y0 = y >> mask.shift;
        const uint32_t y1 = static_cast<uint32_t>((uint64_t(y) + height - 1) >> mask.shift);
        const uint32_t firstWord = x0 / 64u;
        const uint32_t lastWord = x1 / 64u;
        const uint64_t firstBits = ~uint64_t(0) << (x0 % 64u);
        const uint64_t lastBits = ~uint64_t(0) >> (63u - x1 % 64u);

        bool overlaps = false;
        for (uint32_t row = y0; row <= y1 && !overlaps; ++row) {
            for (uint32_t word = firstWord; word <= lastWord; ++word) {
                const uint64_t bits = (word == firstWord ? firstBits : ~uint64_t(0)) &
                                      (word == lastWord ? lastBits : ~uint64_t(0));
                if ((mask.words[size_t(row) * mask.rowWords + word] & bits) != 0) {
                    overlaps = true;
                    break;
                }
            }
        }
        if (overlaps) {
            for (const auto &[level, word] : touchedWords_) { mipMasks_[level].words[word] = 0; }
            touchedWords_.clear();
        }
        for (uint32_t row = y0; row <= y1; ++row) {
            for (uint32_t word = firstWord; word <= lastWord; ++word) {
                const uint64_t bits = (word == firstWord ? firstBits : ~uint64_t(0)) &
                                      (word == lastWord ? lastBits : ~uint64_t(0));
                const size_t index = size_t(row) * mask.rowWords + word;
                if (mask.words[index] == 0) { touchedWords_.emplace_back(mip, index); }
                mask.words[index] |= bits;
            }
        }
        return overlaps;
    }

  private:
    struct MipMask {
        uint32_t shift = 0;
        uint32_t rowWords = 0;
        std::vector<uint64_t> words;
    };
    uint32_t width_;
    uint32_t height_;
    std::array<MipMask, 32> mipMasks_;
    std::vector<std::pair<uint32_t, size_t>> touchedWords_;
};
