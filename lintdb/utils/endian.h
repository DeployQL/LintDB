#pragma once

#include <cstddef>
#include <vector>

namespace lintdb {
template <typename T>
T load_bigendian(void const* bytes) {
    T num = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        num |= static_cast<T>(static_cast<const unsigned char*>(bytes)[i])
                << (8 * (sizeof(T) - i - 1));
    }
    return num;
}

template <typename T>
void store_bigendian(T num, std::vector<unsigned char>& bigEndian) {
    for (int i = sizeof(T) - 1; i >= 0; i--) {
        bigEndian.push_back((num >> (8 * i)) & 0xff);
    }
}
} // namespace lintdb