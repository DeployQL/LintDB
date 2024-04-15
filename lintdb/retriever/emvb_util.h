#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lintdb {
    inline void set_bit_32(const size_t q_term_id, const size_t centroid_id, std::vector<uint32_t> & bitvectors)
    {
        bitvectors[centroid_id] |= (uint64_t)1 << q_term_id;
    }

    void assign_bitvector_32(const std::vector<size_t> &sorted_indexes, const size_t offset, const size_t topt, const size_t query_term_index, std::vector<uint32_t> & bitvectors)
    {
        for (size_t i = offset; i < topt; i++)
        {
            set_bit_32(query_term_index, sorted_indexes[i], bitvectors);
        }
    }

     inline void set_bit_64(const size_t doc_id, std::vector<uint64_t>& bitvectors_centroids)
    {
        size_t slot = doc_id / 64;
        size_t offset = doc_id % 64;

        bitvectors_centroids[slot] |= (uint64_t)1 << offset;
    }

    inline uint64_t check_bit_64(const size_t doc_id, const std::vector<uint64_t>& bitvectors_centroids)
    {
        size_t slot = doc_id / 64;
        size_t offset = doc_id % 64;

        return (bitvectors_centroids[slot] >> offset) & (uint64_t)1;
    }

    int popcount(uint32_t t) {
        return _mm_popcnt_u32(t);
    }
}