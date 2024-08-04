#pragma once

#include <unordered_map>
#include <variant>
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/query/KnnNearestCentroids.h"
#include "lintdb/schema/FieldMapper.h"

namespace lintdb {

class QueryContext {
   public:
    const std::string colbert_context;

    explicit QueryContext(
            const uint64_t tenant,
            const std::string colbert_field,
            const std::shared_ptr<InvertedList> invertedList,
            const std::shared_ptr<FieldMapper> fieldMapper,
            const std::unordered_map<
                    std::string,
                    std::shared_ptr<ICoarseQuantizer>>& coarse_quantizer_map,
            const std::unordered_map<std::string, std::shared_ptr<Quantizer>>&
                    quantizer_map)
            : colbert_context(colbert_field),
              tenant(tenant),
              db_(invertedList),
              fieldMapper_(fieldMapper),
              coarse_quantizer_map(coarse_quantizer_map),
              quantizer_map(quantizer_map) {}

    inline std::shared_ptr<FieldMapper> getFieldMapper() const {
        return fieldMapper_;
    }

    inline std::shared_ptr<InvertedList> getIndex() const {
        return db_;
    }

    inline uint64_t getTenant() const {
        return tenant;
    }

    inline std::shared_ptr<ICoarseQuantizer> getCoarseQuantizer(
            const std::string& field) const {
        return coarse_quantizer_map.at(field);
    }

    inline std::shared_ptr<Quantizer> getQuantizer(
            const std::string& field) const {
        return quantizer_map.at(field);
    }

    inline std::shared_ptr<KnnNearestCentroids> getOrCreateNearestCentroids(
            const std::string& field) {
        if (knnNearestCentroidsMap.find(field) ==
            knnNearestCentroidsMap.end()) {
            auto knnNearestCentroids = std::make_shared<KnnNearestCentroids>();
            knnNearestCentroidsMap.insert(
                    {field, std::move(knnNearestCentroids)});
        }
        return knnNearestCentroidsMap.at(field);
    }

    inline void setNearestCentroids(
            const std::string& field,
            std::shared_ptr<KnnNearestCentroids> knnNearestCentroids) {
        knnNearestCentroidsMap.insert({field, knnNearestCentroids});
    }

   private:
    const uint64_t tenant;
    const std::shared_ptr<InvertedList> db_;
    const std::shared_ptr<FieldMapper> fieldMapper_;
    const std::unordered_map<std::string, std::shared_ptr<ICoarseQuantizer>>&
            coarse_quantizer_map;
    const std::unordered_map<std::string, std::shared_ptr<Quantizer>>&
            quantizer_map;
    std::unordered_map<std::string, std::shared_ptr<KnnNearestCentroids>>
            knnNearestCentroidsMap;
};

} // namespace lintdb
