#pragma once

#include <drogon/HttpController.h>
#include <json/reader.h>
#include <json/writer.h>
#include <stdint.h>
#include "lintdb/index.h"
#include "lintdb/SearchResult.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/query/Query.h"
#include "lintdb/query/QueryNode.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Schema.h"
#include "query_node_translator.h"
#include "result_translator.h"
#include <memory>
#include <string>

using namespace drogon;

namespace v1 {

HttpResponsePtr makeFailedResponse()
{
    Json::Value json;
    json["ok"] = false;
    auto resp = HttpResponse::newHttpJsonResponse(json);
    resp->setStatusCode(k500InternalServerError);
    return resp;
}


class Index: public drogon::HttpController<Index, false> {
   public:
    METHOD_LIST_BEGIN
    METHOD_ADD(Index::search, "/search/{1}", Post);
    METHOD_ADD(Index::add, "/add/{1}", Post);
    METHOD_ADD(Index::update, "/update/{1}", Post);
    METHOD_ADD(Index::remove, "/remove/{1}", Post);
    METHOD_LIST_END

//    void create(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback) {
//        auto jsonPayload = req->jsonObject();
//        if (jsonPayload == nullptr) {
//            callback(makeFailedResponse());
//            return;
//        }
//
//        std::string path = (*jsonPayload)["name"].asString();
//        Json::Value s = (*jsonPayload)["schema"];
//
//        lintdb::Schema schema = lintdb::Schema::fromJson(s);
//        lintdb::Configuration config = lintdb::Configuration();
//
//        try {
//            index_ = std::make_shared<lintdb::IndexIVF>(path, schema, config);
//            Json::Value json;
//            json["ok"] = true;
//            auto resp = HttpResponse::newHttpJsonResponse(json);
//            callback(resp);
//        } catch (const std::exception &e) {
//            callback(makeFailedResponse());
//        }
//    }

//    void train(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback) {
//        auto jsonPayload = req->jsonObject();
//        if (jsonPayload == nullptr) {
//            callback(makeFailedResponse());
//            return;
//        }
//
//        std::vector<lintdb::Document> docs = (*jsonPayload)["documents"]
//
//        try {
//            index_->train(docs);
//            auto resp = HttpResponse::newHttpJsonResponse({{"status", "Index trained successfully"}});
//            callback(resp);
//        } catch (const std::exception &e) {
//            auto resp = HttpResponse::newHttpJsonResponse({{"error", e.what()}});
//            callback(resp);
//        }
//    }

    void search(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback, uint64_t tenant) {
        LOG_DEBUG << "Index search";

        auto jsonPayload = req->jsonObject();
        if (jsonPayload == nullptr) {
            callback(makeFailedResponse());
            return;
        }

        Json::Value options = (*jsonPayload)["options"];
        std::string colbert_field = options["colbert_field"].asString();
        Json::Value q = (*jsonPayload)["query"];
        std::unique_ptr<lintdb::QueryNode> qn = server::QueryNodeJsonTranslator::fromJson(q);
        auto query = lintdb::Query(std::move(qn));

//        lintdb::FieldValue fv("colbert", embeddings, 32);
//        std::unique_ptr<lintdb::VectorQueryNode> root = std::make_unique<lintdb::VectorQueryNode>(fv);
//        auto query = lintdb::Query(std::move(root));


        uint64_t k = (*jsonPayload)["k"].asUInt64();

        lintdb::SearchOptions opts;
        opts.colbert_field = colbert_field;

        try {
            auto results = index_->search(tenant, query, k, opts);
            Json::Value result_list;
            for(auto& r: results) {
                result_list.append(server::SearchResultJsonTranslator::toJson(r));
            }

            Json::Value result;
            result["results"] = result_list;
            auto resp = HttpResponse::newHttpJsonResponse(result);
            callback(resp);
        } catch (const std::exception &e) {
            auto resp = HttpResponse::newHttpJsonResponse({{"error", e.what()}});
            callback(resp);
        }
    }

    void add(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback, uint64_t tenant) {
        auto jsonPayload = req->jsonObject();
        if (jsonPayload == nullptr) {
            callback(makeFailedResponse());
            return;
        }

        Json::Value jd = (*jsonPayload)["documents"];

        std::vector<lintdb::Document> docs;
        for(auto& d : jd) {
            auto doc = lintdb::Document::fromJson(d);
            docs.push_back(doc);
        }

        try {
            index_->add(tenant, docs);
            Json::Value json;
            json["ok"] = true;
            auto resp = HttpResponse::newHttpJsonResponse(json);
            callback(resp);
        } catch (const std::exception &e) {
            auto resp = HttpResponse::newHttpJsonResponse({{"error", e.what()}});
            callback(resp);
        }
    }

    void update(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback, uint64_t tenant) {
        auto jsonPayload = req->jsonObject();
        if (jsonPayload == nullptr) {
            callback(makeFailedResponse());
            return;
        }

        Json::Value jd = (*jsonPayload)["documents"];

        std::vector<lintdb::Document> docs;
        for(auto& d : jd) {
            docs.push_back(lintdb::Document::fromJson(d));
        }

        try {
            index_->update(tenant, docs);
            Json::Value json;
            json["ok"] = true;
            auto resp = HttpResponse::newHttpJsonResponse(json);
            callback(resp);
        } catch (const std::exception &e) {
            auto resp = HttpResponse::newHttpJsonResponse({{"error", e.what()}});
            callback(resp);
        }
    }

    void remove(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback, uint64_t tenant) {
        auto jsonPayload = req->jsonObject();
        if (jsonPayload == nullptr) {
            callback(makeFailedResponse());
            return;
        }

        Json::Value jd = (*jsonPayload)["ids"];

        std::vector<idx_t> ids;
        for(auto& d : jd) {
            ids.push_back(d.asInt64());
        }

        try {
            index_->remove(tenant, ids);
            Json::Value json;
            json["ok"] = true;
            auto resp = HttpResponse::newHttpJsonResponse(json);
            callback(resp);
        } catch (const std::exception &e) {
            auto resp = HttpResponse::newHttpJsonResponse({{"error", e.what()}});
            callback(resp);
        }
    }

   public:
    Index(std::string path, bool read_only): path(path) {
        index_ = std::make_shared<lintdb::IndexIVF>(path, read_only);
    }

   private:
    std::string path;
    std::shared_ptr<lintdb::IndexIVF> index_;
};
}
