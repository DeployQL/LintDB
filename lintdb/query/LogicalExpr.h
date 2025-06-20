#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/DataTypes.h"

namespace lintdb {
namespace query {

/**
 * @brief LogicalExpr represents a logical expression that can be evaluated
 * against data.
 */
class LogicalExpr {
public:
    enum class Type {
        COLUMN,
        LITERAL,
        BINARY,
        VECTOR_SEARCH,
        FILTER
    };

    virtual ~LogicalExpr() = default;

    /**
     * @brief Get the type of this logical expression
     * @return The type of the expression
     */
    virtual Type type() const = 0;

    /**
     * @brief Convert this expression to a field in the schema
     * @param input_schema The input schema
     * @return The field that represents this expression
     */
    Field to_field(const lintdb::Schema& input_schema) const {
        return input_schema.get_field(to_string());
    }

    /**
     * @brief Get a string representation of this expression
     * @return The string representation
     */
    virtual std::string to_string() const = 0;
};

/**
 * @brief Column represents a reference to a column in the data
 */
class Column : public LogicalExpr {
public:
    explicit Column(const std::string& name) : name_(name) {}

    Type type() const override { return Type::COLUMN; }

    std::string to_string() const override {
        return name_;
    }

private:
    std::string name_;
};

/**
 * @brief Literal represents a constant value
 */
class Literal : public LogicalExpr {
public:
    explicit Literal(const std::string& value) : value_(value) {}

    Type type() const override { return Type::LITERAL; }

    std::string to_string() const override {
        return value_;
    }

private:
    std::string value_;
};

/**
 * @brief BinaryExpr represents a binary operation between two expressions
 */
class BinaryExpr : public LogicalExpr {
public:
    enum class Op {
        EQ,
        NEQ,
        GT,
        GTE,
        LT,
        LTE,
        AND,
        OR
    };

    BinaryExpr(std::shared_ptr<LogicalExpr> left,
               Op op,
               std::shared_ptr<LogicalExpr> right)
        : left_(std::move(left)), op_(op), right_(std::move(right)) {}

    Type type() const override { return Type::BINARY; }

    std::string to_string() const override {
        std::string op_str;
        switch (op_) {
            case Op::EQ: op_str = "="; break;
            case Op::NEQ: op_str = "!="; break;
            case Op::GT: op_str = ">"; break;
            case Op::GTE: op_str = ">="; break;
            case Op::LT: op_str = "<"; break;
            case Op::LTE: op_str = "<="; break;
            case Op::AND: op_str = "AND"; break;
            case Op::OR: op_str = "OR"; break;
        }
        return left_->to_string() + " " + op_str + " " + right_->to_string();
    }

    BinaryExpr::Op op() const { return op_; }

    std::shared_ptr<LogicalExpr> left() const { return left_; }
    std::shared_ptr<LogicalExpr> right() const { return right_; }

private:
    std::shared_ptr<LogicalExpr> left_;
    Op op_;
    std::shared_ptr<LogicalExpr> right_;
};

/**
 * @brief Represents a vector search expression
 */
class VectorSearchExpr : public LogicalExpr {
public:
    VectorSearchExpr(const std::string& column_name, const std::vector<float>& query_vector, size_t k)
        : column_name_(column_name), query_vector_(query_vector), k_(k) {}

    Type type() const override { return Type::VECTOR_SEARCH; }

    std::string to_string() const override {
        return "VectorSearch(" + column_name_ + ", k=" + std::to_string(k_) + ")";
    }

    const std::string& column_name() const { return column_name_; }
    const std::vector<float>& query_vector() const { return query_vector_; }
    size_t k() const { return k_; }

private:
    std::string column_name_;
    std::vector<float> query_vector_;
    size_t k_;
};

/**
 * @brief Represents a filter expression
 */
class FilterExpr : public LogicalExpr {
public:
    FilterExpr(const std::string& column_name, const std::string& op, float value)
        : column_name_(column_name), op_(op), value_(value) {}

    Type type() const override { return Type::FILTER; }

    std::string to_string() const override {
        return "Filter(" + column_name_ + " " + op_ + " " + std::to_string(value_) + ")";
    }

    const std::string& column_name() const { return column_name_; }
    const std::string& op() const { return op_; }
    float value() const { return value_; }

private:
    std::string column_name_;
    std::string op_; // ">", "<", ">=", "<=", "=="
    float value_;
};

} // namespace query
} // namespace lintdb 