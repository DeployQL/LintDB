#pragma once

#include <memory>
#include <string>
#include <vector>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/compute/api.h>
#include <arrow/builder.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include "lintdb/query/LogicalExpr.h"

namespace lintdb {
namespace query {

/**
 * @brief Base class for all physical expressions
 */
class PhysicalExpr {
public:
    virtual ~PhysicalExpr() = default;

    /**
     * @brief Evaluate the expression on a record batch
     * @param batch The input record batch
     * @return The resulting column vector
     */
    virtual std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) = 0;

    /**
     * @brief Get a string representation of this expression
     * @return The string representation
     */
    virtual std::string to_string() const = 0;
};

/**
 * @brief Physical expression for column references
 */
class ColumnPhysicalExpr : public PhysicalExpr {
public:
    explicit ColumnPhysicalExpr(int index) : index_(index) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (index_ < 0 || index_ >= batch->num_columns()) {
            throw std::runtime_error("Column index out of bounds: " + std::to_string(index_));
        }
        return batch->column(index_);
    }

    std::string to_string() const override {
        return "#" + std::to_string(index_);
    }

private:
    int index_;
};

/**
 * @brief Physical expression for literal values
 */
class LiteralPhysicalExpr : public PhysicalExpr {
public:
    explicit LiteralPhysicalExpr(const std::string& value) : value_(value) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        // Create a constant array with the literal value
        arrow::StringBuilder builder;
        auto status = builder.Append(value_);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        std::shared_ptr<arrow::Array> array;
        status = builder.Finish(&array);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        return array;
    }

    std::string to_string() const override {
        return "'" + value_ + "'";
    }

private:
    std::string value_;
};

/**
 * @brief Base class for binary physical expressions
 */
class BinaryPhysicalExpr : public PhysicalExpr {
public:
    BinaryPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                      std::shared_ptr<PhysicalExpr> right,
                      BinaryExpr::Op op)
        : left_(std::move(left)), right_(std::move(right)), op_(op) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // TODO: Implement binary operations based on op_
        // This will need to handle different types and operations
        // For now, return the left result as a placeholder
        return left_result;
    }

    std::string to_string() const override {
        std::string op_str;
        switch (op_) {
            case BinaryExpr::Op::EQ: op_str = "="; break;
            case BinaryExpr::Op::GT: op_str = ">"; break;
            case BinaryExpr::Op::GTE: op_str = ">="; break;
            case BinaryExpr::Op::LT: op_str = "<"; break;
            case BinaryExpr::Op::LTE: op_str = "<="; break;
            case BinaryExpr::Op::AND: op_str = "AND"; break;
            case BinaryExpr::Op::OR: op_str = "OR"; break;
            default: op_str = "UNKNOWN"; break;
        }
        return "(" + left_->to_string() + " " + op_str + " " + right_->to_string() + ")";
    }

    BinaryExpr::Op op() const { return op_; }

protected:
    std::shared_ptr<PhysicalExpr> left_;
    std::shared_ptr<PhysicalExpr> right_;
    BinaryExpr::Op op_;
};

/**
 * @brief Helper function to ensure arrays have the same type
 */
arrow::Result<std::shared_ptr<arrow::Array>> ensure_same_type(
    const std::shared_ptr<arrow::Array>& left,
    const std::shared_ptr<arrow::Array>& right) {
    if (left->type()->Equals(right->type())) {
        return arrow::Result<std::shared_ptr<arrow::Array>>(right);
    }
    // TODO: Implement type casting if needed
    return arrow::Status::TypeError("Type mismatch in binary operation");
}

/**
 * @brief Physical expression for equality comparison
 */
class EqPhysicalExpr : public BinaryPhysicalExpr {
public:
    EqPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                   std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::EQ) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure types match
        auto right_cast = ensure_same_type(left_result, right_result);
        if (!right_cast.ok()) {
            throw std::runtime_error(right_cast.status().ToString());
        }

        // Use Arrow compute for efficient comparison
        auto result = arrow::compute::CallFunction("equal", {left_result, right_cast.ValueOrDie()});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " = " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for greater than comparison
 */
class GtPhysicalExpr : public BinaryPhysicalExpr {
public:
    GtPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                   std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::GT) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure types match
        auto right_cast = ensure_same_type(left_result, right_result);
        if (!right_cast.ok()) {
            throw std::runtime_error(right_cast.status().ToString());
        }

        // Use Arrow compute for efficient comparison
        auto result = arrow::compute::CallFunction("greater", {left_result, right_cast.ValueOrDie()});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " > " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for less than comparison
 */
class LtPhysicalExpr : public BinaryPhysicalExpr {
public:
    LtPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                   std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::LT) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure types match
        auto right_cast = ensure_same_type(left_result, right_result);
        if (!right_cast.ok()) {
            throw std::runtime_error(right_cast.status().ToString());
        }

        // Use Arrow compute for efficient comparison
        auto result = arrow::compute::CallFunction("less", {left_result, right_cast.ValueOrDie()});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " < " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for greater than or equals comparison
 */
class GtePhysicalExpr : public BinaryPhysicalExpr {
public:
    GtePhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                    std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::GTE) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure types match
        auto right_cast = ensure_same_type(left_result, right_result);
        if (!right_cast.ok()) {
            throw std::runtime_error(right_cast.status().ToString());
        }

        // Use Arrow compute for efficient comparison
        auto result = arrow::compute::CallFunction("greater_equal", {left_result, right_cast.ValueOrDie()});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " >= " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for less than or equals comparison
 */
class LtePhysicalExpr : public BinaryPhysicalExpr {
public:
    LtePhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                    std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::LTE) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure types match
        auto right_cast = ensure_same_type(left_result, right_result);
        if (!right_cast.ok()) {
            throw std::runtime_error(right_cast.status().ToString());
        }

        // Use Arrow compute for efficient comparison
        auto result = arrow::compute::CallFunction("less_equal", {left_result, right_cast.ValueOrDie()});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " <= " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for logical AND
 */
class AndPhysicalExpr : public BinaryPhysicalExpr {
public:
    AndPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                    std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::AND) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure both arrays are boolean
        if (left_result->type()->id() != arrow::Type::BOOL || 
            right_result->type()->id() != arrow::Type::BOOL) {
            throw std::runtime_error("Logical AND requires boolean operands");
        }

        // Use Arrow compute for efficient logical AND
        auto result = arrow::compute::CallFunction("and", {left_result, right_result});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " AND " + right_->to_string() + ")";
    }
};

/**
 * @brief Physical expression for logical OR
 */
class OrPhysicalExpr : public BinaryPhysicalExpr {
public:
    OrPhysicalExpr(std::shared_ptr<PhysicalExpr> left,
                   std::shared_ptr<PhysicalExpr> right)
        : BinaryPhysicalExpr(std::move(left), std::move(right), BinaryExpr::Op::OR) {}

    std::shared_ptr<arrow::Array> evaluate(const std::shared_ptr<arrow::RecordBatch>& batch) override {
        auto left_result = left_->evaluate(batch);
        auto right_result = right_->evaluate(batch);

        // Ensure both arrays are boolean
        if (left_result->type()->id() != arrow::Type::BOOL || 
            right_result->type()->id() != arrow::Type::BOOL) {
            throw std::runtime_error("Logical OR requires boolean operands");
        }

        // Use Arrow compute for efficient logical OR
        auto result = arrow::compute::CallFunction("or", {left_result, right_result});
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return result.ValueOrDie().make_array();
    }

    std::string to_string() const override {
        return "(" + left_->to_string() + " OR " + right_->to_string() + ")";
    }
};

/**
 * @brief Factory function to create binary physical expressions
 */
std::shared_ptr<PhysicalExpr> create_binary_physical_expr(
    std::shared_ptr<PhysicalExpr> left,
    BinaryExpr::Op op,
    std::shared_ptr<PhysicalExpr> right) {
    switch (op) {
        case BinaryExpr::Op::EQ:
            return std::make_shared<EqPhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::GT:
            return std::make_shared<GtPhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::GTE:
            return std::make_shared<GtePhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::LT:
            return std::make_shared<LtPhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::LTE:
            return std::make_shared<LtePhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::AND:
            return std::make_shared<AndPhysicalExpr>(std::move(left), std::move(right));
        case BinaryExpr::Op::OR:
            return std::make_shared<OrPhysicalExpr>(std::move(left), std::move(right));
        default:
            throw std::runtime_error("Unsupported binary operation");
    }
}

} // namespace query
} // namespace lintdb 