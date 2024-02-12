#ifndef LINTDB_EXCEPTION_H
#define LINTDB_EXCEPTION_H

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace lintdb {
    class LintDBException: public std::exception {
        public:
            explicit LintDBException(const std::string& message): message(message) {};

            LintDBException(const std::string& m, const char* funcName, const char* file, int line){
                int size = snprintf(
                        nullptr,
                        0,
                        "Error in %s at %s:%d: %s",
                        funcName,
                        file,
                        line,
                        m.c_str());
                message.resize(size + 1);
                snprintf(
                        &message[0],
                        message.size(),
                        "Error in %s at %s:%d: %s",
                        funcName,
                        file,
                        line,
                        m.c_str());
            }

            const char* what() const noexcept override {
                return message.c_str();
            };

        private:
            std::string message;
    };
}

#endif