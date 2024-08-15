#include <drogon/drogon.h>
#include "controllers/v1/Index.h"
#include <args.hxx>
#include <iostream>
#include <memory>

using namespace drogon;

int main(int argc, char**argv)
{
    args::ArgumentParser parser("LintDB Server.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::ValueFlag<std::string> path(parser, "path", "Set the path to the database", {'p', "path"});
    args::Flag read_only(parser, "read-only", "Set the database to read-only mode", {'r', "read-only"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    std::string p = args::get(path);
    auto indexController = std::make_shared<v1::Index>(p, !!read_only);

    app().setLogPath("./", "lintdb-server.log")
            .setLogLevel(trantor::Logger::kDebug)
            .addListener("0.0.0.0", 8080)
            .setThreadNum(12)
            .registerController(indexController)
//            .enableRunAsDaemon()
            .run();
}