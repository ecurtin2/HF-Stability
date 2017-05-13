#include "Config.hpp"

Config::Config(std::string OutFileName) {
    std::chrono::time_point<std::chrono::system_clock> thetime;
    thetime = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(thetime);
    date_time_start = std::string(std::ctime(&now));

    OutputFileName = OutFileName;
}

