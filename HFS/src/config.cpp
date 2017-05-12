#include "Config.hpp"

Config::Config(std::string OutFileName) {
    auto thetime = std::chrono::system_clock::now();

    std::time_t now = std::chrono::system_clock::to_time_t(thetime);
    date_time_start = std::string(std::ctime(&now));
    OutputFileName = OutFileName;
}

Config::to_JSON() {

    auto calc_time_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();


}


