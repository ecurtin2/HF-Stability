#include "config.hpp"

Config::Config() {
    start_time = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::to_time_t(start_time);
    date_time_start = std::string(std::ctime(&now));
}

void Config::get_Total_Calculation_Time() {
    auto thetime = std::chrono::system_clock::now();
    using FpSeconds =
        std::chrono::duration<float, std::chrono::seconds::period>;

    static_assert(std::chrono::treat_as_floating_point<FpSeconds::rep>::value,
                  "Rep required to be floating point");
    auto diff = FpSeconds(start_time - thetime);
    Total_Calculation_Time = diff.count();
}

std::string Config::to_JSON() {
    std::string str = "aaaaa";
    return str;
}
