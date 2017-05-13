#ifndef CONFIG_H
#define CONFIG_H


#include <string>
#include <chrono>
#include "constants.hpp"

class Config
{
    public:
        Config();
        std::string to_JSON();

    private:
        scalar Total_Calculation_Time;   /**< Time from start to finish */
        std::string date_time_start;     /**< Time of starting (date, time, year) */
        std::chrono::system_clock::time_point start_time;           /**< Time of starting */
        std::string OutputFileName;      /**< Name of the file to be written to */
        void get_Total_Calculation_Time();

};

#endif // CONFIG_H
