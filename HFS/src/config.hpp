#ifndef CONFIG_H
#define CONFIG_H


class Config
{
    public:
        Config(std::string OutFileName);

    private:
        scalar Total_Calculation_Time;   /**< Time from main() start to finish */
        std::string date_time_start;     /**< Time of starting main() (date, time, year) */
        std::string OutputFileName;      /**< Name of the file to be written to */
};

#endif // CONFIG_H
