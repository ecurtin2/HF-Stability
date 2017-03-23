#include<iostream>
#include<vector>
#include<string>
#include<stdexcept>
#include<assert.h>


void set_val_from_string(double& val, const std::string& str) {
    val = std::stod(str);
}

template <class int_type>
void set_val_from_string(int_type& val, const std::string& str) {
    val = std::stoi(str);
}

void set_val_from_string(float& val, const std::string& str) {
    val = std::stof(str);
}

void set_val_from_string(long double& val, const std::string& str) {
    val = std::stold(str);
}

void set_val_from_string(std::string& val, const std::string& str) {
    val = str;
}

class ConfigParser {
    public:
        std::vector<std::string> args;
        std::vector<std::string> arg_vals;
        std::vector<std::string> arg_names;

        ConfigParser(int inp_argc, char* inp_argv[]) {
            for (int i = 1; i < inp_argc; ++i) {
                std::string arg(inp_argv[i]);
                args.push_back(arg);
            }

            for (auto it : args) {
                if (it.find("--") != std::string::npos) {
                    arg_names.push_back(it);
                } else {
                    arg_vals.push_back(it);
                }
            }
        }

        template <class T>
        void set_val(T& val, std::string varname, bool checkerr=true) {
            /* Set value from args. Raise error if not found unless checkerr=false */
            bool found = false;
            for (unsigned i = 0; i < arg_names.size(); ++i) {
                if (varname == arg_names[i]){
                    set_val_from_string(val, arg_vals[i]);
                    found = true;
                }
            }
            if (!found && checkerr) {
                std::string err = "Variable: " + varname + " not found while parsing arguments";
                throw std::invalid_argument(err);
            }
        }
};
