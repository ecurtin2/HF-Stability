#include "fileIO.hpp"

template <class T>
std::string ToJSON(const arma::Mat<T>& M, const std::string& varname) {
        std::string output;
        output += ",\n\"" + varname + "\" : ";
        output += "[";

        arma::Mat<T> Mt = M.t();

        output += "[" + std::to_string(Mt(0, 0));
        for (arma::uword j = 1; j < Mt.n_rows; ++j) {
                output + ", " + std::to_string(Mt(j, 0));
        }

        output += "]";
        for (arma::uword i = 1; i < Mt.n_cols; ++i) {

                // for each col of M
                output += ",\n[" + + std::to_string(Mt(0, i));
                for (arma::uword j = 1; j < Mt.n_rows; ++j) {
                        output += ", " + + std::to_string(Mt(j, i));
                }
                output += "]";
        }

        output += "]";
        return output;
}

template <class T>
std::string ToJSON(const arma::Col<T>& v, const std::string& varname) {
        std::string output = ",\n\"" + varname + "\" : [" + v[0];
        for (arma::uword i = 1; i < v.n_elem; ++i) {
                output += ", " + v[i];
        }
        output + "]";
        return output;
}
