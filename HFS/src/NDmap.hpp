/** @file NDmap.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header-only implementation of N-dimensional template array.
@details
@date Wednesday, 22 Feb, 2017
*/

#ifndef NDMAP_INCLUDED
#define NDMAP_INCLUDED

#include <iostream>
#include <vector>
#include <stdarg.h>

template <class a_type>
class ndMap {
/** \brief Template-typed N-dimensional array container.
 *  Meant for use only to store and retrieve data. Used to store
 *  N indices mapping to 1 index. Data stored in row-major ordering.
 *  Elements are accessed with the () operator. Needs to be compiled
 *  with -std=c++11 or newer.
 *
 *  ### Output
 *  \include NDmap_example.log
 *  ### Source
 *  \example NDmap_example.cpp
 */

    private:
        std::vector<a_type> data;
        std::vector<a_type> steps;
    public:
        a_type n_elem;
        unsigned ndim;
        std::vector<a_type> shape;

    ndMap(std::vector<a_type>& lengths) {
        shape = lengths;
        ndim = lengths.size();
        n_elem = 1;
        for (auto &i : lengths) {
            n_elem *= i;
        }
        data.resize(n_elem);

        // get stepsizes for indexing
        steps.resize(ndim);
        steps.back() = 1;
        for (auto i = steps.size() - 1; i-- > 0; ) {  // looping in reverse
            steps[i] = shape[i + 1] * steps[i + 1];
        }

        // default to range(N)
        for (a_type i = 0; i < data.size(); ++i) {
            data[i] = i;
        }
    }

    a_type& operator()(a_type i1, ...){
        a_type idx = i1 * steps[0];
        va_list indices;
        va_start(indices, i1);
        for (unsigned i = 1; i < ndim; ++i) {
            idx += va_arg(indices, a_type) * steps[i];
        }
        va_end(indices);
        return data[idx];
    }

    void recursivePrint(unsigned depth, std::vector<a_type>& indices) {
        if (depth > 0) {
            for (a_type i = 0; i < shape[depth-1]; ++i) {
                indices[ndim - depth] = i;
                recursivePrint(depth - 1, indices);
            }
        }else{
            std::cout << "[";
            unsigned i = 0;
            while (i < indices.size() - 1) {
               std::cout << indices[i] << ", ";
               ++i;
            }
            a_type idx = indices[0] * steps[0];
            for (unsigned i = 1; i < ndim; ++i) {
                idx += indices[i] * steps[i];
            }
            std::cout << indices[i] << "] = "<< data[idx] << std::endl;
        }
    }

    void print() {
        std::vector<a_type> indices(ndim);
        recursivePrint(ndim, indices);
    }
};
#endif // NDMAP_INCLUDED
