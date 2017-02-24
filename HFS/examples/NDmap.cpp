#include "NDmap.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<int> lengths = {2,2,2};
    ndMap<int> mymap(lengths);

    std::cout << mymap(0, 1, 0) << std::endl;
    mymap.print();
    return 0;
}
