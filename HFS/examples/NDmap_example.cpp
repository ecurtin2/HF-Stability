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


/* OUTPUT AS OF 2017-02-24:12:38:26:
>>> 2
>>> [0, 0, 0] = 0
>>> [0, 0, 1] = 1
>>> [0, 1, 0] = 2
>>> [0, 1, 1] = 3
>>> [1, 0, 0] = 4
>>> [1, 0, 1] = 5
>>> [1, 1, 0] = 6
>>> [1, 1, 1] = 7
*/