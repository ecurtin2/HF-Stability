#include "HFSnamespace.h"


int main()
{
    HFS::rs = 1.2;
    HFS::Nk = 5;
    HFS::ndim = 2;
    HFS::calc_params();
    HFS::print_params();

}
