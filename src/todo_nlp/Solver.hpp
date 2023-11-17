/**
 * Solver interface for non-linear optimization packages
 * 
 * Author: Nick Goodson
 */ 

#pragma once

#include "libsnopt7_cpp/snoptProblem.hpp"


namespace gp_nlopt
{


class Solver
{
public:

    Solver();

protected:

    virtual int solve();

};


}
