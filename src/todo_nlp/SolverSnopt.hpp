/**
 * Solver interface for the SNOPT non-linear optimization library
 * 
 * Author: Nick Goodson
 */

#pragma once

#include "Solver.hpp"


class InterfaceSnopt;


namespace gp_nlopt
{

class SolverSnopt: public Solver
{
public:

    SolverSnopt(InterfaceSnopt& problem);

    void init();

protected:

    int solve() override;


};

}
