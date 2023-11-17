/**
 * Wraps the Problem object such that it can be passed to 
 * the SNOPT cpp interface
 * 
 * This object is passed to SolverSnopt which performs the optimization
 */

#pragma once

#include "Problem.hpp"


namespace gp_nlopt
{


class InterfaceSnopt
{
public:
    InterfaceSnopt(Problem *nlp);


    static void objectiveAndConstraints(int* mode);  // TODO: add remaining arguments needed for SNOPTA

protected:
    static Problem *nlp_;  // static allows access to this object from static member func

};


}