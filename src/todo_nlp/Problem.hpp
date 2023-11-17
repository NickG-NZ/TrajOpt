/**
 * Non-linear optimization problem
 * Holds the objective, constraints and decision variables
 * 
 * Author: Nick Goodson
 */

#include <functional>
#include <vector>


class ConstraintGroup;
class VariableGroup;


class Problem
{
public:
    Problem();

    void addObjective();

private:
    std::function<void(double *, double *)> objective;
    std::vector<ConstraintGroup *> constraints;
    std::vector<VariableGroup *> variables;

}
