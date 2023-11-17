namespace trajopt {

enum class StatusCode : unsigned int
{
    INIT_SUCCESS = 0,
    NOT_INITIALIZED,
    BAD_INIT,
    INVALID_PROBLEM,
    INVALID_OBJECTIVE,
    INVALID_DYNAMICS,
    SOLVER_FAILURE,
    SOLVER_SUCCESS
};

}
