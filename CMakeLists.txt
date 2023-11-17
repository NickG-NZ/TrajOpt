cmake_minimum_required(VERSION 3.20)
project(trajOpt VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(CMAKE_VERBOSE_MAKEFILE ON FORCE)

add_subdirectory(src)

# Testing
include(CTest)  # automatically calls enable_testing()
if(BUILD_TESTING)  # defined by CTest include
    add_subdirectory(test)
endif()

# Examples
option(BUILD_EXAMPLES OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

