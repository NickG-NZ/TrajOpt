
add_library(milqr SHARED
  milqr.cpp
  milqr.h
  milqr_utils.cpp
  milqr_utils.h
  milqr_utils.hp
)

target_include_directories(milqr
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


