# gtest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        DOWNLOAD_EXTRACT_TIMESTAMP false  # ensure project rebuilds if the content changes
)
FetchContent_MakeAvailable(googletest)
# set(GTEST_HAS_PTHREAD 1)  # should automatically detect this

# add tests
add_executable(kiwio_test
        testAsyncContext.cpp
        )

target_link_libraries(kiwio_test
        PRIVATE
        kiwio
        gtest_main
        )

# automatically find tests
include(GoogleTest)
gtest_discover_tests(kiwio_test)

# Alternative manual registration
#add_test(NAME asyncTest COMMAND kiwio_test)
#...