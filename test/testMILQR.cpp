#include "milqr/MILQR.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include <chrono>

// ensure Gtest configuration is thread-safe
#ifdef GTEST_IS_THREADSAFE
#if(GTEST_IS_THREADSAFE == 0)
#error failed to find pthreads
#endif
#endif

class TrajOptFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        
    }

    void TearDown() override
    {
    }

    trajopt::milqr solver_;
};


TEST_F(TrajOptFixture, Solve) {
    // EXPECT_EQ
    // ASSERT_EQ


  
    ASSERT_EQ(ec, kiwio::ErrorCode::NoError);

}