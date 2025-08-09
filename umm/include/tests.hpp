#pragma once
#include <ublas.hpp>
#include <vector>
#include <string>

namespace umm
{
    namespace tests
    {
        const std::string TEST_SEPARATOR(100, '=');

        void test_universal_moment_matching(
            const std::vector<double>& vols,
            const matrix& corr,
            const double r,
            const double strike,
            const double tenor,
            const double barrier
        );

        void test_variance_estimation_time();

        void run_all_tests();
    }
}