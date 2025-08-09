#pragma once
#include <preliminaries.hpp>
#include <ublas.hpp>
#include <random>

namespace umm
{
    class standard_normal_generator
    {
    public:
        standard_normal_generator(const size_t seed = DEFAULT_RNG_SEED);

        void seed(const size_t seed = DEFAULT_RNG_SEED) 
        { 
            m_rng.seed(seed); 
            m_seed = seed;
        }

        void reset() { seed(m_seed); }
        void populate_standard_normals(std::vector<double>& rn_out, const size_t size);
        std::vector<double> generator_standard_normals(const size_t size);

    private:
        size_t m_seed;
        std::mt19937_64 m_rng;
        std::normal_distribution<double> m_dist;
    };

    enum moment_matching_type
    {
        NO_MOMENT_MATCHING,
        FIRST_ORDER_MOMENT_MATCHING,
        SECOND_ORDER_MOMENT_MATCHING
    };

    class random_normal_generator
    {
    public:
        random_normal_generator(const std::vector<double>& mean, const matrix& cov, const moment_matching_type& mm_type = NO_MOMENT_MATCHING);

        void populate_random_normals(std::vector<double>& out, const size_t sample_size);

        void seed(const size_t seed = DEFAULT_RNG_SEED)
        {
            m_rng.seed(seed);
        }

    private:
        std::vector<double> m_mean;
        matrix m_cov;
        matrix m_rho;
        size_t m_dim;
        moment_matching_type m_mm_type;
        standard_normal_generator m_rng;
    };
}