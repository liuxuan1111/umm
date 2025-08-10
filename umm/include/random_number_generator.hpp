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

    enum moment_matching_scheme
    {
        LINEAR_MOMENT_MATCHING,
        NON_LINEAR_MOMENT_MATCHING
    };

    class moment_matching
    {
    public:
        moment_matching(const std::vector<double>& mean, const matrix& cov);
        void apply_moment_mathcing(std::vector<double>& out, const moment_matching_type& mm_type) const;

    private:
        std::vector<double> m_mean;
        matrix m_cov;
        matrix m_sqrt_cov;
        size_t m_dim;
    };

    class random_sample_generator_base
    {
    public:
        virtual ~random_sample_generator_base() {};
        virtual void seed(const size_t seed = DEFAULT_RNG_SEED) = 0;
        virtual void populate_random_samples(std::vector<double>& out, const size_t sample_size) = 0;
    };

    typedef std::shared_ptr<random_sample_generator_base> random_sample_generator_ptr_t;

    enum random_distribution_label
    {
        NORMAL_DISTRIBUTION,
        EXPONENTIAL_DISTRIBUTION,
        INVALID_DISTRIBUTION
    };

    class random_normal_generator : public random_sample_generator_base
    {
    public:
        random_normal_generator(const std::vector<double>& mean, const matrix& cov, const moment_matching_type& mm_type = NO_MOMENT_MATCHING);

        virtual void populate_random_samples(std::vector<double>& out, const size_t sample_size);

        virtual void seed(const size_t seed = DEFAULT_RNG_SEED)
        {
            m_rng.seed(seed);
        }

    private:
        std::vector<double> m_mean;
        matrix m_cov;
        matrix m_sqrt_cov;
        size_t m_dim;
        moment_matching_type m_mm_type;
        moment_matching m_mm;
        standard_normal_generator m_rng;
    };

    class random_exponential_generator : public random_sample_generator_base
    {
    public:
        random_exponential_generator(const double lamb, const moment_matching_scheme& mm_scheme = LINEAR_MOMENT_MATCHING, const moment_matching_type& mm_type = NO_MOMENT_MATCHING);

        virtual void seed(const size_t seed = DEFAULT_RNG_SEED)
        {
            m_rng.seed(seed);
            m_seed = seed;
        }
        
        virtual void populate_random_samples(std::vector<double>& rn_out, const size_t size);

        double cdf(const double x);
        double inv_cdf(const double x);

    private:
        size_t m_seed;
        std::mt19937_64 m_rng;
        double m_lamb;
        std::exponential_distribution<double> m_dist;
        moment_matching_scheme m_mm_scheme;
        moment_matching_type m_mm_type;
        moment_matching m_mm;
    };
}