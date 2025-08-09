#pragma once
#include <random_number_generator.hpp>
#include <integrand.hpp>

namespace umm
{
    class monte_carlo
    {
    public:
        monte_carlo(const integrand_ptr_t& f, const std::vector<double>& mean, const matrix& cov, const moment_matching_type& mm_type)
            : m_f(f), m_mean(mean), m_cov(cov), m_dim(mean.size()), m_mm_type(mm_type)
        {
            m_rng = std::shared_ptr<random_normal_generator>(new random_normal_generator(mean, cov, m_mm_type));
        }

        double compute_integral(const size_t sample_size, const size_t seed = DEFAULT_RNG_SEED);
        double compute_sample_variance();
        double compute_variance_by_derivative_integral();
        double compute_variance_by_independent_simulations(const size_t sample_size, const size_t nb_seeds);

    private:
        std::shared_ptr<random_normal_generator> m_rng;
        std::vector<double> m_mean;
        matrix m_cov;
        size_t m_dim;
        integrand_ptr_t m_f;
        moment_matching_type m_mm_type;
        std::vector<double> m_f_vals;
        std::vector<double> m_random_normals;
    };
}