#pragma once
#include <random_number_generator.hpp>
#include <integrand.hpp>

namespace umm
{
    class monte_carlo
    {
    public:
        monte_carlo(
            const integrand_ptr_t& f, 
            const std::vector<double>& mean, 
            const matrix& cov, 
            const random_distribution_label& dist_label, 
            const moment_matching_type& mm_type,
            const moment_matching_scheme& mm_scheme = LINEAR_MOMENT_MATCHING
        )
            : m_f(f), m_mean(mean), m_cov(cov), m_dim(mean.size()), m_mm_type(mm_type)
        {
            switch (dist_label)
            {
            case NORMAL_DISTRIBUTION:
            {
                m_rng = random_sample_generator_ptr_t(new random_normal_generator(mean, cov, m_mm_type));
                break;
            }
            case EXPONENTIAL_DISTRIBUTION:
            {
                m_rng = random_sample_generator_ptr_t(new random_exponential_generator(1.0 / m_mean.front(), mm_scheme, m_mm_type));
                break;
            }
            default:
                THROW("not implemented");
            }
            
        }

        double compute_integral(const size_t sample_size, const size_t seed = DEFAULT_RNG_SEED);
        double compute_sample_variance();
        double compute_variance_by_derivative_integral();
        double compute_variance_by_independent_simulations(const size_t sample_size, const size_t nb_seeds);

    private:
        random_sample_generator_ptr_t m_rng;
        std::vector<double> m_mean;
        matrix m_cov;
        size_t m_dim;
        integrand_ptr_t m_f;
        moment_matching_type m_mm_type;
        std::vector<double> m_f_vals;
        std::vector<double> m_random_normals;
    };
}