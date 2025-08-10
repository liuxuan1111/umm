#include <random_number_generator.hpp>
#include <speical_functions.hpp>
#include <algorithm>

namespace umm
{
    standard_normal_generator::standard_normal_generator(const size_t seed) : m_seed(seed)
    {
        m_rng = std::mt19937_64(m_seed);
        m_dist = std::normal_distribution<double>(0.0, 1.0);
    }

    void standard_normal_generator::populate_standard_normals(std::vector<double>& rn_out, const size_t size)
    {
        rn_out.resize(0);
        rn_out.reserve(size);
        for (size_t i = 0; i < size; ++i)
            rn_out.push_back(m_dist(m_rng));
    }

    moment_matching::moment_matching(const std::vector<double>& mean, const matrix& cov)
        : m_mean(mean), m_cov(cov), m_sqrt_cov(sqrtm(cov)), m_dim(mean.size())
    {
        VERIFY(m_cov.rows() == m_dim && m_cov.cols() == m_dim, "inconsistent dimension");
    }

    void moment_matching::apply_moment_mathcing(std::vector<double>& out, const moment_matching_type& mm_type) const
    {
        if (mm_type == NO_MOMENT_MATCHING)
            return;

        const size_t sample_size = out.size() / m_dim;
        std::vector<double> arr_temp(m_dim);
        for (size_t k = 0; k < sample_size; ++k)
        {
            for (size_t i = 0; i < m_dim; ++i)
            {
                arr_temp[i] = 0.0;
                for (size_t j = 0; j < m_dim; ++j)
                    arr_temp[i] += m_sqrt_cov(i, j) * out[m_dim * k + j];
            }
            for (size_t i = 0; i < m_dim; ++i)
                out[m_dim * k + i] = m_mean[i] + arr_temp[i];
        }

        std::vector<double> sample_mean(m_dim, 0.0);
        std::vector<double> sample_var_flatten(m_dim * m_dim, 0.0);
        for (size_t k = 0; k < sample_size; ++k)
        {
            for (size_t i = 0; i < m_dim; ++i)
                sample_mean[i] += out[m_dim * k + i];
        }
        for (size_t i = 0; i < m_dim; ++i)
            sample_mean[i] /= sample_size;

        /* subtract sample mean */
        for (size_t k = 0; k < sample_size; ++k)
        {
            for (size_t i = 0; i < m_dim; ++i)
                out[m_dim * k + i] -= sample_mean[i];
        }

        /* match variance */
        if (mm_type == SECOND_ORDER_MOMENT_MATCHING)
        {
            for (size_t k = 0; k < sample_size; ++k)
            {
                for (size_t i = 0; i < m_dim; ++i)
                {
                    for (size_t j = i; j < m_dim; ++j)
                        sample_var_flatten[m_dim * i + j] += out[m_dim * k + i] * out[m_dim * k + j];
                }
            }
            for (size_t i = 0; i < m_dim; ++i)
            {
                for (size_t j = i; j < m_dim; ++j)
                {
                    sample_var_flatten[m_dim * i + j] = sample_var_flatten[m_dim * i + j] / sample_size;
                    sample_var_flatten[m_dim * j + i] = sample_var_flatten[m_dim * i + j];
                }
            }

            const matrix sample_var(sample_var_flatten, m_dim, m_dim);
            const matrix& Q = m_sqrt_cov * sqrtm(sample_var).inverse();

            for (size_t k = 0; k < sample_size; ++k)
            {
                for (size_t i = 0; i < m_dim; ++i)
                {
                    arr_temp[i] = 0.0;
                    for (size_t j = 0; j < m_dim; ++j)
                        arr_temp[i] += Q(i, j) * out[m_dim * k + j];
                }
                for (size_t i = 0; i < m_dim; ++i)
                    out[m_dim * k + i] = arr_temp[i];
            }
        }

        /* add back theoretical mean */
        for (size_t k = 0; k < sample_size; ++k)
        {
            for (size_t i = 0; i < m_dim; ++i)
                out[m_dim * k + i] += m_mean[i];
        }
    }

    random_normal_generator::random_normal_generator(const std::vector<double>& mean, const matrix& cov, const moment_matching_type& mm_type)
        : m_mean(mean), m_cov(cov), m_sqrt_cov(sqrtm(cov)), m_dim(mean.size()), m_rng(standard_normal_generator()), m_mm(mean, cov), m_mm_type(mm_type)
    {
        VERIFY(m_cov.rows() == m_dim && m_cov.cols() == m_dim, "inconsistent dimension");
    }

    void random_normal_generator::populate_random_samples(std::vector<double>& out, const size_t sample_size)
    {
        m_rng.populate_standard_normals(out, m_dim * sample_size);
        m_mm.apply_moment_mathcing(out, m_mm_type);
    }

    random_exponential_generator::random_exponential_generator(
        const double lamb, 
        const moment_matching_scheme& mm_scheme, 
        const moment_matching_type& mm_type
    )
        : 
        m_lamb(lamb),
        m_seed(DEFAULT_RNG_SEED), 
        m_mm_scheme(mm_scheme), 
        m_mm_type(mm_type), 
        m_mm(moment_matching(std::vector<double>(1, 1.0 / lamb), std::vector<double>(1, 1.0 / lamb)))
    {
        m_rng = std::mt19937_64(m_seed);
        m_dist = std::exponential_distribution<double>(1.0);
        if (m_mm_scheme == NON_LINEAR_MOMENT_MATCHING)
            m_mm = moment_matching(std::vector<double>(1, 0.0), std::vector<double>(1, 1.0));
    }

    void random_exponential_generator::populate_random_samples(std::vector<double>& rn_out, const size_t size)
    {
        rn_out.resize(0);
        rn_out.reserve(size);
        for (size_t i = 0; i < size; ++i)
            rn_out.push_back(m_dist(m_rng));

        if (m_mm_type == NO_MOMENT_MATCHING)
            return;

        switch (m_mm_scheme)
        {
        case moment_matching_scheme::LINEAR_MOMENT_MATCHING:
        {
            m_mm.apply_moment_mathcing(rn_out, m_mm_type);
            return;
        }
        case moment_matching_scheme::NON_LINEAR_MOMENT_MATCHING:
        {
            auto trans = [this](const double x) 
            { 
                return special_functions::inverse_normal_cdf(cdf(x)); 
            };

            auto inv_trans = [this](const double x)
            {
                return inv_cdf(special_functions::normal_cdf(x));
            };

            std::transform(rn_out.begin(), rn_out.end(), rn_out.begin(), trans);
            m_mm.apply_moment_mathcing(rn_out, m_mm_type);
            std::transform(rn_out.begin(), rn_out.end(), rn_out.begin(), inv_trans);
            return;
        }
        default:
            return;
        }
    }

    double random_exponential_generator::cdf(const double x)
    {
        return 1.0 - std::exp(-m_lamb * x);
    }

    double random_exponential_generator::inv_cdf(const double x)
    {
        return -std::log(1.0 - x) / m_lamb;
    }
}