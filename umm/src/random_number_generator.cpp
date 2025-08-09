#include <random_number_generator.hpp>

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

    std::vector<double> standard_normal_generator::generator_standard_normals(const size_t size)
    {
        std::vector<double> rn_out;
        populate_standard_normals(rn_out, size);
        return rn_out;
    }

    random_normal_generator::random_normal_generator(const std::vector<double>& mean, const matrix& cov, const moment_matching_type& mm_type)
        : m_mean(mean), m_cov(cov), m_rho(sqrtm(cov)), m_dim(mean.size()), m_rng(standard_normal_generator()), m_mm_type(mm_type)
    {
        VERIFY(m_cov.rows() == m_dim && m_cov.cols() == m_dim, "inconsistent dimension");
    }

    void random_normal_generator::populate_random_normals(std::vector<double>& out, const size_t sample_size)
    {
        m_rng.populate_standard_normals(out, m_dim * sample_size);
        std::vector<double> arr_temp(m_dim);
        for (size_t k = 0; k < sample_size; ++k)
        {
            for (size_t i = 0; i < m_dim; ++i)
            {
                arr_temp[i] = 0.0;
                for (size_t j = 0; j < m_dim; ++j)
                    arr_temp[i] += m_rho(i, j) * out[m_dim * k + j];
            }
            for (size_t i = 0; i < m_dim; ++i)
                out[m_dim * k + i] = m_mean[i] + arr_temp[i];
        }

        std::vector<double> sample_mean(m_dim, 0.0);
        std::vector<double> sample_var_flatten(m_dim * m_dim, 0.0);
        if (m_mm_type >= FIRST_ORDER_MOMENT_MATCHING)
        {
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
            if (m_mm_type == SECOND_ORDER_MOMENT_MATCHING)
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
                const matrix& Q = m_rho * sqrtm(sample_var).inverse();

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
    }
}