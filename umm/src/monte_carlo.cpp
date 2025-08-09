#include <monte_carlo.hpp>

namespace umm
{
    double monte_carlo::compute_integral(const size_t sample_size, const size_t seed)
    {
        m_rng->seed(seed);
        m_f_vals.resize(0);
        m_f_vals.reserve(sample_size);
        m_random_normals.resize(0);
        m_rng->populate_random_normals(m_random_normals, sample_size);

        double sum = 0.0;
        for (const_vector_iterator_t it = m_random_normals.cbegin(); it != m_random_normals.cend(); it += m_dim)
        {
            const double v = (*m_f)(it);
            sum += v;
            m_f_vals.push_back(v);
        }
        return sum / sample_size;
    }

    double monte_carlo::compute_sample_variance()
    {
        double mean_f = 0.0;
        double mean_ff = 0.0;
        std::vector<double> mean_xf(m_dim, 0.0);
        std::vector<double> mean_xxf(m_dim * m_dim, 0.0);
        const_vector_iterator_t itx = m_random_normals.cbegin();
        const_vector_iterator_t itf = m_f_vals.cbegin();
        for (; itf != m_f_vals.cend(); ++itf, itx += m_dim)
        {
            const double f = *itf;
            mean_f += f;
            mean_ff += f * f;

            if (m_mm_type >= FIRST_ORDER_MOMENT_MATCHING)
            {
                for (size_t i = 0; i < m_dim; ++i)
                {
                    const double xi = *(itx + i);
                    mean_xf[i] += (xi - m_mean[i]) * f;
                }

                if (m_mm_type == SECOND_ORDER_MOMENT_MATCHING)
                {
                    for (size_t i = 0; i < m_dim; ++i)
                    {
                        for (size_t j = i; j < m_dim; ++j)
                        {
                            const double xi = *(itx + i);
                            const double xj = *(itx + j);
                            mean_xxf[m_dim * i + j] += (xi - m_mean[i]) * (xj - m_mean[j]) * f;
                        }
                    }
                }
            }
        }

        const size_t sample_size = m_f_vals.size();
        mean_f /= sample_size;
        mean_ff /= sample_size;
        for (size_t i = 0; i < m_dim; ++i)
        {
            mean_xf[i] /= sample_size;
            for (size_t j = i; j < m_dim; ++j)
            {
                mean_xxf[m_dim * i + j] /= sample_size;
                mean_xxf[m_dim * j + i] = mean_xxf[m_dim * i + j];
            }
        }

        const double var_f = mean_ff - mean_f * mean_f;
        switch (m_mm_type)
        {
        case umm::NO_MOMENT_MATCHING:
        {
            return var_f / sample_size;
        }
        case umm::FIRST_ORDER_MOMENT_MATCHING:
        {
            const matrix mat_mean_xf(mean_xf, m_dim, 1);
            const matrix& m1 = mat_mean_xf.transpose() * m_cov.inverse() * mat_mean_xf;
            return (var_f - m1(0, 0)) / sample_size;
        }
        case umm::SECOND_ORDER_MOMENT_MATCHING:
        {
            const matrix mat_mean_xf(mean_xf, m_dim, 1);
            const matrix mat_mean_xxf(mean_xxf, m_dim, m_dim);
            const matrix& m1 = mat_mean_xf.transpose() * m_cov.inverse() * mat_mean_xf;
            const matrix mat_mean_f(std::vector<double>(m_dim, mean_f));
            const matrix& m2 = sqrtm(m_cov.inverse()) * mat_mean_xxf * sqrtm(m_cov.inverse()) - mat_mean_f;
            return (var_f - m1(0, 0) - 0.5 * (m2 * m2).trace()) / sample_size;
        }
        default:
            THROW("unexpected exception");
        }
    }

    double monte_carlo::compute_variance_by_derivative_integral()
    {
        double mean_f = 0.0;
        double mean_ff = 0.0;
        const size_t sample_size = m_f_vals.size();
        const_vector_iterator_t itf = m_f_vals.cbegin();
        for (auto itf = m_f_vals.cbegin(); itf != m_f_vals.cend(); ++itf)
        {
            const double f = *itf;
            mean_f += f;
            mean_ff += f * f;
        }
        mean_f /= sample_size;
        mean_ff /= sample_size;
        const double var_f = mean_ff - mean_f * mean_f;

        const double dx = 0.01;
        std::vector<double> mean_df(m_dim, 0.0);
        std::vector<double> mean_ddf(m_dim * m_dim, 0.0);

        if (m_mm_type >= FIRST_ORDER_MOMENT_MATCHING)
        {
            for (size_t i = 0; i < m_dim; ++i)
            {
                std::vector<double> mean_up(m_mean);
                std::vector<double> mean_dn(m_mean);
                mean_up[i] += dx;
                mean_dn[i] -= dx;
                monte_carlo mc_up(m_f, mean_up, m_cov, m_mm_type);
                monte_carlo mc_dn(m_f, mean_dn, m_cov, m_mm_type);
                const double mean_f_up = mc_up.compute_integral(sample_size);
                const double mean_f_dn = mc_dn.compute_integral(sample_size);
                mean_df[i] = (mean_f_up - mean_f_dn) / (2.0 * dx);
            }
        }

        if (m_mm_type == SECOND_ORDER_MOMENT_MATCHING)
        {
            for (size_t i = 0; i < m_dim; ++i)
            {
                for (size_t j = i; j < m_dim; ++j)
                {
                    std::vector<double> mean_up_up(m_mean);
                    std::vector<double> mean_up_dn(m_mean);
                    std::vector<double> mean_dn_up(m_mean);
                    std::vector<double> mean_dn_dn(m_mean);
                    mean_up_up[i] += dx;
                    mean_up_up[j] += dx;
                    mean_up_dn[i] += dx;
                    mean_up_dn[j] -= dx;
                    mean_dn_up[i] -= dx;
                    mean_dn_up[j] += dx;
                    mean_dn_dn[i] -= dx;
                    mean_dn_dn[j] -= dx;
                    monte_carlo mc_up_up(m_f, mean_up_up, m_cov, m_mm_type);
                    monte_carlo mc_up_dn(m_f, mean_up_dn, m_cov, m_mm_type);
                    monte_carlo mc_dn_up(m_f, mean_dn_up, m_cov, m_mm_type);
                    monte_carlo mc_dn_dn(m_f, mean_dn_dn, m_cov, m_mm_type);

                    const double mean_f_up_up = mc_up_up.compute_integral(sample_size);
                    const double mean_f_up_dn = mc_up_dn.compute_integral(sample_size);
                    const double mean_f_dn_up = mc_dn_up.compute_integral(sample_size);
                    const double mean_f_dn_dn = mc_dn_dn.compute_integral(sample_size);
                    mean_ddf[m_dim * i + j] = (mean_f_up_up - mean_f_up_dn - mean_f_dn_up + mean_f_dn_dn) / (4.0 * dx * dx);
                    mean_ddf[m_dim * j + i] = mean_ddf[m_dim * i + j];
                }
            }
        }

        const matrix mat_mean_df(mean_df, 1, m_dim);
        const matrix mat_mean_ddf(mean_ddf, m_dim, m_dim);
        const matrix& m1 = mat_mean_df * m_cov * mat_mean_df.transpose();
        const matrix& m2 = m_cov * mat_mean_ddf;
        return (var_f - m1(0, 0) - 0.5 * (m2 * m2).trace()) / sample_size;
    }

    double monte_carlo::compute_variance_by_independent_simulations(const size_t sample_size, const size_t nb_seeds)
    {
        std::mt19937_64 rng;
        rng.seed(2025);
        std::uniform_int_distribution<> dist(10, 10000);
        std::vector<size_t> seeds;
        for (size_t i = 0; i < nb_seeds; ++i)
            seeds.push_back((size_t)dist(rng));
        
        std::vector<double> results;
        for (const size_t seed : seeds)
            results.push_back(compute_integral(sample_size, seed));
        double mean_res = 0.0;
        double mean_res_sq = 0.0;
        for (const double r : results)
        {
            mean_res += r;
            mean_res_sq += r * r;
        }
        mean_res /= nb_seeds;
        mean_res_sq /= nb_seeds;
        return mean_res_sq - mean_res * mean_res;
    }
}