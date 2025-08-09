#include <integrand.hpp>

namespace umm
{
    double ki_put::operator()(const const_vector_iterator_t& it) const
    {
        double perf = INF;
        const_vector_iterator_t cit = it;
        for (size_t i = 0; i < m_nb_assets; ++i, ++cit)
        {
            const double s = std::exp((m_r - 0.5 * m_vols[i] * m_vols[i]) * m_tenor + std::sqrt(m_tenor) * m_vols[i] * (*cit));
            perf = std::min(perf, s);
        }
        return perf <= m_barrier ? std::exp(-m_r * m_tenor) * std::max(0.0, m_strike - perf) : 0.0;
    }
}