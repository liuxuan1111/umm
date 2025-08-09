#pragma once
#include <preliminaries.hpp>
#include <vector>
#include <iterator>
#include <memory>

namespace umm
{
    typedef std::vector<double>::const_iterator const_vector_iterator_t;
    typedef std::vector<double>::iterator vector_iterator_t;

    class integrand
    {
    public:
        integrand() {};

        virtual double operator()(const const_vector_iterator_t& it) const = 0;
        virtual ~integrand() {};
    };

    typedef std::shared_ptr<integrand> integrand_ptr_t;

    class ki_put : public integrand
    {
    public:
        ki_put(const std::vector<double>& vols, const double r, const double strike, const double tenor, const double barrier)
            : m_vols(vols), m_nb_assets(vols.size()), m_r(r), m_strike(strike), m_tenor(tenor), m_barrier(barrier)
        {}

        virtual double operator()(const const_vector_iterator_t& it) const;

    private:
        std::vector<double> m_vols;
        size_t m_nb_assets;
        double m_r;
        double m_strike;
        double m_tenor;
        double m_barrier;
    };
}