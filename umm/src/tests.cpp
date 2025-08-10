#include <tests.hpp>
#include <timer.hpp>
#include <monte_carlo.hpp>
#include <fstream>

namespace umm
{
    void tests::test_universal_moment_matching(
        const std::vector<double>& vols, 
        const matrix& corr, 
        const double r, 
        const double strike, 
        const double tenor, 
        const double barrier
    )
    {
        std::cout << "running test_universal_moment_matching..." << std::endl;
        std::ofstream file;
        const std::string fname = umm_root_dir() + "/tests/test_results_universal_moment_matching.csv";
        file.open(fname);
        file << "SampleSize,PV(IID),PV(MM1),PV(MM2),SE(IID),SE(MM1),SE(MMSeed1),SE(MM2),SE(MMSeed2)" << std::endl;

        const size_t dim = vols.size();
        const std::vector<double> mean(dim, 0.0);
        integrand_ptr_t f = std::shared_ptr<ki_put>(new ki_put(vols, r, strike, tenor, barrier));
        monte_carlo mc(f, mean, corr, NORMAL_DISTRIBUTION, NO_MOMENT_MATCHING);
        monte_carlo mm1(f, mean, corr, NORMAL_DISTRIBUTION, FIRST_ORDER_MOMENT_MATCHING);
        monte_carlo mm2(f, mean, corr, NORMAL_DISTRIBUTION, SECOND_ORDER_MOMENT_MATCHING);
        const size_t nb_seeds = 500;
        const std::vector<size_t> sample_sizes { 10000, 20000, 40000, 80000, 160000, 320000, 500000 };
        for (const size_t sample_size : sample_sizes)
        {
            const double v_mc = mc.compute_integral(sample_size);
            const double se_mc = std::sqrt(mc.compute_sample_variance());

            const double v_mm1 = mm1.compute_integral(sample_size);
            const double se_mm1 = std::sqrt(mm1.compute_sample_variance());
            const double se_seed_mm1 = std::sqrt(mm1.compute_variance_by_independent_simulations(sample_size, nb_seeds));

            const double v_mm2 = mm2.compute_integral(sample_size);
            const double se_mm2 = std::sqrt(mm2.compute_sample_variance());
            const double se_seed_mm2 = std::sqrt(mm2.compute_variance_by_independent_simulations(sample_size, nb_seeds));


            
            std::cout << "sample size: " << sample_size << "; " << "IID: " << v_mc << " (" << se_mc << "); "
                << "MM1: " << v_mm1 << " (" << se_mm1 << ", " << se_seed_mm1 << "); "
                << "MM2: " << v_mm2 << " (" << se_mm2 << ", " << se_seed_mm2 << ") " << std::endl;

            file << sample_size << "," << v_mc << "," << v_mm1 << "," << v_mm2 << "," << se_mc << "," << se_mm1 << "," 
                << se_seed_mm1 << "," << se_mm2 << "," << se_seed_mm2 << std::endl;
        }
        file.close();
        std::cout << TEST_SEPARATOR << std::endl;
    }

    void tests::test_variance_estimation_time()
    {
        std::cout << "running test_variance_estimation_time..." << std::endl;
        std::ofstream file;
        const std::string fname = umm_root_dir() + "/tests/test_results_variance_estimation_time.csv";
        file.open(fname);
        file << "Dimension,CycleTime(IID),CycleTime(MM1),CycleTime(MM2),CycleTime(MM1+SE),CycleTime(MM1+DSE)," 
            << "CycleTime(MM1+SSE),CycleTime(MM2+SE),CycleTime(MM2+DSE),CycleTime(MM2+SSE)"  
            << std::endl;
        const std::vector<size_t> dims{ 1, 2, 4, 8, 15, 20, 30, 40 };
        const size_t sample_size = 50000;
        const size_t nb_seeds = 100;
        timer clock;
        for (const size_t dim : dims)
        {
            const matrix corr(std::vector<double>(dim, 1.0));
            const std::vector<double> mean(dim, 0.0);
            const std::vector<double> vols(dim, 0.3);
            const double r = 0.05;
            const double strike = 1.0;
            const double tenor = 1.0;
            const double barrier = 0.8;
            integrand_ptr_t f = std::shared_ptr<ki_put>(new ki_put(vols, r, strike, tenor, barrier));
            monte_carlo mc(f, mean, corr, NORMAL_DISTRIBUTION, NO_MOMENT_MATCHING);
            monte_carlo mm1(f, mean, corr, NORMAL_DISTRIBUTION, FIRST_ORDER_MOMENT_MATCHING);
            monte_carlo mm2(f, mean, corr, NORMAL_DISTRIBUTION, SECOND_ORDER_MOMENT_MATCHING);

            clock.start();
            const double pv_iid = mc.compute_integral(sample_size);
            clock.stop();
            const double ct_iid = clock.duration();

            clock.start();
            const double se_iid = mc.compute_sample_variance();
            clock.stop();
            const double ct_iid_se = ct_iid + clock.duration();

            clock.start();
            const double pv_mm1 = mm1.compute_integral(sample_size);
            clock.stop();
            const double ct_mm1 = clock.duration();

            clock.start();
            const double se_mm1 = mm1.compute_sample_variance();
            clock.stop();
            const double ct_mm1_se = ct_mm1 + clock.duration();

            clock.start();
            const double se_mm1_deriv = mm1.compute_variance_by_derivative_integral();
            clock.stop();
            const double ct_mm1_deriv_se = ct_mm1 + clock.duration();

            clock.start();
            const double se_mm1_seed = mm1.compute_variance_by_independent_simulations(sample_size, nb_seeds);
            clock.stop();
            const double ct_mm1_seed_se = ct_mm1 + clock.duration();

            clock.start();
            const double pv_mm2 = mm2.compute_integral(sample_size);
            clock.stop();
            const double ct_mm2 = clock.duration();

            clock.start();
            const double se_mm2 = mm2.compute_sample_variance();
            clock.stop();
            const double ct_mm2_se = ct_mm2 + clock.duration();

            clock.start();
            const double se_mm2_deriv = mm2.compute_variance_by_derivative_integral();
            clock.stop();
            const double ct_mm2_deriv_se = ct_mm2 + clock.duration();

            clock.start();
            const double se_mm2_seed = mm2.compute_variance_by_independent_simulations(sample_size, nb_seeds);
            clock.stop();
            const double ct_mm2_seed_se = ct_mm2 + clock.duration();

            file << dim << "," << ct_iid << "," << ct_mm1 << "," << ct_mm2 << "," << ct_mm1_se << "," << ct_mm1_deriv_se 
                << "," << ct_mm1_seed_se << "," << ct_mm2_se << "," << ct_mm2_deriv_se << "," << ct_mm2_seed_se 
                << std::endl;

            std::cout << "n: " << dim << ", IID: " << ct_iid << ", MM1: " << ct_mm1 << ", MM2: " << ct_mm2 
                << ", MM1+SE: " << ct_mm1_se << ", MM1+DSE: " << ct_mm1_deriv_se << ", MM1+SSE: " << ct_mm1_seed_se
                << ", MM2+SE: " << ct_mm2_se << ", MM2+DSE: " << ct_mm2_deriv_se << ", MM2+SSE: " << ct_mm2_seed_se 
                << std::endl;
        }
        file.close();
        std::cout << TEST_SEPARATOR << std::endl;
    }

    void tests::test_non_linear_moment_matching()
    {
        std::cout << "running test_non_linear_moment_matching..." << std::endl;
        std::ofstream file;
        const std::string fname = umm_root_dir() + "/tests/test_results_non_linear_moment_matching.csv";
        file.open(fname);
        file << "SampleSize,PV(IID),PV(MM1),PV(NLMM1),SE(IID),SE(MMSeed1),SE(NLMMSeed1),SE(MMSeed2),SE(NLMMSeed2)" << std::endl;

        class test_func : public integrand
        {
            virtual double operator()(const const_vector_iterator_t& it) const
            {
                const double x = *it;
                const double x_min = 0.25;
                const double x_max = 0.75;
                if (x <= x_min || x >= x_max)
                    return 0.0;
                return std::exp(-0.25 / std::sqrt((x - x_min)) - 0.25 / std::sqrt((x_max - x)));
            }
        };

        const double lamb = 1.0;
        const std::vector<double> mean(1, 1.0 / lamb);
        const std::vector<double> cov(1, 1.0 / (lamb * lamb));
        integrand_ptr_t f(new test_func());
        monte_carlo mc(f, mean, cov, EXPONENTIAL_DISTRIBUTION, NO_MOMENT_MATCHING, LINEAR_MOMENT_MATCHING);
        monte_carlo mm1(f, mean, cov, EXPONENTIAL_DISTRIBUTION, FIRST_ORDER_MOMENT_MATCHING, LINEAR_MOMENT_MATCHING);
        monte_carlo mm2(f, mean, cov, EXPONENTIAL_DISTRIBUTION, SECOND_ORDER_MOMENT_MATCHING, LINEAR_MOMENT_MATCHING);
        monte_carlo nlmm1(f, mean, cov, EXPONENTIAL_DISTRIBUTION, FIRST_ORDER_MOMENT_MATCHING, NON_LINEAR_MOMENT_MATCHING);
        monte_carlo nlmm2(f, mean, cov, EXPONENTIAL_DISTRIBUTION, SECOND_ORDER_MOMENT_MATCHING, NON_LINEAR_MOMENT_MATCHING);

        const size_t nb_seeds = 500;
        const std::vector<size_t> sample_sizes { 10000, 20000, 40000, 80000, 160000, 320000, 500000 };
        for (const size_t sample_size : sample_sizes)
        {
            const double v_mc = mc.compute_integral(sample_size);
            const double se_mc = std::sqrt(mc.compute_sample_variance());

            const double v_mm1 = mm1.compute_integral(sample_size);
            const double se_seed_mm1 = std::sqrt(mm1.compute_variance_by_independent_simulations(sample_size, nb_seeds));

            const double v_nlmm1 = nlmm1.compute_integral(sample_size);
            const double se_seed_nlmm1 = std::sqrt(nlmm1.compute_variance_by_independent_simulations(sample_size, nb_seeds));

            const double v_mm2 = mm2.compute_integral(sample_size);
            const double se_seed_mm2 = std::sqrt(mm2.compute_variance_by_independent_simulations(sample_size, nb_seeds));

            const double v_nlmm2 = nlmm2.compute_integral(sample_size);
            const double se_seed_nlmm2 = std::sqrt(nlmm2.compute_variance_by_independent_simulations(sample_size, nb_seeds));

            std::cout << "sample size: " << sample_size << "; " << "IID: " << v_mc << " (" << se_mc << "); "
                << "MM1: " << v_mm1 << " (" << se_seed_mm1 << "); " << "NLMM1: " << v_nlmm1 << " (" << se_seed_nlmm1 << ") " 
                << "MM2: " << v_mm2 << " (" << se_seed_mm2 << "); " << "NLMM2: " << v_nlmm2 << " (" << se_seed_nlmm2 << ") "
                << std::endl;

            file << sample_size << "," << v_mc << "," << v_mm1 << "," << v_nlmm1 
                << "," << se_mc << "," << se_seed_mm1 << "," << se_seed_nlmm1 << "," << se_seed_mm2 << "," << se_seed_nlmm2
                << std::endl;
        }
        file.close();
        std::cout << TEST_SEPARATOR << std::endl;
    }

    void tests::run_all_tests()
    {
        const double r = 0.05;
        const double strike = 1.0;
        const double tenor = 1.0;
        const double barrier = 0.8;

        std::vector<double> vols(1, 0.3);
        matrix corr(std::vector<double>(1, 1.0), 1, 1);
        test_universal_moment_matching(vols, corr, r, strike, tenor, barrier);

        vols = std::vector<double>{ 0.3, 0.2, 0.4 };
        corr = matrix(std::vector<double>{ 1.0, 0.3, 0.1, 0.3, 1.0, 0.5, 0.1, 0.5, 1.0 }, vols.size(), vols.size());
        test_universal_moment_matching(vols, corr, r, strike, tenor, barrier);

        test_non_linear_moment_matching();
        
        //test_variance_estimation_time();
    }
}