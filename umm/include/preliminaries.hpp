#pragma once
#include <stdexcept>
#include <limits>
#include <iostream>
#include <string>

namespace umm
{
/* macros */
#define PRINT_ERROR(msg) std::cout << "\033[1;31m" << msg  << "(" << __FILE__  <<  ", line " << __LINE__ << ")" << "\033[1;0m" << std::endl
#define THROW(msg) { PRINT_ERROR(msg); throw std::runtime_error(msg); }
#define VERIFY(cond, msg) if (!(cond)) THROW(msg);

/* constants */
    const size_t DEFAULT_RNG_SEED = 202508;
    const double INF = std::numeric_limits<double>::infinity();
    
    inline const std::string& umm_root_dir()
    {
        static std::string path("");

        if (path.empty())
        {
            path = std::string(__FILE__);
            const size_t levels_up = 2;
            size_t loc = path.size();
            for (size_t i = 0; i < levels_up; ++i)
            {
                size_t loc = path.find_last_of("\\/");
                path.erase(loc, path.size() - loc);
            }
        }

        return path;
    }

}
