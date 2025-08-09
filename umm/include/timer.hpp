#pragma once
#include <chrono>

namespace umm
{
    class timer
    {
    public:
        timer() {}

        /* start or restart */
        inline void start() 
        { 
            m_start = std::chrono::steady_clock::now();
        }

        inline void stop() 
        { 
            m_end = std::chrono::steady_clock::now();
        }

        /* time elapsed in seconds */
        inline double duration() const
        {
            return 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count();
        }

    private:
        std::chrono::steady_clock::time_point m_start;
        std::chrono::steady_clock::time_point m_end;
    };
}