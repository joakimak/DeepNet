#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace Timing
{

using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

template<typename R>
class Timer
{
public:
    using Resolution = R;
    using Time_point = decltype(std::chrono::high_resolution_clock::now);
    using Clock = std::chrono::high_resolution_clock;

    Timer() : elapsed_{0}, running_{false}{}

    void start()
    {
        if(!running_){
            last_ = Clock::now().time_since_epoch();
            running_ = true;
        }

    }

    void stop()
    {
        if(running_){
            auto t2 = Clock::now();
            elapsed_ += std::chrono::duration_cast<Resolution>(t2 - Clock::time_point(last_)).count();
            running_ = false;
        }
    }

    void reset()
    {
        elapsed_ = 0;
        running_ = false;
    }

    double elapsed()
    {
        return elapsed_;
    }

private:
    Clock::duration last_;
    double elapsed_;
    bool running_;
};

}

#endif // TIMER_H
