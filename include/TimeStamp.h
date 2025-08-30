#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <chrono>
#include <ctime>
#include <string>

class TimeStamp
{
public:
    static uint64_t GetUtcTimeMs()
    {
        auto now = std::chrono::system_clock::now();
        auto sinceEpoch = now.time_since_epoch();
        auto durationMs = 
            std::chrono::duration_cast<std::chrono::milliseconds>(sinceEpoch);
        uint64_t ms = durationMs.count();
        return ms;
    }
    
    static uint64_t GetUtcTimeNs()
    {
        auto now = std::chrono::system_clock::now();
        auto sinceEpoch = now.time_since_epoch();
        auto durationNs = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(sinceEpoch);
        uint64_t ns = durationNs.count();
        return ns;
    }

    static std::string GetTimeStampNs()
    {
        uint64_t ns = GetUtcTimeNs();
        return std::to_string(ns);
    }

    static std::string GetTimeStampMs()
    {
        uint64_t ms = GetUtcTimeMs();
        return std::to_string(ms);
    }
};

#endif // TIMESTAMP_H