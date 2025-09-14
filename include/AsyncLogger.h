#ifndef ASYNCLOGGER_H
#define ASYNCLOGGER_H

#include "Logger.h"
#include <mutex>
#include <thread>
#include <condition_variable>


class AsyncLogger : public Logger
{
public:
    AsyncLogger(
        std::string logDirectory="", 
        LogLevel minLevel=DEFAULT_MIN_LOG_LEVEL
    );
    ~AsyncLogger();
private:
    std::mutex _outboxMutex;
    std::mutex _condMutex;
    std::thread _thread;
    std::condition_variable _cv;
    bool _isLogging;
    

    void ProcessLogQueue();
    void AddMessage(
        std::string text, LogLevel level, uint64_t timestamp
    ) override;
};

#endif //ASYNCLOGGER_H