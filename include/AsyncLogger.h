#ifndef ASYNCLOGGER_H
#define ASYNCLOGGER_H

#include "Logger.h"

#include <shared_mutex>
#include <thread>
#include <condition_variable>

namespace Logging
{

class AsyncLogger : public Logger
{
public:
    AsyncLogger(
        std::string logDirectory="", 
        LogLevel minLevel=DEFAULT_MIN_LOG_LEVEL
    );
    ~AsyncLogger();
private:
    std::shared_mutex _queueMutex;
    std::mutex _condMutex;
    std::thread _thread;
    std::condition_variable _cv;
    bool _isLogging;
    

    void ProcessLogQueue();
    void AddMessage(std::tuple<std::string, LogLevel, uint64_t> msg) override;
};

};
#endif //ASYNCLOGGER_H