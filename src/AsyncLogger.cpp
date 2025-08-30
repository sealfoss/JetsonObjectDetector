#include "AsyncLogger.h"
#include <iostream>

using namespace std;
using namespace Logging;


AsyncLogger::AsyncLogger(string logDirectory, LogLevel minLevel) 
: Logger(logDirectory, minLevel)
{
    _isLogging = true;
    _thread = std::thread(&AsyncLogger::ProcessLogQueue, this);
}

AsyncLogger::~AsyncLogger()
{
    _queueMutex.lock();
    _isLogging = false;
    _queueMutex.unlock();
    _cv.notify_all();

    if (_thread.joinable())
    {
        _thread.join();
    }
}

void AsyncLogger::ProcessLogQueue()
{
    tuple<string, LogLevel, uint64_t> msg;
    unique_lock<mutex> lock(_condMutex);

    while (_isLogging)
    {
        _cv.wait(lock, [this] { return !_logQueue.empty() || !_isLogging; });

        while (!_logQueue.empty())
        {
            msg = _logQueue.front();
            _logQueue.pop();
            lock.unlock();
            WriteLog(msg);
            lock.lock();
        }
    }
}

void AsyncLogger::AddMessage(tuple<string, LogLevel, uint64_t> msg)
{
    try
    {
        std::unique_lock lock(_queueMutex);
        _logQueue.push(msg);
        lock.unlock();
        _cv.notify_one();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
}
