#include "AsyncLogger.h"
#include <iostream>

using namespace std;


AsyncLogger::AsyncLogger(string logDirectory, LogLevel minLevel) 
: Logger(logDirectory, minLevel)
{
    _isLogging = true;
    _thread = std::thread(&AsyncLogger::ProcessLogQueue, this);
}

AsyncLogger::~AsyncLogger()
{
    _outboxMutex.lock();
    _isLogging = false;
    _outboxMutex.unlock();
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
        _cv.wait(lock, [this] { return !_outbox.empty() || !_isLogging; });
        _outboxMutex.lock();

        while (!_outbox.empty())
        {
            msg = _outbox.front();
            WriteLog(msg);
            _outbox.erase(_outbox.begin());
        }

        _outboxMutex.unlock();
    }
}

void AsyncLogger::AddMessage(string text, LogLevel level, uint64_t timestamp) 
{
    tuple<string, LogLevel, uint64_t> msg = make_tuple(text, level, timestamp);
    _outboxMutex.lock();
    _outbox.push_back(msg);
    _outboxMutex.unlock();
    _cv.notify_one();
}
