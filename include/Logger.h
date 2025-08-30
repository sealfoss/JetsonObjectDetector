#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <queue>
#include <utility>
#include <tuple>

void LogErr(const std::string& message);
void LogWarn(const std::string& message);
void LogDebug(const std::string& message);
void LogInfo(const std::string& message);
void LogTrace(const std::string& message);

enum LogLevel
{
    LOG_VERBOSE = 0,
    LOG_INFO = 1,
    LOG_DEBUG = 2,
    LOG_WARNING = 3,
    LOG_ERROR = 4
};

#define DEFAULT_MIN_LOG_LEVEL LOG_INFO
#define DEFAULT_PRINT_TO_CONSOLE true

class Logger
{
public:
    Logger(
        std::string logDirectoryPath="", 
        LogLevel minLevel=DEFAULT_MIN_LOG_LEVEL,
        bool printToConsole=DEFAULT_PRINT_TO_CONSOLE
    );
    virtual ~Logger();
    static void Log(const std::string& message, const LogLevel level);
    virtual void SetPrintToConsole(bool print) { _printConsole = print; }
    virtual bool IsPrintingToConsole() { return _printConsole; }

protected:
    std::queue<std::tuple<std::string, LogLevel, uint64_t>> _logQueue;
    std::string _filepath = "";
    uint64_t _logNum = 0;
    bool _isLogging = false;
    bool _printConsole = false;


    static Logger* GetInstance();
    static const std::string GetLogLevelName(const LogLevel level);
    bool CreateLogFile();
    bool WriteLog(const std::tuple<std::string, LogLevel, uint64_t>& msg);
    virtual void AddMessage(std::tuple<std::string, LogLevel, uint64_t> msg)=0;
};

#endif // LOGGER_H