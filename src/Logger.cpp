#include "Logger.h"
#include "TimeStamp.h"
#include <fstream>
#include <iostream>

#define MSG_ERR "[ERROR]"
#define MSG_WARN "[WARNING]"
#define MSG_INFO "[INFO]"
#define MSG_DEBUG "[DEBUG]"
#define MSG_VERBOSE "[VERBOSE]"

using namespace std;
static Logger* s_Instance = nullptr;
static LogLevel s_MinLevel = LOG_VERBOSE;

static string s_LogLevelNames[] = { 
    MSG_VERBOSE, MSG_INFO, MSG_DEBUG, MSG_WARN, MSG_ERR 
};

void LogErr(const string& message)
{
    s_Instance->Log(message, LOG_ERROR);
}

void LogWarn(const string& message)
{
    s_Instance->Log(message, LOG_WARNING);
}

void LogDebug(const string& message)
{
    s_Instance->Log(message, LOG_DEBUG);
}

void LogInfo(const string& message)
{
    s_Instance->Log(message, LOG_INFO);
}

void LogTrace(const string& message)
{
    s_Instance->Log(message, LOG_VERBOSE);
}


Logger::Logger(string directoryPath, LogLevel minLevel, bool printToConsole)
: _printConsole(printToConsole)
{
    uint64_t utcNs;

    if(s_Instance != nullptr)
    {
        throw runtime_error("Logger instance already exists");
    }
    else
    {
        utcNs = TimeStamp::GetUtcTimeNs();
        directoryPath = directoryPath == "" ? "." : directoryPath;
        _filepath = directoryPath + "/log_" + to_string(utcNs) + ".txt";
        if(CreateLogFile())
        {
            s_Instance = this;
            s_MinLevel = minLevel;
        }
        else
            throw runtime_error("Failed to create log file at " + _filepath);
    }
}

Logger::~Logger()
{
    if(s_Instance == this)
        s_Instance = nullptr;
}

void Logger::Log(const string& message, const LogLevel level)
{
    uint64_t utcNs;
    tuple<string, LogLevel, uint64_t> msg;

    if(level >= s_MinLevel)
    {
        utcNs = TimeStamp::GetUtcTimeNs();
        msg = make_tuple(message, level, utcNs);
        s_Instance->AddMessage(msg);
    }
}

Logger* Logger::GetInstance()
{
    return s_Instance;
}

const string Logger::GetLogLevelName(const LogLevel level)
{
    return s_LogLevelNames[level];
}

bool Logger::CreateLogFile()
{
    bool success = false;
    ofstream logFile(_filepath);
    if (logFile)
    {
        logFile.close();
        success = true;
    }
    return success;
}

bool Logger::WriteLog(const tuple<string, LogLevel, uint64_t>& msg)
{
    bool success = false;
    
    uint64_t logNum;
    string levelName;
    ofstream logFile(_filepath, ios::app);

    if(logFile)
    {
        auto [text, level, utcNs] = msg;
        logNum = _logNum++;
        levelName = GetLogLevelName(level);

        logFile << logNum << ":\t" << levelName << "(" << utcNs 
            << ")>\t" << text << endl;
        logFile.close();

        if(IsPrintingToConsole())
        {
            if(level == LOG_ERROR)
            {
                cerr << logNum << ":\t" << levelName << "(" << utcNs 
                    << ")>\t" << text << endl;
            }
            else
            {
                cout << logNum << ":\t" << levelName << "(" << utcNs 
                    << ")>\t" << text << endl;
            }
        }

        success = true;
    }
    
    return success;
}

