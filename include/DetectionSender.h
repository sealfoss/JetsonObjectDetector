#ifndef DETECTIONSENDER_H
#define DETECTIONSENDER_H

#include <string>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <thread>
#include <arpa/inet.h>

#define DS_DEFAULT_IP "234.56.78.9"
#define DS_DEFAULT_PORT 1337
#define DS_DEFAULT_MSG_BYTE_LEN 14
#define DS_DEFAULT_NUM_BUFFS 10


class DetectionSender
{
public:
    DetectionSender(
        std::string ip=DS_DEFAULT_IP, ushort port=DS_DEFAULT_PORT, 
        std::size_t msgLen=DS_DEFAULT_MSG_BYTE_LEN,
        std::size_t maxNumMsgs=DS_DEFAULT_MSG_BYTE_LEN
    );
    ~DetectionSender();
    inline bool IsNewMsgAvailable();
    inline bool IsInitialized();
    inline bool IsSending();
    virtual void AddDetections(const ushort* detection, const ushort num);
    void SendDetections();

private:
    const char* _ip;
    const ushort _port;
    const std::size_t _msgLen;
    const std::size_t _msgByteLen;
    const std::size_t _maxNumMsgs;
    const std::size_t _buffsByteLen;
    sockaddr_in _dest;
    sockaddr* _addr = nullptr;
    std::size_t _addrSz;
    int _sock = -1;
    int _numAdded = -1;
    bool _initialized = false;
    bool _sending = false;
    ushort** _buffs = nullptr;
    std::shared_mutex _dataMutex;
    std::mutex _cvMutex;
    std::condition_variable _cv;
    std::thread _thread;

    bool InitSocket();
    void SendDetection(const float* detection);
    inline void DeleteBuffers();
    inline void Lock(bool shared=false);
    inline void Unlock(bool shared=false);
    inline void Wait();
    inline void Notify();
    inline bool SetFlag(bool* addr, bool val);
    inline bool GetFlag(bool* addr);
};

#endif // DETECTIONSENDER_H