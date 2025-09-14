#include "DetectionSender.h"
#include "Logger.h"
#include <cstring>

using namespace std;


DetectionSender::DetectionSender(
    string ip, ushort port, size_t msgLen, size_t maxNumMsgs)
: _ip(ip.c_str()), _port(port), _msgLen(msgLen)
, _msgByteLen(msgLen * sizeof(ushort))
, _maxNumMsgs(maxNumMsgs), _buffsByteLen(msgLen * maxNumMsgs * sizeof(ushort))
{
    _buffs = new ushort*[maxNumMsgs];
    memset(_buffs, 0, _buffsByteLen);
    SetFlag(&_initialized, InitSocket());
    if(IsInitialized())
        _thread = thread(&DetectionSender::SendDetections, this);
}

DetectionSender::~DetectionSender()
{
    SetFlag(&_sending, false);
    if(_thread.joinable())
        _thread.join();
    DeleteBuffers();
}

bool DetectionSender::InitSocket()
{
    bool success = false;
    _sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    
    try
    {
        if(_sock < 0)
        {
            LogErr("Unable to create socket: " + to_string(_sock));
        }
        else
        {
            _addrSz = sizeof(_dest);
            memset(&_dest, 0, _addrSz);
            _dest.sin_family = AF_INET;
            inet_pton(AF_INET, _ip, &_dest.sin_addr);
            _addr = (sockaddr*)&_dest;
            success = true;
        }
    }
    catch(const exception& e)
    {
        LogErr("Unable to create socket. Error: " + string(e.what()));
    }

    return success;
}


void DetectionSender::SendDetections()
{
    int result, i;
    
    SetFlag(&_sending, true);

    while(IsSending())
    {
        Wait();

        while(IsNewMsgAvailable())
        {
            Lock();

            for(i = 0; i < _numAdded; i++)
            {
                result = sendto(
                    _sock, _buffs[i], _msgByteLen, 0, _addr, _addrSz
                );

                if(result < 0)
                {
                    LogErr(
                        "Failed to send message, result: " + to_string(result)
                    );
                    _sending = false;
                    break;
                }
            }

            memset(_buffs, 0, _buffsByteLen);
            _numAdded = 0;
            Unlock();
        }
    }
}

void DetectionSender::AddDetections(const ushort* detections, const ushort num)
{
    size_t numMsgs = num <= _maxNumMsgs ? num : _maxNumMsgs;
    size_t byteLen = numMsgs * sizeof(ushort);
    Lock();
    memcpy(_buffs, detections, byteLen);
    _numAdded = numMsgs;
    Unlock();
}

void DetectionSender::DeleteBuffers()
{
    if(_buffs)
    {
        delete[] _buffs;
        _buffs = nullptr;
    }
}

void DetectionSender::Lock(bool shared)
{
    if(shared)
        _dataMutex.lock_shared();
    else
        _dataMutex.lock();
}

void DetectionSender::Unlock(bool shared)
{
    if(shared)
        _dataMutex.unlock_shared();
    else
        _dataMutex.unlock();
}

void DetectionSender::Wait()
{
    unique_lock<mutex> lock(_cvMutex);
    _cv.wait(lock, [this] { return IsNewMsgAvailable() || !IsSending(); });
}

void DetectionSender::Notify()
{
    _cv.notify_all();
}

bool DetectionSender::SetFlag(bool* addr, bool val)
{
    bool prev;
    Lock();
    prev = *addr;
    *addr = val;
    Unlock();
    return prev;
}

bool DetectionSender::GetFlag(bool* addr)
{
    bool val;
    Lock(true);
    val = *addr;
    Unlock(true);
    return val;
}

bool DetectionSender::IsInitialized()
{
    return GetFlag(&_initialized);
}

bool DetectionSender::IsSending()
{
    return GetFlag(&_sending);
}

bool DetectionSender::IsNewMsgAvailable()
{
    bool newMsg;
    Lock(true);
    newMsg = _numAdded > 0;
    Unlock(true);
    return newMsg;
}