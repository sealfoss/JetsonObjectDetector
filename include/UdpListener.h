#ifndef UDPLISTENER_H
#define UDPLISTENER_H

#include <string>
#include <mutex>
#include <thread>
#include "UdpReceiver.h"

#define UDPLSTR_DEFAULT_NAME "UDP-Listener"
#define UDPLSTR_DEFAULT_GROUP "234.56.78.9"
#define UDPLSTR_DEFAULT_PORT 1337

class UdpListener
{
public:
    UdpListener(
        std::string name=UDPLSTR_DEFAULT_NAME,
        std::string groupIp=UDPLSTR_DEFAULT_GROUP,
        unsigned int port=UDPLSTR_DEFAULT_PORT,
        unsigned int buffLen=UDPRX_DEFAULT_BUFFLEN,
        unsigned int timeoutSeconds=UDPRX_DEFAULT_TIMEOUTS,
        UdpReceiver* receiver=nullptr
    );

    ~UdpListener();

    static bool IsMulticastAddress(std::string address);

    std::string GetName() { return _name; }

    void StartListening();

    void StopListening();

    bool IsListening();

    void Callback(std::string message);

private:
    const std::string _name;

    const std::string _groupIp;

    const unsigned int _port;

    const unsigned int _buffLen;

    const unsigned int _timeoutSecs;

    UdpReceiver* _receiver;

    bool _listening;

    char* _buff = nullptr;

    std::thread _thread;

    std::mutex _dataMutex;

    static void*  UdpListen(void*  data);

    inline void Lock();

    inline void Unlock();

    void SetListening(bool listening);
};

#endif // UDPLISTENER_H
