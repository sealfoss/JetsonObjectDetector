#ifndef UDPRECEIVER_H
#define UDPRECEIVER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

#define UDPRX_DEFAULT_NAME "UDP-RX"
#define UDPRX_DEFAULT_BUFFLEN 1024
#define UDPRX_DEFAULT_TIMEOUTS 5
#define UDPRX_IP_DELIM ':'
#define UDPRX_MULTICAST_MIN 224
#define UDPRX_MULTICAST_MAX 239

typedef void (*MsgCallback)(std::string);

class UdpListener;

class UdpReceiver
{
public:
    UdpReceiver(
        MsgCallback callback=nullptr,
        std::string name=UDPRX_DEFAULT_NAME, 
        unsigned int buffLen=UDPRX_DEFAULT_BUFFLEN,
        unsigned int timeoutSecs=UDPRX_DEFAULT_TIMEOUTS
    );

    ~UdpReceiver();

    void ListenOn(unsigned int port, std::string groupIp, std::string name);

    void StopListening(std::string name);

    bool IsReceiving();

    std::vector<std::string> GetReceived();

    void OnMessageReceived(std::string name, std::string message);


private:
    std::unordered_map<std::string, UdpListener*> _listeners;

    std::vector<std::string> _listenerNames;

    std::vector<std::string> _received;

    std::mutex _dataMutex;

    bool _receiving;

    const std::string _name;

    const unsigned int _buffLen;

    const unsigned int _timeoutSecs;

    const MsgCallback _callback;

    void Lock();

    void Unlock();

    void SetReceiving(bool receiving);
};

#endif // UDPRECEIVER_H
