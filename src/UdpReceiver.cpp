#include "UdpReceiver.h"
#include "UdpListener.h"
#include "Logger.h"
#include <algorithm>

using namespace std;

UdpReceiver::UdpReceiver(MsgCallback callback, string name,
    unsigned int buffLen, unsigned int timeoutSecs)
    :   _name(name), _buffLen(buffLen), _timeoutSecs(timeoutSecs),
        _callback(callback)
    { 
        SetReceiving(true);
    }

UdpReceiver::~UdpReceiver()
{
    UdpListener* listener = nullptr;
    int i;
    SetReceiving(false);
    Lock();

    for(i = 0; i < _listenerNames.size(); i++)
    {
        listener = _listeners[_listenerNames[i]];
        delete listener;
    }

    _listenerNames.clear();
    _listeners.clear();

    Unlock();
}

void UdpReceiver::ListenOn(unsigned int port, string groupIp, string name)
{
    string listenerName = _name + "_" + groupIp + ":" + to_string(port);
    UdpListener* listener = new UdpListener(listenerName, groupIp, port,
        _buffLen, _timeoutSecs, this);
    _listeners[listenerName] = listener;
    _listenerNames.push_back(listenerName);
}

void UdpReceiver::StopListening(string name)
{
    vector<string>::iterator itr = 
        find(_listenerNames.begin(), _listenerNames.end(), name);
    UdpListener* listener = itr != _listenerNames.end() ? 
        _listeners[name] : nullptr;

    if(listener)
    {
        _listenerNames.erase(itr);
        _listeners.erase(name);
        delete listener;
    }
}

void UdpReceiver::OnMessageReceived(string name, string message)
{
    LogInfo("Received message, " + name + ":\n" + message);

    if(_callback)
        _callback(message);
}

void UdpReceiver::Lock()
{
    _dataMutex.lock();
}

void UdpReceiver::Unlock()
{
    _dataMutex.unlock();
}

void UdpReceiver::SetReceiving(bool receiving)
{
    Lock();
    _receiving = receiving;
    Unlock();
}

bool UdpReceiver::IsReceiving()
{
    bool receiving = false;
    Lock();
    receiving = _receiving;
    Unlock();
    return receiving;
}

vector<string> UdpReceiver::GetReceived()
{
    vector<string> received;
    Lock();

    for(string message : _received)
        received.push_back(message);
    _received.clear();
    Unlock();
    return received;
}
