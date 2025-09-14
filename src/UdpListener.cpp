#include "UdpListener.h"
#include "UdpReceiver.h"
#include "Logger.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cstring>
#include <unistd.h>


using namespace std;

UdpListener::UdpListener(string name, string groupIp, unsigned int port,
    unsigned int buffLen, unsigned int timeoutSeconds,
    UdpReceiver* receiver) : _name(name), _groupIp(groupIp), _port(port),
    _buffLen(buffLen), _timeoutSecs(timeoutSeconds), _receiver(receiver)
{
    _buff = new char[_buffLen];
    StartListening();
}

UdpListener::~UdpListener()
{
    StopListening();
    delete _buff;
    _buff = nullptr;
}

void UdpListener::StartListening()
{
    _thread = thread(UdpListen, (void*) this);
}

void UdpListener::StopListening()
{
    SetListening(false);

    if(_thread.joinable())
    {
        _thread.join();
    }
}

bool UdpListener::IsListening()
{
    bool listening = false;
    Lock();
    listening = _listening;
    Unlock();
    return listening;
}

void* UdpListener::UdpListen(void* data)
{
    UdpListener* listener = (UdpListener*)data;
    string message;
    string groupIp = listener->_groupIp;
    unsigned int port = listener->_port;
    unsigned int buffLen = listener->_buffLen;
    unsigned int timeoutSecs = listener->_timeoutSecs;
    char* buff = listener->_buff;
    struct timeval timeout;
    struct sockaddr_in localAddr;
    struct ip_mreq group;
    int result, option, end, sockfd;
    bool multicast = IsMulticastAddress(groupIp);
    bool success = true;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    success = sockfd >= 0;

    if(success)
    {
        timeout.tv_sec = timeoutSecs;
        timeout.tv_usec = 0;
        result = setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, 
                            &timeout, sizeof(timeout));
        success = result >= 0;
    }

    if(success)
    {
        memset(&localAddr, 0, sizeof(localAddr));
        localAddr.sin_family = AF_INET;
        localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
        localAddr.sin_port = htons(port);
        result = bind(sockfd, (struct sockaddr*)&localAddr, sizeof(localAddr));
        success = result >= 0;
    }

    if(success && multicast)
    {
        group.imr_multiaddr.s_addr = inet_addr(groupIp.c_str());
        group.imr_interface.s_addr = htonl(INADDR_ANY);
        result = setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                (char*)&group, sizeof(group)); 
        success = result >= 0;
    }

    if(success)
    {
        listener->SetListening(true);

        while(listener->IsListening())
        {
            end = recv(sockfd, buff, buffLen, 0);

            if(end > 0)
            {
                buff[end] = '\0';
                message = string(buff);
                listener->Callback(message);
            }    
        }

        if(multicast)
        {
            result = setsockopt(sockfd, IPPROTO_IP, IP_DROP_MEMBERSHIP, 
                (char*)&group, sizeof(group));

            close(sockfd);
            success = result >= 0;
        }
    }
    return nullptr;
}

void UdpListener::Callback(string message)
{
    _receiver->OnMessageReceived(_name, message);
}

void UdpListener::Lock()
{
    _dataMutex.lock();
}

void UdpListener::Unlock()
{
    _dataMutex.unlock();
}

void UdpListener::SetListening(bool listening)
{
    string msg;
    Lock();
    _listening = listening;
    Unlock();

    if(listening)
        msg = "Receiving UDP messages from listener " + _name;
    else
        msg = "Listener " + _name + " shutting down.";

    LogDebug(msg);
}



bool UdpListener::IsMulticastAddress(string address)
{
    bool isMulticast = false;
    int i = 0;
    int idZero = -1;
    string sub; 

    if(address.size() > 0)
    {
        while(i < address.size() && address[i] != UDPRX_IP_DELIM)
            i++;

        if(i < address.size())
        {
            sub = address.substr(0, i);
            idZero = stoi(sub);

            if(idZero >= UDPRX_MULTICAST_MIN && idZero <= UDPRX_MULTICAST_MAX)
                isMulticast = true;
        }
    }

    return isMulticast;
}
