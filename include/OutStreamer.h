#ifndef OUTSTREAMER_H
#define OUTSTREAMER_H

#include "VideoStreamer.h"
#include <mutex>
#include <thread>


#define OS_DEFAULT_PIPE "appsrc name=source ! \
video/x-raw,format=RGBA,width=1280,height=704 ! nvvidconv ! \
video/x-raw(memory:NVMM),format=NV12 ! \
nvv4l2h265enc iframeinterval=2 insert-sps-pps=true maxperf-enable=true ! \
rtph265pay ! udpsink name=sink host=192.168.68.81 port=1337 sync=false"

#define OS_DEFAULT_WIDTH 1280
#define OS_DEFAULT_HEIGHT 704
#define OS_DEFAULT_FPS 30
#define OS_DEFAULT_FMT VS_RGBA_FMT
#define OS_DEFAULT_AUTOSTART false

class OutStreamer : public VideoStreamer
{
public:
    OutStreamer(
        int width=OS_DEFAULT_WIDTH, 
        int height=OS_DEFAULT_HEIGHT,
        int fps=OS_DEFAULT_FPS, 
        std::string fmt=OS_DEFAULT_FMT,
        std::string pipeline=OS_DEFAULT_PIPE, 
        bool autoStart=OS_DEFAULT_AUTOSTART
    );

    ~OutStreamer();

    static void OnNeed(GstElement *src, unsigned int unused, void* gData);

    static void OnEnough(GstElement *src, unsigned int unused, void* gData);

    void ManageStream() override;

    static void PushBuffers(OutStreamer* streamer);

    void WriteNextBuffer(cv::Mat& img);

protected:
    const int _w;
    const int _h;
    const int _fps;
    const int _chans;
    const int _byteLen;
    const std::string _fmt = 0;
    bool _buffUpdated;
    bool _push = false;
    cv::Mat _buffA;
    cv::Mat _buffB;
    cv::Mat* _current = nullptr;
    GstBufferPool* _pool = nullptr;
    GstElement* _src = nullptr;
    std::thread _buffThread;
    std::mutex _buffMutex;
    std::mutex _pushMutex;


    inline cv::Mat GetNextVideoFrame();

    inline void SetPushFlag(bool push);

    inline bool ShouldPush();

    inline bool WasBuffUpdated();

    bool BuildPipeline() override;
};

#endif // OUTSTREAMER_H