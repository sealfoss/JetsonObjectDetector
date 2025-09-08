#ifndef OUTSTREAMER_H
#define OUTSTREAMER_H

#include "VideoStreamer.h"
#include "Logger.h"
#include <mutex>
#include <thread>
#include <gstreamer-1.0/gst/app/app.h>

#define OS_DEFAULT_PIP "appsrc name=source ! \
video/x-raw,format=RGBA,width=1280,height=704 ! nvvidconv ! \
video/x-raw(memory:NVMM),format=NV12 ! \
nvv4l2h265enc iframeinterval=1 insert-sps-pps=true maxperf-enable=true ! \
rtph265pay ! udpsink name=sink host=192.168.68.70 port=1337 sync=false"

#define OS_DEFAULT_WIDTH 1280
#define OS_DEFAULT_HEIGHT 704
#define OS_DEFAULT_FPS 60
#define OS_DEFAULT_NUM_BUFFS 5
#define OS_DEFAULT_FMT VS_RGB_FMT

class OutStreamer : public VideoStreamer
{
public:
    OutStreamer(
        int width=OS_DEFAULT_WIDTH, int height=OS_DEFAULT_HEIGHT,
        int fps=OS_DEFAULT_FPS, int numBuffs=OS_DEFAULT_NUM_BUFFS, 
        std::string fmt=OS_DEFAULT_FMT,
        std::string pipelineDescription=OS_DEFAULT_PIP, bool autoStart=false
    ) : VideoStreamer(pipelineDescription, nullptr, autoStart)
    , _w(width), _h(height), _fps(fps), _numFrameBuffs(numBuffs), _fmt(fmt)
    {
        int i;
        _chans = fmt == VS_RGBA_FMT ? 4 : VS_RGB_FMT ? 3 : 1;
        _byteLen = _w * _h * _chans;
        _imgFrames = new cv::Mat[_numFrameBuffs];

        for(i = 0; i < _numFrameBuffs; i++)
            _imgFrames[i] = cv::Mat(_h, _w, CV_MAKETYPE(CV_8U, _chans));
    }

    ~OutStreamer() 
    {
        if(IsStreaming())
            Stop();


        if(_pool)
        {
            gst_buffer_pool_set_active(_pool, FALSE);
            gst_object_unref(_pool);
            _pool = nullptr;
        }

        if(_imgFrames)
            delete[] _imgFrames;
    }

    static void OnNeedData(GstElement *src, guint unused, gpointer gData)
    {
        OutStreamer* os = (OutStreamer*) gData;
        os->SetPushFlag(true);
        (void) src;
        (void) unused;
    }

    static void OnEnoughData(GstElement *src, guint unused, gpointer gData)
    {
        OutStreamer* os = (OutStreamer*) gData;
        os->SetPushFlag(false);
        (void) src;
        (void) unused;
    }

    void ManageStream() override
    {
        try
        {
            LogDebug("Starting video stream...");
            gst_element_set_state(_pipeline, GST_STATE_PLAYING);
            _buffThread = std::thread(&OutStreamer::PushBuffers, this);
            g_main_loop_run(_loop);
            LogDebug("Video stream ended.");
            SetStreamingFlag(false);
            gst_element_set_state(_pipeline, GST_STATE_NULL);
            if(_buffThread.joinable())
                _buffThread.join();
        }
        catch(const std::exception& e)
        {
            LogErr(
                "Out stream mamangement failed. Error: " 
                + std::string(e.what())
            );
        }
    }

    static void PushBuffers(OutStreamer* streamer)
    {
        GstBufferPool* pool = streamer->_pool;
        GstElement* src = streamer->_src;
        GstBuffer* buff = nullptr;
        GstMapInfo map;
        GstFlowReturn ret;
        std::string err = "";
        guint count;
        cv::Mat img;
        int sleep = 1000 / streamer->_fps;
        int byteLen = streamer->_byteLen;
        int fps = streamer->_fps;
        bool success, streaming;

        try
        {
            streaming = streamer->IsStreaming();
            while(streaming)
            {
                success = streamer->ShouldPush();

                if(!success)
                    LogTrace("Stream not yet ready for a buffer.");

                if(success)
                {
                    count = streamer->GetFrameCount();
                    img = streamer->GetNextVideoFrame();
                    success = !img.empty();
                    if(!success)
                        LogDebug("Empty image received, ignoring.");
                }

                if(success)
                {
                    ret = gst_buffer_pool_acquire_buffer(pool, &buff, NULL);
                    success = ret == GST_FLOW_OK && GST_IS_BUFFER(buff);
                    if(!success)
                        LogErr("Could not acquire buffer from pool.");
                }

                if(success)
                {
                    gst_buffer_map(buff, &map, GST_MAP_WRITE);
                    memcpy(map.data, img.ptr(), byteLen);
                    GST_BUFFER_PTS(buff) = 
                        gst_util_uint64_scale(count, GST_SECOND, fps);
                    GST_BUFFER_DURATION(buff) = 
                        gst_util_uint64_scale(1, GST_SECOND, fps);
                    ret = gst_app_src_push_buffer(GST_APP_SRC(src), buff);
                    gst_buffer_unmap(buff, &map);
                    
                    if(ret != GST_FLOW_OK)
                    {
                        success = false;
                        err = "Failed to push buffer to out stream: ";
                        err += std::string(gst_flow_get_name(ret));
                        LogErr(err);
                    }
                    else
                    {
                        LogTrace("Pushed buffer.");
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(sleep)
                        );
                    }
                }
            }
        }
        catch(const std::exception& e)
        {
            err = "Thread error: " + std::string(e.what());
            LogErr(err);
        }

        LogDebug("Buffer push thread exiting...");
    }

    void WriteNextBuffer(cv::Mat& img)
    {
        int idx = GetFrameCount() % _numFrameBuffs;
        img.copyTo(_imgFrames[idx]);
    }

    guint GetFrameCount()
    {
        guint frameCount;
        _buffMutex.lock();
        frameCount = _frameCount;
        _buffMutex.unlock();
        return frameCount;
    }

    guint IncrementFrameCount()
    {
        guint prev;
        _buffMutex.lock();
        prev = _frameCount;
        _frameCount++;
        _buffMutex.unlock();
        return prev;
    }

protected:
    int _w = 0;
    int _h = 0;
    int _fps = 0;
    int _chans = 0;
    int _byteLen = 0;
    int _numFrameBuffs = 0;
    std::string _fmt = "";
    GstBufferPool* _pool = nullptr;
    GstElement* _src = nullptr;
    cv::Mat* _imgFrames = nullptr;
    guint _frameCount = 0;
    bool _push = false;
    std::thread _buffThread;
    std::mutex _buffMutex;
    std::mutex _pushMutex;

    inline cv::Mat& GetNextVideoFrame()
    {
        int idx = IncrementFrameCount() % _numFrameBuffs;
        return _imgFrames[idx];
    }

    inline void SetPushFlag(bool push)
    {
        _pushMutex.lock();
        _push = push;
        _pushMutex.unlock();
    }

    inline bool ShouldPush()
    {
        bool push;
        _pushMutex.lock();
        push = _push;
        _pushMutex.unlock();
        return push;
    }

    bool BuildPipeline() override 
    { 
        bool success = false;
        GError* err = nullptr;
        GstBus* bus = nullptr;
        gpointer gData = nullptr;
        GstCaps *caps = nullptr;
        GstStructure* config = nullptr;

        try
        {
            _pipeline = gst_parse_launch(_description.c_str(), &err);

            if(_pipeline != nullptr)
            {
                _src = gst_bin_get_by_name(GST_BIN(_pipeline), VS_SRC_NAME);
                _loop = g_main_loop_new(NULL, FALSE);
                bus = gst_element_get_bus(_pipeline);
                
                if(_src && _loop && bus)
                {
                    gData = (gpointer) this;
                    caps = gst_caps_new_simple(
                        "video/x-raw",
                        "format", G_TYPE_STRING, _fmt.c_str(),
                        "width", G_TYPE_INT, _w,
                        "height", G_TYPE_INT, _h,
                        "framerate", GST_TYPE_FRACTION, _fps, 1,
                        NULL
                    );
                    
                    if(caps)
                    {
                        g_object_set(
                            _src, "caps", caps, "is-live", TRUE, 
                            "emit-signals", TRUE, "format", GST_FORMAT_TIME, 
                            NULL
                        );
                        _pool = gst_buffer_pool_new();
                        config = gst_buffer_pool_get_config(_pool);
                        gst_buffer_pool_config_set_params(
                            config, caps, _w * _h * _chans, 2, VS_MAX_BUFFS);
                        gst_buffer_pool_set_config(_pool, config);
                        gst_buffer_pool_set_active(_pool, TRUE);
                        gst_caps_unref(caps);
                    }
                    else
                    {
                        LogErr("Failed to create appsrc caps.");
                    }

                    g_signal_connect(
                        _src, "need-data", G_CALLBACK(OnNeedData), gData
                    );
                    g_signal_connect(
                        _src, "enough-data", G_CALLBACK(OnEnoughData), gData
                    );
                    gst_bus_add_watch(bus, BusCall, _loop);
                    gst_object_unref(bus);
                    success = true;
                }
            }

            if(success)
                LogDebug("Successfully built video out stream pipeline.");
            else
                LogErr("Failed to build video stream pipeline.");
        }
        catch(const std::exception& e)
        {
            LogErr("Pipline build failed, error: " + std::string(e.what()));
        }

        return success;
    }   
};

#endif // OUTSTREAMER_H