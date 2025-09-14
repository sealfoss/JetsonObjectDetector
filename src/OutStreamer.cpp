#include "OutStreamer.h"
#include "Logger.h"
#include <gstreamer-1.0/gst/app/app.h>

using namespace std;
using namespace cv;

inline int Fmt2Chans(string fmt)
{
    return fmt == VS_RGBA_FMT ? 4 : VS_RGB_FMT ? 3 : 1;
}

OutStreamer::OutStreamer(
    int width, int height, int fps, string fmt, string pipeline, bool autoStart
)   : VideoStreamer(pipeline, nullptr, false)
    , _w(width), _h(height), _fps(fps), _chans(Fmt2Chans(fmt))
    , _byteLen(_w * _h * _chans), _fmt(fmt)
{
    _buffA = cv::Mat(_h, _w, CV_MAKETYPE(CV_8U, _chans));
    _buffB = cv::Mat(_h, _w, CV_MAKETYPE(CV_8U, _chans));
    _current = &_buffA;

    if(autoStart)
        Start();
}

OutStreamer::~OutStreamer()
{
    if(IsStreaming())
        Stop();

    if(_pool)
    {
        gst_buffer_pool_set_active(_pool, FALSE);
        gst_object_unref(_pool);
        _pool = nullptr;
    }
}

void OutStreamer::OnNeed(GstElement *src, guint unused, gpointer gData)
{
    OutStreamer* os = (OutStreamer*) gData;
    os->SetPushFlag(true);
    (void) src;
    (void) unused;
}

void OutStreamer::OnEnough(GstElement *src, guint unused, gpointer gData)
{
    OutStreamer* os = (OutStreamer*) gData;
    os->SetPushFlag(false);
    (void) src;
    (void) unused;
}

void OutStreamer::ManageStream()
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

void OutStreamer::PushBuffers(OutStreamer* streamer)
{
    GstBufferPool* pool = streamer->_pool;
    GstElement* src = streamer->_src;
    GstBuffer* buff = nullptr;
    GstMapInfo map;
    GstFlowReturn ret;
    std::string err = "";
    cv::Mat img;
    int byteLen = streamer->_byteLen;
    bool success, streaming;

    try
    {
        streaming = streamer->IsStreaming();
        while(streaming)
        {
            success = streamer->ShouldPush();

            if(!success)
                LogTrace("Stream not yet ready for a buffer.");

            success = streamer->WasBuffUpdated();

            if(!success)
                LogTrace("Image buffer has not yet been updated.");

            if(success)
            {
                ret = gst_buffer_pool_acquire_buffer(pool, &buff, NULL);
                success = ret == GST_FLOW_OK && GST_IS_BUFFER(buff);
                if(!success)
                    LogErr("Could not acquire buffer from pool.");
            }

            if(success)
            {
                img = streamer->GetNextVideoFrame();
                success = !img.empty();
                if(!success)
                    LogDebug("Empty image received, ignoring.");
            }
        
            if(success)
            {
                gst_buffer_map(buff, &map, GST_MAP_WRITE);
                memcpy(map.data, img.ptr(), byteLen);
                GST_BUFFER_PTS(buff) = gst_util_get_timestamp();
                GST_BUFFER_DTS(buff) = GST_BUFFER_PTS(buff);
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

void OutStreamer::WriteNextBuffer(cv::Mat& img)
{
    _buffMutex.lock();
    img.copyTo(*_current);
    _buffUpdated = true;
    _buffMutex.unlock();
}

Mat OutStreamer::GetNextVideoFrame()
{
    Mat frame;
    _buffMutex.lock();
    frame = *_current;
    _buffUpdated = false;
    if(_current == &_buffA)
        _current = &_buffB;
    else
        _current = &_buffA;
    _buffMutex.unlock();
    return frame;
}

void OutStreamer::SetPushFlag(bool push)
{
    _pushMutex.lock();
    _push = push;
    _pushMutex.unlock();
    LogTrace("Out stream push flag set to: " + to_string(push));
}

bool OutStreamer::ShouldPush()
{
    bool push;
    _pushMutex.lock();
    push = _push;
    _pushMutex.unlock();
    return push;
}

bool OutStreamer::WasBuffUpdated()
{
    bool updated;
    _buffMutex.lock();
    updated = _buffUpdated;
    _buffMutex.unlock();
    return updated;
}

bool OutStreamer::BuildPipeline() 
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
                    _src, "need-data", G_CALLBACK(OnNeed), gData
                );
                g_signal_connect(
                    _src, "enough-data", G_CALLBACK(OnEnough), gData
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