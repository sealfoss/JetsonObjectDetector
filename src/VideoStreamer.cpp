#include "VideoStreamer.h"
#include "BufferConsumer.h"
#include "Logger.h"
#include <cudaEGL.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda.hpp>
#include <gstreamer-1.0/gst/gst.h>
#include <sstream>

using namespace std;    
using namespace cv;


VideoStreamer::VideoStreamer(
    string pipelineDescription, BufferConsumer* consumer, bool autoStart
) : _description(pipelineDescription), _consumer(consumer)
{
    if(autoStart)
        Start();
}

VideoStreamer::~VideoStreamer()
{
    if(IsStreaming())
        Stop();
}

void VideoStreamer::ManageStream()
{
    try
    {
        LogDebug("Starting video stream...");
        gst_element_set_state(_pipeline, GST_STATE_PLAYING);
        g_main_loop_run(_loop);
        LogDebug("Video stream ended.");
        SetStreamingFlag(false);
        gst_element_set_state(_pipeline, GST_STATE_NULL);
    
    }
    catch(const std::exception& e)
    {
        LogErr("Video Stream failed, Error:\n" + string(e.what()));
    }
}

bool VideoStreamer::BuildPipeline()
{
    bool success = false;
    GError* err = nullptr;
    GstBus* bus = nullptr;
    GstPad *sinkPad = nullptr;
    gpointer gData = nullptr;

    try
    {
        _pipeline = gst_parse_launch(_description.c_str(), &err);

        if(_pipeline != nullptr)
        {
            _sink = gst_bin_get_by_name(GST_BIN(_pipeline), VS_SINK_NAME);
            _loop = g_main_loop_new(NULL, FALSE);
            bus = gst_element_get_bus(_pipeline);
            
            if(_sink && _loop && bus)
            {
                gData = (gpointer) this;
                g_object_set(G_OBJECT(_sink), VS_EMIT_NAME, true, NULL);
                g_signal_connect(
                    _sink, VS_NEWSAMPLE_NAME, G_CALLBACK(OnNewSample), gData
                );
                sinkPad = gst_element_get_static_pad(_sink, VS_SINK_NAME);
                gst_pad_add_probe(
                    sinkPad, VS_CHECK_DIMS, OnProbePad, gData, NULL
                );
                gst_bus_add_watch(bus, BusCall, _loop);
                gst_object_unref(bus);
                success = true;
            }
        }
    }
    catch(const std::exception& e)
    {
        LogErr("Failed to build pipeline, error: " + string(e.what()));
    }

    return success;
}

GstPadProbeReturn VideoStreamer::OnProbePad(
    GstPad *pad, GstPadProbeInfo *info, gpointer gData
)
{
    VideoStreamer* streamer = (VideoStreamer*) gData;
    GstEvent *event = nullptr;
    GstCaps *caps = nullptr;
    GstStructure *structure = nullptr;
    const gchar *format;
    gint width, height, channels, depth, byteLen;
    event = GST_PAD_PROBE_INFO_EVENT(info);
    (void)pad; // Mark pad as unused argument.

    if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) 
    {   
        width = height = channels = depth = byteLen = 0;
        gst_event_parse_caps(event, &caps);
        structure = gst_caps_get_structure(caps, 0);
        format = gst_structure_get_string(structure, VS_FMT_KEY);
        if (format) 
        {
            if (g_strcmp0(format, VS_RGB_FMT) == 0) 
            {
                channels = 3;
                depth = 8;
            } 
            else if (g_strcmp0(format, VS_RGBA_FMT) == 0)
            {
                channels = 4;
                depth = 8;
            } 
            else if (g_strcmp0(format, VS_NV12_FMT) == 0) 
            {
                channels = 3;
                depth = 8;
            } 
            else if(g_strcmp0(format, VS_GRAY8_FMT) == 0)
            {
                channels = 1;
                depth = 8;
            } 
        }
        gst_structure_get_int(structure, VS_WIDTH_KEY, &width);
        gst_structure_get_int(structure, VS_HEIGHT_KEY, &height);
        byteLen = width * height * channels;
        streamer->SetFrameInfo(
            width, height, channels, depth, byteLen
        );
    }
    return GST_PAD_PROBE_OK;
}

void VideoStreamer::SetFrameInfo(
    int width, int height, int channels, int depth, int byteLen
)
{
    _frameMutex.lock();
    _frameInfo = { width, height, channels, depth, byteLen };
    _frameInfoUpdated = true;
    _frameMutex.unlock();
}

VideoFrameInfo VideoStreamer::GetFrameInfo()
{
    VideoFrameInfo info;
    _frameMutex.lock();
    info = _frameInfo;
    _frameInfoUpdated = false;
    _frameMutex.unlock();
    return info;
}

bool VideoStreamer::WasFrameInfoUpdated()
{
    bool updated = false;
    _frameMutex.lock_shared();
    updated = _frameInfoUpdated;
    _frameMutex.unlock_shared();
    return updated;
}

GstFlowReturn VideoStreamer::OnNewSample(GstAppSink *sink, gpointer gData)
{
    GstFlowReturn flow = GST_FLOW_OK;
    GstBuffer* buffer = nullptr;
    GstSample* sample = gst_app_sink_pull_sample(sink);
    VideoStreamer* streamer = (VideoStreamer*) gData;

    if(sample)
    {
        buffer = gst_sample_get_buffer(sample);

        if(buffer)
        {
            gst_buffer_ref(buffer);

            if(!streamer->_consumer->AddBuffer(buffer))
                gst_buffer_unref(buffer);
        }
        
        gst_sample_unref(sample);
    }

    return flow;
}

gboolean VideoStreamer::BusCall(GstBus* bus, GstMessage* msg, gpointer data)
{
    VideoStreamer* streamer = (VideoStreamer*) data;
    gchar *debugInfo = nullptr;
    GError *err = nullptr;
    GMainLoop *loop = (GMainLoop *)data;
    GstState old, current, pending;
    stringstream sstream;
    (void)bus; // Mark bus argument unused.

    switch (GST_MESSAGE_TYPE(msg)) 
    {
        case GST_MESSAGE_EOS:
        {
            LogDebug("End of stream.");
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_ERROR: 
        {    
            gst_message_parse_error(msg, &err, &debugInfo);
            sstream << "Error from element " << GST_OBJECT_NAME(msg->src) 
                << ": " << err->message << std::endl << "Debugging info: " 
                << (debugInfo ? debugInfo : "none") << std::endl;
            LogErr(sstream.str());
            g_clear_error(&err);
            g_free(debugInfo);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_STATE_CHANGED:
        {
            gst_message_parse_state_changed(msg, &old, &current, &pending);
            sstream << "Pipeline state changed from \""
                << gst_element_state_get_name(old) << "\" to \""
                << gst_element_state_get_name(current) << "\". ";
            sstream << "Pending: " << gst_element_state_get_name(pending);
            LogDebug(sstream.str());
            streamer->RecordCurrentState(current);
        }
        default:
            break;
    }

    return TRUE;
}

void VideoStreamer::RecordCurrentState(GstState state)
{
    _streamMutex.lock();
    _current = state;
    _streamMutex.unlock();
}

GstState VideoStreamer::GetCurrentState()
{
    GstState current;
    _streamMutex.lock_shared();
    current = _current;
    _streamMutex.unlock_shared();
    return current;
}

bool VideoStreamer::Start()
{
    bool success = false;
    LogDebug("Starting video stream with pipeline: " + _description);
    if(BuildPipeline())
    {
        LockStream(true);
        if (!_streaming)
        {
            _streaming = true;
            _thread = std::thread(&VideoStreamer::ManageStream, this);
            success = true;
        }
        UnlockStream(true);
    }
    return success;
}

bool VideoStreamer::Stop()
{
    bool success = false;
    GstEvent *eos_event = nullptr;
    LogDebug("Stopping video stream...");
    eos_event = gst_event_new_eos();
    gst_element_send_event(_pipeline, eos_event);
    if (_thread.joinable())
        _thread.join();
    success = true;
    return success;
}

bool VideoStreamer::IsStreaming()
{
    bool capturing;
    LockStream();
    capturing = _streaming;
    UnlockStream();
    return capturing;
}

bool VideoStreamer::SetStreamingFlag(bool capturing)
{
    bool wasCapturing;
    LockStream(true);
    wasCapturing = _streaming;
    _streaming = capturing;
    UnlockStream(true);
    return wasCapturing;
}

void VideoStreamer::OnFrameAvailable()
{
    LockFrame(true);
    _available = true;
    UnlockFrame(true);
}

bool VideoStreamer::IsFrameAvailable()
{
    bool available;
    LockFrame();
    available = _available;
    UnlockFrame();
    return available;
}