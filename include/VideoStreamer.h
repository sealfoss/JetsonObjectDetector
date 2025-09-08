#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <gstreamer-1.0/gst/gst.h>
#include <gstreamer-1.0/gst/base/gstbasetransform.h>
#include <gstreamer-1.0/gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <shared_mutex>
#include <thread>

#define VS_MAX_BUFFS 10
#define VS_WIDTH_KEY "width"
#define VS_HEIGHT_KEY "height"
#define VS_FMT_KEY "format"
#define VS_RGB_FMT "RGB"
#define VS_RGBA_FMT "RGBA"
#define VS_NV12_FMT "NV12"
#define VS_GRAY8_FMT "GRAY8"
#define VS_SRC_NAME "source"
#define VS_SINK_NAME "sink"
#define VS_EMIT_NAME "emit-signals"
#define VS_NEWSAMPLE_NAME "new-sample"
#define VS_CHECK_DIMS (GstPadProbeType)(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM)

struct VideoFrameInfo
{
    int width=0;
    int height=0;
    int channels=0;
    int depth=0;
    int byteLen=0;
};

class BufferConsumer;

class VideoStreamer {
public:
    VideoStreamer(
        std::string pipelineDescription, BufferConsumer* consumer=nullptr,
        bool autoStart=false
    );
    virtual ~VideoStreamer();

    bool Start();
    bool Stop();
    bool IsFrameAvailable();
    void OnFrameAvailable();
    bool IsStreaming();
    bool SetStreamingFlag(bool capturing);
    virtual void ManageStream();
    void RecordCurrentState(GstState state);
    GstState GetCurrentState();
    void SetFrameInfo(
        int width, int height, int channels, int depth, int byteLen
    );
    VideoFrameInfo GetFrameInfo();
    bool WasFrameInfoUpdated();
    static gboolean BusCall(GstBus* bus, GstMessage* msg, gpointer data);
    static GstFlowReturn OnNewSample(GstAppSink *sink, gpointer data);
    static GstPadProbeReturn OnProbePad(
        GstPad *pad, GstPadProbeInfo *info, gpointer gData
    );

protected:
    std::string _description = "";
    BufferConsumer* _consumer = nullptr;
    GMainLoop* _loop = nullptr;
    GstElement* _pipeline = nullptr;
    GstElement* _source = nullptr;
    GstElement* _sink = nullptr;
    GstState _current = GST_STATE_NULL;
    bool _streaming = false;
    bool _available = false;
    std::shared_mutex _streamMutex;
    std::shared_mutex _frameMutex;
    std::thread _thread;
    cv::Mat _cpuFrame;
    cv::cuda::GpuMat _gpuFrame;
    VideoFrameInfo _frameInfo;
    bool _frameInfoUpdated = false;

    virtual bool BuildPipeline();

    static GstFlowReturn TransformMem (
        GstBaseTransform* btrans, GstBuffer* inbuf
    );

    inline void NotifyAvailable()
    {
        if(TryLockFrame())
        {
            _available = true;
            UnlockFrame();
        }
    }

    inline cv::Mat GetFrame()
    {
        cv::Mat frame;
        if (TryLockFrame())
        {
            frame = _cpuFrame.clone();
            UnlockFrame();
        }
        return frame;
    }

    inline void SetFrame(const cv::cuda::GpuMat frame)
    {
        _gpuFrame = frame;
        _available = false;
    }

    inline bool TryLockFrame(bool write=false)
    {
        return write ? _frameMutex.try_lock() : _frameMutex.try_lock_shared();
    }

    inline void LockStream(bool write=false)
    {
        if (write)
            _streamMutex.lock();
        else
            _streamMutex.lock_shared();
    }

    inline void UnlockStream(bool write=false)
    {
        if (write)
            _streamMutex.unlock();
        else
            _streamMutex.unlock_shared();
    }

    inline void LockFrame(bool write=false)
    {
        if (write)
            _frameMutex.lock();
        else
            _frameMutex.lock_shared();
    }

    inline void UnlockFrame(bool write=false)
    {
        if (write)
            _frameMutex.unlock();
        else
            _frameMutex.unlock_shared();
    }
};

#endif // VIDEOSTREAMER_H