#include "OpenCvStreamer.h"

using namespace cv;
using namespace std;

OpenCvStreamer::OpenCvStreamer(const std::string& pipeline) 
    : _pipeline(pipeline)
{
    Start();
}

void OpenCvStreamer::CaptureVideo(OpenCvStreamer* streamer)
{
    Mat frame;
    while (streamer->IsCapturing())
    {
        if (streamer->TryLockFrame(true) && streamer->_cap.read(frame))
        {
            streamer->SetFrame(frame);
            streamer->UnlockFrame(true);
        }
    }
}

bool OpenCvStreamer::Start()
{
    bool success = false;
    LockStream(true);
    if (!_capturing)
    {
        _cap = VideoCapture(_pipeline.c_str(), cv::CAP_GSTREAMER);
        _capturing = _cap.isOpened();
        if (_capturing)
        {
            _thread = std::thread(&OpenCvStreamer::CaptureVideo, this);
            success = true;
        }
    }
    UnlockStream(true);
    return success;
}

bool OpenCvStreamer::Stop()
{
    bool success = false;
    if (SetCapturing(false))
    {
        _cap.release();
        if (_thread.joinable())
            _thread.join();
        success = true;
    }
    return success;
}

bool OpenCvStreamer::IsCapturing()
{
    bool capturing;
    LockStream();
    capturing = _capturing;
    UnlockStream();
    return capturing;
}

bool OpenCvStreamer::SetCapturing(bool capturing)
{
    bool wasCapturing;
    LockStream(true);
    wasCapturing = _capturing;
    _capturing = capturing;
    UnlockStream(true);
    return wasCapturing;
}

void OpenCvStreamer::OnFrameAvailable()
{
    LockFrame(true);
    _available = true;
    UnlockFrame(true);
}

bool OpenCvStreamer::IsFrameAvailable()
{
    bool available;
    LockFrame();
    available = _available;
    UnlockFrame();
    return available;
}
