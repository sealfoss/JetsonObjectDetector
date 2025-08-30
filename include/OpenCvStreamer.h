#ifndef OPENCVSTREAMER_h
#define OPENCVSTREAMER_h

#include <opencv2/opencv.hpp>
#include <string>
#include <shared_mutex>
#include <thread>


class OpenCvStreamer {
public:
    OpenCvStreamer(const std::string& pipeline);
    ~OpenCvStreamer();

    bool Start();
    bool Stop();
    bool IsFrameAvailable();
    void OnFrameAvailable();
    bool IsCapturing();
    bool SetCapturing(bool capturing);
    static void CaptureVideo(OpenCvStreamer* streamer);

private:
    cv::Mat _frame;
    cv::VideoCapture _cap;
    std::shared_mutex _streamMutex;
    std::shared_mutex _frameMutex;
    std::thread _thread;
    std::string _pipeline = "";
    bool _capturing = false;
    bool _available = false;

    inline cv::Mat GetFrame()
    {
        cv::Mat frame;
        if (TryLockFrame())
        {
            frame = _frame.clone();
            UnlockFrame();
        }
        return frame;
    }

    inline void SetFrame(const cv::Mat frame)
    {
        _frame = frame;
        _available = true;
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

#endif // OPENCVSTREAMER_h