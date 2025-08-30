#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <string>
#include <shared_mutex>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <gstreamer-1.0/gst/gst.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class NvBufProcessor;
class BufferConsumer;
class VideoStreamer;
struct VideoFrameInfo;

class ObjectDetector
{
public:
    ObjectDetector();
    ~ObjectDetector();
    void OpenVideoStream(std::string pipleineDescription);
    void CloseVideoStream();
    bool StartDetecting(std::string modelPath);
    bool StopDetecting();
    bool IsDetecting();
    bool WasCpuImageUpdated();
    cv::Mat GetCpuImage();
    void Notify();
    
    BufferConsumer* GetConsumer() { return _consumer; }

private:
    VideoStreamer* _streamer = nullptr;
    BufferConsumer* _consumer = nullptr;
    NvBufProcessor* _processor = nullptr;
    std::shared_mutex _mutex;
    std::mutex _condMutex;
    std::condition_variable _cv;
    std::thread _thread;
    bool _cpuImgUpdated = false;
    bool _detecting = false;
    int _frameWidth = 0;
    int _frameHeight = 0;
    int _frameChannels = 0;
    int _frameByteLen = 0;
    int _frameType = 0;

    cv::Mat _imgCpu;
    cv::cuda::GpuMat _imgGpu;

    void DetectObjects();
    void RunInference(GstBuffer* buffer);
    bool LoadModel(std::string modelPath);
    inline void UpdateFrameDims();
    inline void CreateGpuImg(uchar* img);
    inline void CreateCpuImg();
    inline void WriteCpuImg();
};

#endif // OBJECTDETECTOR_H