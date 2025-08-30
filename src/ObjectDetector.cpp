#include "ObjectDetector.h"
#include "BufferConsumer.h"
#include "NvBufProcessor.h"
#include "VideoStreamer.h"

using namespace std;
using namespace cv;

ObjectDetector::ObjectDetector() 
: _consumer(new BufferConsumer(this)), _processor(new NvBufProcessor())
{
    gst_init(nullptr, nullptr);
}

ObjectDetector::~ObjectDetector()
{
    StopDetecting();
    CloseVideoStream();
    delete _consumer;
    delete _processor;
    gst_deinit();
}

void ObjectDetector::Notify()
{
    _cv.notify_all();
}


void ObjectDetector::OpenVideoStream(string pipelineDescription)
{
    CloseVideoStream();
    _streamer = new VideoStreamer(pipelineDescription, _consumer);
}

void ObjectDetector::CloseVideoStream()
{
    if(_streamer)
    {
        delete _streamer;
        _streamer = nullptr;
    }
}

bool ObjectDetector::StartDetecting(std::string modelPath)
{
    bool success = false;
    if(LoadModel(modelPath))
    {
        _mutex.lock();
        _detecting = true;
        _mutex.unlock();
        _thread = thread(&ObjectDetector::DetectObjects, this);
        success = true;
    }
    return success;
}

bool ObjectDetector::StopDetecting()
{
    bool wasDetecting;
    _mutex.lock();
    wasDetecting = _detecting;
    _detecting = false;
    _mutex.unlock();
    _cv.notify_all();

    if (_thread.joinable())
    {
        _thread.join();
    }
    return wasDetecting;
}

bool ObjectDetector::LoadModel(std::string modelPath)
{
    bool success = true;
    return success;
}

void ObjectDetector::DetectObjects()
{
    GstBuffer* buff = nullptr;
    unique_lock<mutex> lock(_condMutex);

    while(IsDetecting())
    {
        _cv.wait(
            lock, [this] { return _consumer->HasBuffers() || !_detecting; }
        );

        while(_consumer->HasBuffers())
        {
            
            lock.unlock();
            buff = _consumer->GetLastBuffer();
            if(_streamer->WasFrameInfoUpdated())
                UpdateFrameDims();
            if(buff != nullptr)
                RunInference(buff);
            lock.lock();
        }
    }
}

void ObjectDetector::UpdateFrameDims()
{
    VideoFrameInfo info = _streamer->GetFrameInfo();
    _frameWidth = info.width;
    _frameHeight = info.height;
    _frameChannels = info.channels;
    _frameByteLen = info.byteLen;
    _frameType = CV_MAKETYPE(CV_8U,_frameChannels);
    if(!_imgCpu.empty())
        _imgCpu = Mat();
    if(!_imgGpu.empty())
        _imgGpu = cuda::GpuMat();
}

bool ObjectDetector::IsDetecting()
{
    bool detecting;
    _mutex.lock_shared();
    detecting = _detecting;
    _mutex.unlock_shared();
    return detecting;
}

void ObjectDetector::CreateGpuImg(uchar* img)
{
    _imgGpu = cuda::GpuMat(_frameHeight, _frameWidth, _frameType, img);
}

void ObjectDetector::RunInference(GstBuffer* buffer)
{
    uchar* img = nullptr;

    if(GST_IS_BUFFER(buffer))
        img = _processor->MapNvmmToCuda(buffer);

    if(img != nullptr)
    {
        _mutex.lock();
        CreateGpuImg(img);
        WriteCpuImg();
        _processor->Unmap();
        _consumer->UnrefLastBuffer();
        _mutex.unlock();
    }
}

void ObjectDetector::WriteCpuImg()
{
    _imgGpu.download(_imgCpu);
    _cpuImgUpdated = true;
}

bool ObjectDetector::WasCpuImageUpdated()
{
    bool updated = false;
    if(_mutex.try_lock_shared())
    {
        updated = _cpuImgUpdated;
        _mutex.unlock_shared();
    }
    return updated;
}

Mat ObjectDetector::GetCpuImage()
{
    cv::Mat img;
    _mutex.lock();
    _imgCpu.copyTo(img);
    _cpuImgUpdated = false;
    _mutex.unlock();
    return img;
}


