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
#include <memory>
#include <NvInfer.h>
#include <filesystem>
#include <nppi.h>
#include <nppi_data_exchange_and_initialization.h>

#define YOLO11_IN_IDX 0
#define YOLO11_OUT_IDX 1
#define YOLO11_H_IDX 2
#define YOLO11_W_IDX 3
#define YOLO11_ATTRIB_IDX 1
#define YOLO11_DETECT_IDX 2
#define YOLO11_CLASSES_OFFSET 4
#define YOLO11_NUM_CHANNELS 3

class AsyncLogger;
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
    AsyncLogger* _logger = nullptr;
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

    cv::Mat _cpuImg;
    cv::cuda::GpuMat _gpuImg;
    cv::cuda::GpuMat _gpuPlanarImg;
    cv::cuda::GpuMat _gpuModelInput;
    cv::cuda::GpuMat _gpuModelOutput;
    uchar* _gpuPlanarImgBuff = nullptr;
    uchar* _gpuModelInBuff = nullptr;
    uchar* _gpuModelOutBuff = nullptr;
    float* _redPlane = nullptr;
    float* _greenPlane = nullptr;
    float* _bluePlane = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;
    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::Dims _shapeIn;
    nvinfer1::Dims _shapeOut;
    cudaStream_t _stream;
    uint64_t _engineLen = 0;
    uint64_t _imgW = 0;
    uint64_t _imgH = 0;
    uint64_t _imgC = 0;
    uint64_t _modelInH = 0;
    uint64_t _modelInW = 0;
    uint64_t _planeSize = 0;
    uint64_t _outputLayerSize = 0;
    uint64_t _inputRgbBuffLen = 0;
    uint64_t _outputLayerByteLen = 0;
    uint64_t _inputLayerByteLen = 0;
    uint64_t _attribsSize = 0;
    uint64_t _detectsSize = 0;
    uint64_t _numClasses = 0;
    std::string _inputTensorName;
    std::string _outputTensorName;
    Npp8u* _intPlanes[YOLO11_NUM_CHANNELS];
    Npp32f* _floatPlanes[YOLO11_NUM_CHANNELS];
    int _intSteps[YOLO11_NUM_CHANNELS];
    NppiSize _roi;
    
    std::string Onnx2Engine(std::filesystem::path& onnxFile);
    void DetectObjects();
    void RunInference(GstBuffer* buffer);
    bool LoadModel(std::string modelPath);
    bool AllocateInputBuffer();
    bool LoadImageInput();
    inline void UpdateFrameDims();
    inline void CreateGpuImg(uchar* img);
    inline void CreateCpuImg();
    inline void WriteCpuImg();
    inline bool CheckImageDims(cv::cuda::GpuMat& img);
    inline bool RunTestInference();
    inline bool AllocateCudaBuffer(uchar** buffer, std::size_t buffLen);
    inline bool DeallocateCudaBuffer(uchar** buffer);
};

#endif // OBJECTDETECTOR_H