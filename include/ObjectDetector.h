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

#define OD_DEFAULT_MIN_CONFIDENCE 0.6f
#define OD_DEFAULT_IOU_THRESHOLD 0.45f
#define OD_DEFAULT_MAX_DET 1000
#define OD_DEFAULT_NUM_IMG_CHANS 4
#define OD_DEFAULT_NUM_MODEL_CHANS 3
#define OD_DEFAULT_IN_IDX 0
#define OD_DEFAULT_OUT_IDX 1
#define OD_DEFAULT_H_IDX 2
#define OD_DEFAULT_W_IDX 3
#define OD_DEFAULT_DET_INFO_LEN_IDX 1
#define OD_DEFAULT_DET_PROPOSALS_LEN_IDX 2
#define OD_DEFAULT_CLASSES_OFFSET 4
#define OD_DEFAULT_NUM_CHANNELS 3
#define OD_INPUT_MAG 255.0
#define OD_TEST_FILE "../models/Toothbrush.jpg"
#define OD_WRITE_DEBUG_IMGS true
#define OD_NORM_CONST 255.0f


class AsyncLogger;
class NvCudaMapper;
class BufferConsumer;
class VideoStreamer;
struct VideoFrameInfo;

struct Detection
{
    int classId = 0;
    float confidence = 0;
    cv::Rect2f bbox;
};


class ObjectDetector
{
public:
    ObjectDetector(
        float minConfidence=OD_DEFAULT_MIN_CONFIDENCE,
        float iouThreshold=OD_DEFAULT_IOU_THRESHOLD,
        int maxDetections=OD_DEFAULT_MAX_DET,
        int numImageChannels=OD_DEFAULT_NUM_IMG_CHANS,
        int numModelInputChannels=OD_DEFAULT_NUM_MODEL_CHANS,
        int inTensoridx=OD_DEFAULT_IN_IDX,
        int outTensoridx=OD_DEFAULT_OUT_IDX,
        int heightIdx=OD_DEFAULT_H_IDX,
        int widthIdx=OD_DEFAULT_W_IDX,
        int detectionInfoLenIdx=OD_DEFAULT_DET_INFO_LEN_IDX,
        int proposalsLenIdx=OD_DEFAULT_DET_PROPOSALS_LEN_IDX,
        int classesOffset=OD_DEFAULT_CLASSES_OFFSET,
        std::string testFilepath=OD_TEST_FILE,
        bool writeCpuDebugImgs=OD_WRITE_DEBUG_IMGS
    );
    ~ObjectDetector();
    float GetMinConfidence();
    bool SetMinConfidence(float minConf);
    float GetIouThreshold();
    bool SetIouThreshold(float iouThresh);
    void OpenVideoStream(std::string pipleineDescription);
    void CloseVideoStream();
    bool StartDetecting(std::string modelPath);
    bool StopDetecting();
    bool IsDetecting();
    void Notify();
    std::vector<Detection> GetLatestDetections();
    
    BufferConsumer* GetConsumer() { return _consumer; }

private:
    const int _maxDets;
    const int _numImgChans;
    const int _numModelChans;
    const int _inIdx;
    const int _outIdx;
    const int _hIdx;
    const int _wIdx;
    const int _detectionInfoLenIdx;
    const int _proposalsLenIdx;
    const int _classOffset;
    const std::string _testFilepath;
    float _minConf = 0;
    float _iouThresh = 0;
    AsyncLogger* _logger = nullptr;
    VideoStreamer* _streamer = nullptr;
    BufferConsumer* _consumer = nullptr;
    NvCudaMapper* _processor = nullptr;
    std::shared_mutex _mutex;
    std::mutex _condMutex;
    std::shared_mutex _detectionsMutex;
    std::condition_variable _cv;
    std::thread _thread;
    bool _writeCpuDebugImgs = false;
    bool _cpuImgUpdated = false;
    bool _detecting = false;
    int _frameWidth = 0;
    int _frameHeight = 0;
    int _frameChannels = 0;
    int _frameByteLen = 0;
    int _frameType = 0;
    cv::Mat _cpuImg;
    cv::Mat _cpuModelOutput;
    cv::cuda::GpuMat _gpuImg;
    cv::cuda::GpuMat _gpuPlanarImg;
    cv::cuda::GpuMat _gpuModelIn;
    cv::cuda::GpuMat _gpuModelOut;
    cv::cuda::GpuMat _gpuModelOutT;
    cv::Size2i _modelInSz;
    uchar* _gpuPlanarImgBuff = nullptr;
    uchar* _gpuModelInBuff = nullptr;
    uchar* _gpuModelOutBuff = nullptr;
    uchar* _gpuModelOutTBuff = nullptr;
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
    uint64_t _detectionInfoLen = 0;
    uint64_t _predictionsLen = 0;
    uint64_t _numClasses = 0;
    std::string _inputTensorName;
    std::string _outputTensorName;

    //Npp8u* _intImgPlanes[3];
    //Npp8u* _intModelPlanes[4];
    //Npp32f* _floatPlanes[3];
    //Npp32f _consts[3] = {255.0, 255.0, 255.0};
    //int _intSteps[3];
    Npp8u** _intImgPlanes = nullptr;
    Npp8u** _intModelPlanes = nullptr;
    Npp32f** _floatPlanes = nullptr;
    Npp32f* _normConsts = nullptr;
    NppiSize _nppRoi;
    std::vector<std::string> _inputTensorNames;
    std::vector<std::string> _outputTensorNames;
    std::vector<uchar**> _buffs;

    
    std::vector<int> _idxs;
    std::vector<int> _ids;
    std::vector<cv::Rect> _bboxes;
    std::vector<float> _scores;
    std::vector<Detection> _detections;
    
    void GetTensorNames(const nvinfer1::ICudaEngine& engine);
    std::string Onnx2Engine(std::filesystem::path& onnxFile);
    void DetectObjects();
    void RunInference(GstBuffer* buffer);
    bool LoadModel(std::string modelPath);
    bool TestInference();
    bool RunInfernce();
    inline bool LoadImageInput();
    inline bool ProcessModelOutput();
    inline void UpdateFrameDims();
    inline void CreateCpuImg();
    inline bool CheckImageDims(cv::cuda::GpuMat& img);
    inline bool AllocateCudaBuffer(uchar** buffer, std::size_t buffLen);
    inline bool DeallocateCudaBuffer(uchar** buffer);
    inline void WriteDebugImages();
};

#endif // OBJECTDETECTOR_H