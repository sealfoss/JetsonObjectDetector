#include "ObjectDetector.h"
#include "BufferConsumer.h"
#include "NvBufProcessor.h"
#include "VideoStreamer.h"
#include "AsyncLogger.h"
#include "FormatInput.cuh"
#include "NvOnnxParser.h"
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <sstream>

#define OBJDET_TEST_FILE "/home/reed/repos/JetsonObjectDetector/models/Lenna.png"
#define OBJDET_ONNX_EXT ".onnx"
#define OBJDET_ENG_EXT ".engine"
#define OBJDET_ENG_MEM 4ULL * 1024 * 1024 * 1024

using namespace std;
using namespace cv;
using namespace nvinfer1;
using namespace nvonnxparser;
namespace fs = std::filesystem;

ObjectDetector::ObjectDetector() 
: _logger(new AsyncLogger()), _consumer(new BufferConsumer(this)), _processor(new NvBufProcessor())
{
    gst_init(nullptr, nullptr);
}

ObjectDetector::~ObjectDetector()
{
    StopDetecting();
    CloseVideoStream();
    delete _consumer;
    delete _processor;
    delete _logger;
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
    try
    {
        if(LoadModel(modelPath))
        {
            _mutex.lock();
            _detecting = true;
            _mutex.unlock();
            _thread = thread(&ObjectDetector::DetectObjects, this);
            _streamer->Start();
            success = true;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
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
    if(!_cpuImg.empty())
        _cpuImg = Mat();
    if(!_gpuImg.empty())
        _gpuImg = cuda::GpuMat();
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
    _gpuImg = cuda::GpuMat(_frameHeight, _frameWidth, _frameType, img);
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
    _gpuImg.download(_cpuImg);
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
    _cpuImg.copyTo(img);
    _cpuImgUpdated = false;
    _mutex.unlock();
    return img;
}

string ObjectDetector::Onnx2Engine(fs::path& onnxFile)
{
    ofstream outFile;
    IBuilder* builder = nullptr;
    IParser* parser = nullptr;
    INetworkDefinition* network = nullptr;
    IBuilderConfig* config = nullptr;
    IHostMemory* serialized = nullptr;
    auto logLevel = static_cast<int32_t>(ILogger::Severity::kINFO);
    string onnxPath = onnxFile.string();
    string modelName = onnxFile.stem().string();
    string modelDir = onnxFile.parent_path().string();
    string engineFilename = modelName + ".engine";
    string enginePath = fs::path(modelDir) / fs::path(engineFilename);
    string msg = "";
    string modelNum = "01";
    int iter = 1;

    try
    {
        builder = createInferBuilder(*_logger);
        if(builder)
            network = builder->createNetworkV2(0); // 0 for explicit batch
        if(network)
            parser = createParser(*network, *_logger);
        if(parser)
        {
            msg = "Parsing Tensor RT engine from onnx file at path: ";
            LogDebug(msg + onnxPath);
            if(parser->parseFromFile(onnxPath.c_str(), logLevel))
                config = builder->createBuilderConfig();
            else
            {
                for(int32_t i = 0; i < parser->getNbErrors(); i++)
                {
                    msg = string(parser->getError(i)->desc());
                    LogErr("Parser error: " + msg);
                }
            }
        }
        if(config)
        {
            config->setMemoryPoolLimit(
                MemoryPoolType::kWORKSPACE, OBJDET_ENG_MEM
            );
            msg = "Attempting to build serialized Tensor RT engine...";
            LogDebug(msg);
            serialized = builder->buildSerializedNetwork(*network, *config);
        }
        if(serialized)
        {
            while(fs::exists(enginePath))
            {
                iter++;
                modelNum = iter < 10 ? "0" + to_string(iter) : to_string(iter);
                engineFilename = modelName + "_" + modelNum + ".engine";
                enginePath = fs::path(modelDir) / fs::path(engineFilename);
            }

            msg = "Tensor RT engine data complete. Writing to file at path: ";
            LogDebug(msg + enginePath);
            outFile = ofstream(enginePath, ios::binary);
            outFile.write(
                static_cast<const char*>(serialized->data()), 
                serialized->size()
            );
        }
    }
    catch(const std::exception& e)
    {
        msg = "Error thrown while generating Tensor RT engine:\n";
        LogErr(msg + string(e.what()));
    }

    if(serialized)
        delete serialized;
    if(config)
        delete config;
    if(network)
        delete network;
    if(parser)
        delete parser;
    if(builder)
        delete builder;
    
    enginePath = fs::exists(enginePath) ? enginePath : "";
    return enginePath;
}

bool ObjectDetector::LoadModel(string modelPath)
{
    nvinfer1::ICudaEngine* engine = nullptr;
    size_t size;
    vector<char> buffer;
    ifstream file;
    fs::path modelFile(modelPath);
    fs::path ext = modelFile.extension();
    string enginePath = "";
    string msg;
    bool success = false;

    if(ext.string() == OBJDET_ONNX_EXT)
        enginePath = Onnx2Engine(modelFile);
    else if(ext.string() == OBJDET_ENG_EXT)
        enginePath = modelPath;

    if(enginePath != "")
        file = ifstream(enginePath, std::ios::binary);
    
    if (!file.is_open())
        LogErr("Failed to open engine file: " + enginePath);
    else
    {
        file.seekg(0, ios::end);
        size = file.tellg();
        file.seekg(0, ios::beg);
        buffer = vector<char>(size);
        file.read(buffer.data(), size);
        file.close();
        _runtime = createInferRuntime(*_logger);
        if (!_runtime)
            LogErr("Failed to create TensorRT runtime");
        else
        {
            engine = _runtime->deserializeCudaEngine(buffer.data(), size);
            if(engine)
            {
                msg = "Read and deserialized Tensor RT engine from file:\n";
                LogDebug(msg + enginePath);
            }
            else
                LogErr("Failed to deserialize TensorRT engine");
        }
    }

    if(engine != nullptr)
    {
        _context = engine->createExecutionContext();
        if(_context != nullptr)
            LogDebug("Obtained context from loaded engine file.");
        else
            LogErr("Failed get context from engine.");
    }

    if(_context)
    {
        cudaStreamCreate(&_stream);
        _engine = engine;
        _inputTensorName = string(_engine->getIOTensorName(YOLO11_IN_IDX));
        _outputTensorName = string(_engine->getIOTensorName(YOLO11_OUT_IDX));
        _shapeIn = _engine->getTensorShape(_inputTensorName.c_str());
        _shapeOut = _engine->getTensorShape(_outputTensorName.c_str());
        _modelInH = _shapeIn.d[YOLO11_H_IDX];
        _modelInW = _shapeIn.d[YOLO11_W_IDX];
        _attribsSize = _shapeOut.d[YOLO11_ATTRIB_IDX];
        _detectsSize = _shapeOut.d[YOLO11_DETECT_IDX];
        _numClasses = _attribsSize - YOLO11_CLASSES_OFFSET;
        _planeSize = _modelInH * _modelInW;
        _inputLayerSize = _modelInW * _modelInH * YOLO11_NUM_CHANNELS;
        _outputLayerSize = _attribsSize * _detectsSize;
        _inputLayerByteLen = _inputLayerSize * sizeof(float);
        _outputLayerByteLen = _outputLayerSize * sizeof(float);

        if(AllocateCudaBuffer(&_gpuModelInBuffer, _inputLayerByteLen))
        {
            _gpuModelInput = cuda::GpuMat(
                _modelInH, _modelInW, CV_32FC3, _gpuModelInBuffer
            );
            _redPlane = (float*)_gpuModelInBuffer;
            _greenPlane = _redPlane + _planeSize;
            _bluePlane = _greenPlane + _planeSize;

            if(AllocateCudaBuffer(&_gpuModelOutBuffer, _outputLayerByteLen))
            {
                _gpuModelOutput = cuda::GpuMat(
                    1, _outputLayerSize, CV_32FC1, _gpuModelOutBuffer
                );
            }

            success = RunTestInference();
        }

        if(success)
            msg = "Model loaded from Tensor RT .engine file successfully.";
        else
            msg = "Model loaded from .engine file but inference test failed.";
        LogDebug(msg);
    }

    return success;
}

bool ObjectDetector::RunTestInference()
{
    cuda::GpuMat gpuInput(_modelInH, _modelInW, CV_32FC3);
    Mat testImg = cv::imread(OBJDET_TEST_FILE);
    Mat testInput, testOutput;
    Rect roi;
    stringstream stream;

    if(testImg.cols >= (int)_modelInW && testImg.rows >= (int)_modelInH)
    {
        roi = Rect(testImg.cols/2, testImg.rows/2, _modelInW, _modelInH);
        testInput = testImg(roi);
    }
    else
    {
        cv::resize(
            testImg, testInput, Size(_modelInH, _modelInW), 0, 0, INTER_CUBIC
        );
    }

    cvtColor(testInput, testInput, COLOR_BGR2RGB);
    gpuInput.upload(testInput);
    _gpuModelOutput.upload(Mat::zeros(1, _outputLayerSize, CV_32FC1));
    ConvertRgbaToPlanarRgbAndNormalize(
        gpuInput.ptr(), _redPlane, _greenPlane, 
        _bluePlane, _modelInW, _modelInH, _stream
    );
    _context->setInputTensorAddress(_inputTensorName.c_str(), _gpuModelInBuffer);
    _context->setOutputTensorAddress( _outputTensorName.c_str(), _gpuModelOutBuffer);
    cudaStreamSynchronize(_stream);
    _context->enqueueV3(_stream);
    cudaStreamSynchronize(_stream);
    std::vector<float> outputData(_outputLayerSize);
    cudaMemcpy(outputData.data(), _gpuModelOutput.ptr(), _outputLayerSize * sizeof(float), cudaMemcpyDeviceToHost);

    stream << "M = " << endl;
    for(float f : outputData)
        if(f > 0)
            stream << to_string(f) << ", ";
    stream << "\nEnd M\n";
    LogDebug(stream.str());
    exit(1);
    return true;
}

bool ObjectDetector::CheckImageDims(cuda::GpuMat& img)
{
    bool widthGood = img.cols == _shapeIn.d[2];
    bool heightGood = widthGood && img.rows == _shapeIn.d[1];
    bool channelsGood = heightGood && img.channels() == 4;
    return channelsGood;
}

bool ObjectDetector::LoadImageInput(bool checkDims)
{
    if(checkDims)
    {
        LogDebug("Checking image dimensions...");
    }

    ConvertRgbaToPlanarRgbAndNormalize(
        _gpuImg.ptr(), _redPlane, _greenPlane, _bluePlane, _modelInW, _modelInH
    );
   return false;
}

bool ObjectDetector::AllocateCudaBuffer(uchar** buffer, size_t buffLen)
{
    bool success = cudaMalloc(buffer, buffLen) == cudaSuccess;
    if(!success)
    {
        *buffer = nullptr;
        LogErr("CUDA allocation failed, size: " + to_string(buffLen));
    }
    return success;
}

bool ObjectDetector::DeallocateCudaBuffer(uchar** buffer)
{
    bool success = cudaFree(buffer) == cudaSuccess;
    if(success)
        *buffer = nullptr;
    else
        LogErr("Failed to deallocation buffer.");
    return success;
}