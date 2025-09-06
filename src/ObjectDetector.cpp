#include "ObjectDetector.h"
#include "BufferConsumer.h"
#include "NvCudaMapper.h"
#include "VideoStreamer.h"
#include "AsyncLogger.h"
#include "FormatInput.cuh"
#include "NvOnnxParser.h"
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <sstream>
#include <algorithm>


#define OD_ONNX_EXT ".onnx"
#define OD_ENG_EXT ".engine"
#define OD_ENG_MEM 4ULL * 1024 * 1024 * 1024
#define OD_VEC_CAP 1000


using namespace std;
using namespace cv;
using namespace nvinfer1;
using namespace nvonnxparser;
namespace fs = std::filesystem;


ObjectDetector::ObjectDetector(
    float minConfidence, float iouThreshold, int maxDetections, 
    int numImgChans, int numModelChans, int inTensorIdx, int outTensorIdx, 
    int heightIdx, int widthIdx, int detectionInfoLenIdx, int proposalsLenIdx, 
    int classesOffset, string testFilepath, bool writeCpuDebugImgs
) 
: _maxDets(maxDetections), _numImgChans(numImgChans)
, _numModelChans(numModelChans), _inIdx(inTensorIdx), _outIdx(outTensorIdx)
, _hIdx(heightIdx), _wIdx(widthIdx), _detectionInfoLenIdx(detectionInfoLenIdx)
, _proposalsLenIdx(proposalsLenIdx), _classOffset(classesOffset)
, _testFilepath((fs::current_path() / fs::path(testFilepath)).string())
, _logger(new AsyncLogger()), _consumer(new BufferConsumer(this))
, _processor(new NvCudaMapper()), _writeDetectionImg(writeCpuDebugImgs)
{
    gst_init(nullptr, nullptr);
    _minConf = minConfidence;
    _iouThresh = iouThreshold;
}

ObjectDetector::~ObjectDetector()
{
    for(uchar** buffer : _buffs)
        cudaFree(buffer);
    StopDetecting();
    CloseVideoStream();
    delete _consumer;
    delete _processor;
    delete _logger;
    gst_deinit();

    if(_intImgPlanes)
        delete[] _intImgPlanes;
    if(_intModelPlanes)
        delete[] _intModelPlanes;
    if(_floatPlanes)
        delete[] _floatPlanes;
    if(_normConsts)
        delete _normConsts;
}

void ObjectDetector::Notify()
{
    _cv.notify_all();
}

void ObjectDetector::OpenVideoStream(string pipelineDescription)
{
    CloseVideoStream();
    LogDebug("Opening video stream with pipeline: " + pipelineDescription);
    _streamer = new VideoStreamer(pipelineDescription, _consumer, true);
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
    LogDebug("Loading detection model from filepath: " + modelPath);

    try
    {
        if(LoadModel(modelPath))
        {
            _mutex.lock();
            _detecting = true;
            _mutex.unlock();
            _thread = thread(&ObjectDetector::DetectObjects, this);
            if(_streamer)
                _streamer->Start();
            success = true;
        }
    }
    catch(const std::exception& e)
    {
        LogErr("Failed to start object detector, error: " + string(e.what()));
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

void ObjectDetector::RunInference(GstBuffer* buffer)
{
    uchar* img = nullptr;

    if(GST_IS_BUFFER(buffer))
        img = _processor->MapNvmmToCuda(buffer);
    else
        LogErr("GstBuffer passed to RunInference is not valid.");

    if(img != nullptr)
    {
        _mutex.lock();
        _gpuImg = cuda::GpuMat(_frameHeight, _frameWidth, _frameType, img);
        LoadImageInput();
        _processor->Unmap();
        _consumer->UnrefLastBuffer();
        _context->enqueueV3(_stream);
        cudaStreamSynchronize(_stream);
        ProcessModelOutput();
        _mutex.unlock();
    }
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
    int i;

    try
    {
        builder = createInferBuilder(*_logger);
        if(builder)
            network = builder->createNetworkV2(0);
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
                for(i = 0; i < parser->getNbErrors(); i++)
                {
                    msg = string(parser->getError(i)->desc());
                    LogErr("Parser error: " + msg);
                }
            }
        }
        if(config)
        {
            config->setMemoryPoolLimit(
                MemoryPoolType::kWORKSPACE, OD_ENG_MEM
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

void ObjectDetector::GetTensorNames(const ICudaEngine& engine) 
{
    int i;
    string tensorName;
    nvinfer1::TensorIOMode iomode;
    stringstream stream;

    stream << "Engine has " << engine.getNbIOTensors() 
        << " input/output tensors.";
    LogDebug(stream.str());
    stream.str("");

    for (i = 0; i < engine.getNbIOTensors(); ++i) 
    {
        tensorName = string(engine.getIOTensorName(i));
        stream << "Tensor " << i << " \"" << tensorName << "\": ";
        iomode = engine.getTensorIOMode(tensorName.c_str());

        if (iomode == nvinfer1::TensorIOMode::kINPUT)
        {
            _inputTensorNames.push_back(tensorName);
            stream << "(INPUT)";
        }
        else if (iomode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            _outputTensorNames.push_back(tensorName);
            stream << "(OUTPUT)";
        }
        else 
        {
            stream << "(UNKNOWN)";
        }

        LogDebug(stream.str());
        stream.str("");
    }
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
    int i;
    bool success = false;

    if(ext.string() == OD_ONNX_EXT)
        enginePath = Onnx2Engine(modelFile);
    else if(ext.string() == OD_ENG_EXT)
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
        GetTensorNames(*_engine);
        _inputTensorName = string(_engine->getIOTensorName(_inIdx));
        _outputTensorName = string(_engine->getIOTensorName(_outIdx));
        _shapeIn = _engine->getTensorShape(_inputTensorName.c_str());
        _shapeOut = _engine->getTensorShape(_outputTensorName.c_str());
        _modelInH = _shapeIn.d[_hIdx];
        _modelInW = _shapeIn.d[_wIdx];
        _planeSize = _modelInH * _modelInW;
        _inputRgbBuffLen = _planeSize * _numImgChans;
        _inputLayerByteLen = _planeSize * _numModelChans * sizeof(float);
        _detectionInfoLen = _shapeOut.d[_detectionInfoLenIdx];
        _predictionsLen = _shapeOut.d[_proposalsLenIdx];
        _numClasses = _detectionInfoLen - _classOffset;
        _outputLayerSize = _detectionInfoLen * _predictionsLen;
        _outputLayerByteLen = _outputLayerSize * sizeof(float);
        _nppRoi = {(int)_modelInW, (int)_modelInH};
        _modelInSz = Size2i(_modelInW, _modelInH);
        success = AllocateCudaBuffer(&_gpuPlanarImgBuff, _inputRgbBuffLen);

        if(success)
        {

            _gpuPlanarImg = cuda::GpuMat(
                _modelInH, _modelInW, CV_MAKETYPE(CV_8U, _numImgChans), 
                _gpuPlanarImgBuff
            );
            
            success = AllocateCudaBuffer(&_gpuModelInBuff, _inputLayerByteLen);
        }

        if(success)
        {
            try
            {
                _intImgPlanes = new Npp8u*[_numImgChans];
                _intModelPlanes = new Npp8u*[_numModelChans];
                _floatPlanes = new Npp32f*[_numModelChans];
                _normConsts =  new Npp32f[_numModelChans];
                for(i = 0; i < _numImgChans; i++)
                    _intImgPlanes[i] = _gpuPlanarImgBuff + (_planeSize * i);
                for(i = 0; i < _numModelChans; i++)
                {
                     _intModelPlanes[i] = _intImgPlanes[i];
                     _normConsts[i] = OD_NORM_CONST;
                }
            }
            catch(const std::exception& e)
            {
                success = false;
            }
        }

        if(success)
        {
            _gpuModelIn = cuda::GpuMat(
                _modelInH, _modelInW, CV_MAKETYPE(CV_32F, 3), 
                _gpuModelInBuff
            );
            _redPlane = (float*)_gpuModelInBuff;
            _greenPlane = _redPlane + _planeSize;
            _bluePlane = _greenPlane + _planeSize;
            _floatPlanes[0] = _redPlane;
            _floatPlanes[1] = _greenPlane;
            _floatPlanes[2] = _bluePlane;
            _context->setInputTensorAddress(
                _inputTensorName.c_str(), _gpuModelInBuff
            );
            success = AllocateCudaBuffer(
                &_gpuModelOutBuff, _outputLayerByteLen
            );
        }

        if(success)
        {
            _gpuModelOut = cuda::GpuMat(
                _detectionInfoLen, _predictionsLen, CV_32FC1, _gpuModelOutBuff
            );
            
            _context->setOutputTensorAddress(
                _outputTensorName.c_str(), _gpuModelOutBuff
            );
            success = AllocateCudaBuffer(
                &_gpuModelOutTBuff, _outputLayerByteLen
            );
        }

        if(success)
        {
            _gpuModelOutT = cuda::GpuMat(
                _predictionsLen, _detectionInfoLen, CV_32FC1, _gpuModelOutTBuff
            );
            _cpuModelOutput = 
                Mat(_predictionsLen, _detectionInfoLen, CV_32FC1);
        }

        

        if(success && TestInference())
        {
            msg = "Model loaded from Tensor RT .engine file successfully.";
            LogDebug(msg);
        }
        else
        {
            if(success)
                msg = "Model loaded from file but inference test failed.";
            else
                msg = "Unable to allocate neccessary CUDA buffers.";
            LogErr(msg);
            success = false;
        }
    }

    return success;
}

bool ObjectDetector::LoadImageInput()
{
    bool success = false;
    int channels = _gpuImg.channels();
    NppStatus status = nppiCopy_8u_C4P4R(
        _gpuImg.data, _gpuImg.step, _intImgPlanes, _gpuImg.cols, _nppRoi
    ); 
    cudaStreamSynchronize(0);

    if(status == NPP_NO_ERROR)
    {
        if(_writeDetectionImg)
            WriteDetectionImg();
        status = nppiConvert_8u32f_C3R(
            _gpuPlanarImg.data, _gpuPlanarImg.cols * _numModelChans, 
            (float*)_gpuModelInBuff, _gpuModelIn.step, _nppRoi
        );
        cudaStreamSynchronize(0);

        if(status == NPP_NO_ERROR)
        {
            status = nppiDivC_32f_C3IR(
                _normConsts, (float*)_gpuModelIn.data, 
                _gpuModelIn.step, _nppRoi
            );

            cudaStreamSynchronize(0);
            
            if(status == NPP_NO_ERROR)
                success = true;
            else
                LogErr("Failed to normalize float value input.");
        }
        else
            LogErr("Failed convert split RGB channels to float values.");
    }
    else
        LogErr("Failed to split RGB channels. Num: " + to_string(channels));

    return success;
}

void ObjectDetector::WriteDetectionImg()
{
    _imgOutMutex.lock();
    _gpuImg.download(_cpuDetectionImg);
    _imgOutMutex.unlock();
}

void ObjectDetector::GetDetectionImg(Mat& outImg)
{
    _imgOutMutex.lock();
    _cpuDetectionImg.copyTo(outImg);
    _imgOutMutex.unlock();
}

bool ObjectDetector::ProcessModelOutput()
{
    stringstream stream;
    cv::Mat objClass, prediction, scores;
    int i, classId;
    float confidence, x, y, w, h, halfW, halfH;
    float *predictionPtr, *firstElement, *lastElement, *maxElement;
    bool success = false;

    cuda::transpose(_gpuModelOut, _gpuModelOutT);
    _gpuModelOutT.download(_cpuModelOutput);

    for(i = 0; i < _cpuModelOutput.rows; i++)
    {
        prediction = _cpuModelOutput.row(i);
        predictionPtr = (float*)prediction.data;
        firstElement = ((float*)prediction.data) + 4;
        lastElement = firstElement + 80;
        maxElement = std::max_element(firstElement, lastElement);
        confidence = *maxElement;
        
        if(confidence > _minConf)
        {
            classId = maxElement - firstElement;
            x = (*predictionPtr);
            y = (*(predictionPtr+1));
            w = (*(predictionPtr+2));
            h = (*(predictionPtr+3));
            halfW = w * 0.5f;
            halfH = h * 0.5f;
            _bboxes.push_back(Rect2f(x-halfW, y-halfH, x+halfW, y+halfW));
            _scores.push_back(confidence);
            _ids.push_back(classId);
        }
    }

    if(_ids.size() > 0)
    {
        dnn::NMSBoxes(_bboxes, _scores, _minConf, _iouThresh, _idxs);
        success = _idxs.size() > 0;

        if(success)
        {
            _detectionsMutex.lock();
            for(int idx : _idxs)
            {
                _detections.push_back(
                    Detection { _ids[idx], _scores[idx], _bboxes[idx] }
                );
            }
            _detectionsMutex.unlock();
        }

        _ids = vector<int>();
        _idxs = vector<int>();
        _scores = vector<float>();
        _bboxes = vector<Rect>();
    }

    return success;
}

vector<Detection> ObjectDetector::GetLatestDetections()
{
    
    _detectionsMutex.lock();
    vector<Detection> detections(_detections);
    _detections.clear();
    _detections = vector<Detection>();
    _detectionsMutex.unlock();
    return detections;
}

bool ObjectDetector::TestInference()
{
    vector<Detection> detections;
    string msg;
    cuda::GpuMat gpuInput(_modelInH, _modelInW, CV_32FC3);
    Mat testImg = cv::imread(_testFilepath);
    Mat testInput, testOutput;
    Rect roi;

    cv::resize(
        testImg, testInput, Size(_modelInW, _modelInH), 0, 0, INTER_CUBIC
    );
    cvtColor(testInput, testInput, COLOR_BGR2RGB);
    _gpuImg.upload(testInput);

    if(LoadImageInput())
    {
        _context->enqueueV3(_stream);
        cudaStreamSynchronize(_stream);

        if(ProcessModelOutput())
        {
            detections = GetLatestDetections();
            msg = "Test Inference class: " + to_string(detections[0].classId);
            LogDebug(msg);
        }
        else
        {
            msg = "Test inference failed.";
            LogErr(msg);
        }
    }
    else
        LogErr("Failed to load image into model.");

    return true;
}

bool ObjectDetector::CheckImageDims(cuda::GpuMat& img)
{
    bool widthGood = img.cols == _shapeIn.d[2];
    bool heightGood = widthGood && img.rows == _shapeIn.d[1];
    bool channelsGood = heightGood && img.channels() == 4;
    return channelsGood;
}

bool ObjectDetector::AllocateCudaBuffer(uchar** buffer, size_t buffLen)
{
    bool success = cudaMalloc(buffer, buffLen) == cudaSuccess;

    if(success)
        _buffs.push_back(buffer);
    else
    {
        *buffer = nullptr;
        LogErr("CUDA allocation failed, size: " + to_string(buffLen));
    }

    return success;
}

bool ObjectDetector::DeallocateCudaBuffer(uchar** buffer)
{
    auto toRemove = _buffs.end();
    bool success = cudaFree(buffer) == cudaSuccess;

    if(success)
    {
        for(auto itr = _buffs.begin(); itr != _buffs.end(); itr++)
            if(*itr == buffer)
                toRemove = itr;
        if(toRemove != _buffs.end())
            _buffs.erase(toRemove);
        *buffer = nullptr;
    }
    else
        LogErr("Failed to deallocation buffer.");

    return success;
}

float ObjectDetector::GetMinConfidence()
{
    float minConf = 0;
    _mutex.lock_shared();
    minConf = _minConf;
    _mutex.unlock_shared();
    return minConf;
}

bool ObjectDetector::SetMinConfidence(float minConf)
{
    bool success = _mutex.try_lock();

    if(success)
    {
        _minConf = minConf;
        _mutex.unlock();
    }
    return success;
}

float ObjectDetector::GetIouThreshold()
{
    float iouThresh = 0;
    _mutex.lock_shared();
    iouThresh = _iouThresh;
    _mutex.unlock_shared();
    return iouThresh;
}

bool ObjectDetector::SetIouThreshold(float iouThresh)
{
    bool success = _mutex.try_lock();

    if(success)
    {
        _iouThresh = iouThresh;
        _mutex.unlock();
    }

    return success;
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

