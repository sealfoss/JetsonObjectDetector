#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "ObjectDetector.h"
#include "Logger.h"
#include "nlohmann/json.hpp"


#define DEFAULT_MODELPATH "/home/reed/repos/JetsonObjectDetector/models/yolo11m/yolo11m.engine"

#define DRONE_PIPELEINE "udpsrc port=5600 ! \
application/x-rtp,media=video,clock-rate=90000,encoding-name=H265 ! \
rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! \
video/x-raw,format=BGRx ! queue ! appsink"

#define UDP_PIPELINE "udpsrc port=1337 ! \
application/x-rtp,media=video,clock-rate=90000,encoding-name=H265 ! \
rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! \
video/x-raw(memory:NVMM),width=1280,height=704,format=RGBA ! \
appsink name=sink"

#define TEST_PIPELINE "videotestsrc do-timestamp=true ! \
video/x-raw,format=RGBA ! nvvidconv ! video/x-raw(memory:NVMM) ! \
appsink name=sink"

#define DEFAULT_PIPELINE UDP_PIPELINE
#define DEFAULT_CONFIG_PATH "../DetectionConfig.json"


using namespace std;
using namespace nlohmann;
namespace fs = filesystem;

struct DetectionConfig
{
    string pipeline;
    string modelPath;
    string error;

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE (
        DetectionConfig, pipeline, modelPath
    )
};

bool ReadConfigFile(const std::string filepath, DetectionConfig& config)
{
    ifstream file;
    string fileText;
    stringstream sstream;
    json configJson;
    bool success = false;

    try
    {
        file = ifstream(filepath);
        if(file.is_open())
        {
            sstream << file.rdbuf();
            file.close();
            fileText = sstream.str();
            configJson = json::parse(fileText);
            config.from_json(configJson, config);
            success = 
                config.modelPath.size() > 0 && config.pipeline.size() > 0;
        }
    }
    catch(const std::exception& e)
    {
        sstream.str("");
        sstream << "Failed to read detection config from file at path \"" 
        << filepath << "\". Error:\n" << string(e.what());
        config.error = sstream.str();
    }

    return success;
}


int main(int argc, char** argv) 
{
    vector<Detection> detections;
    DetectionConfig config;
    stringstream stream;
    string configPath = argc > 1 ? argv[1] : DEFAULT_CONFIG_PATH;
    ObjectDetector* detector = new ObjectDetector();

    if(!ReadConfigFile(configPath, config))
    {
        LogErr(config.error);
        config.modelPath = DEFAULT_MODELPATH;
        config.pipeline = DEFAULT_PIPELINE;
    }

    if(detector->StartDetecting(config.modelPath))
        detector->OpenVideoStream(config.pipeline);

    while(detector->IsDetecting())
    {
        detections = detector->GetLatestDetections();
        if(detections.size() > 0)
        {
            for(Detection det : detections)
            {
                stream.str("");
                stream << "Detected Object, class: " << det.classId
                    << ", confidence: " << det.confidence << ", x: " 
                    << det.bbox.x << ", y: " << det.bbox.y << ", width: " 
                    << det.bbox.width << ", height: " << det.bbox.height;
                LogDebug(stream.str());
            }
        }
    }

    LogDebug("Exiting, goodbye.");
    delete detector;
    return 0;
}