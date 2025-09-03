#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <vector>
#include "ObjectDetector.h"
#include "Logger.h"


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


using namespace std;

int main(int argc, char** argv) 
{
    gst_init(nullptr, nullptr);
    ObjectDetector* detector = nullptr;
    string pipeline = argc > 1 ? argv[1] : DEFAULT_PIPELINE;
    string modelPath = argc > 2 ? argv[2] : DEFAULT_MODELPATH;
    stringstream stream;
    vector<Detection> detections;

    detector = new ObjectDetector();
    detector->OpenVideoStream(pipeline);
    detector->StartDetecting(modelPath);

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