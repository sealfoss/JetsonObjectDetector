#include <iostream>
#include "ObjectDetector.h"
#include "AsyncLogger.h"
#include <opencv2/opencv.hpp>
#include <string>

/* 
gst-launch-1.0 udpsrc port=5600 ! 
application/x-rtp,media=video,clock-rate=90000,encoding-name=H265 ! 
rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! 
video/x-raw,width=1280,height=720 ! queue ! autovideosink
*/

#define DRONE_PIPELEINE "udpsrc port=5600 ! \
application/x-rtp,media=video,clock-rate=90000,encoding-name=H265 ! \
rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! \
video/x-raw,format=BGRx ! queue ! appsink"

#define DEFAULT_PIPELINE "videotestsrc do-timestamp=true ! video/x-raw,format=RGBA ! nvvidconv ! video/x-raw(memory:NVMM) ! appsink name=sink"

using namespace std;
using namespace Logging;


int main(int argc, char** argv) 
{
    int num = 0;
    string filename;
    string pipeline = argc > 1 ? argv[1] : DEFAULT_PIPELINE;
    AsyncLogger* log = new AsyncLogger();
    ObjectDetector* detector = new ObjectDetector();
    detector->OpenVideoStream(pipeline);
    detector->StartDetecting("");
    

    while(detector->IsDetecting())
    {
        if(detector->WasCpuImageUpdated())
        {
            cv::Mat img = detector->GetCpuImage();
            if(!img.empty())
            {
                filename = "ProcessedImage_" + to_string(num++) + ".png";
                cv::imwrite(filename, img);
            }
            else
                LogDebug("Empty image.");
        }
    }

    delete detector;
    delete log;
    return 0;
}