#ifndef FORMATINPUT_CUH
#define FORMATINPUT_CUH

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void ConvertRgbaToPlanarRgbAndNormalize(
    const unsigned char* rgbaData, 
    float* redPlane, 
    float* greenPlane, 
    float* bluePlane, 
    int width, 
    int height, 
    cudaStream_t stream=0
);

#endif // FORMATINPUT_CUH