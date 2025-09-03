#ifndef NVCUDAMAPPER_H
#define NVCUDAMAPPER_H

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <gstreamer-1.0/gst/gst.h>

class ObjectDetector;
class NvBufSurface;

class NvCudaMapper
{
public:
    NvCudaMapper();
    ~NvCudaMapper();
    uchar* MapNvmmToCuda(GstBuffer *buffer);
    inline int GetWidth();
    inline int GetHeight();
    inline int GetPitch();
    inline int GetByteLen();
    void Unmap();

private:
    NvBufSurface* _surf = nullptr;
    uchar* _cudaPtr = nullptr;
    GstBuffer* _buff = nullptr;
    GstMapInfo _map;
    int _width = 0;
    int _height = 0;
    int _pitch = 0;
    int _byteLen = 0;
};

#endif // NVCUDAMAPPER_H