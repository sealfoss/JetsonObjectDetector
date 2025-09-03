#include "NvCudaMapper.h"
#include "Logger.h"
#include <string>
#include <gst/gst.h>
#include <nvbufsurftransform.h>
#include <nvbufsurface.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>


NvCudaMapper::NvCudaMapper(){ }

NvCudaMapper::~NvCudaMapper()
{
    Unmap();
}

uchar* NvCudaMapper::MapNvmmToCuda(GstBuffer *buffer)
{
    std::string err;
    
    NvBufSurfaceParams params;
    CUresult status;
    CUeglFrame eglFrame;
    EGLImageKHR eglImg = nullptr;
    CUgraphicsResource pResource = nullptr;

    if(GST_IS_BUFFER(buffer) && gst_buffer_map(buffer, &_map, GST_MAP_READ)) 
    {
        _buff = buffer;
        _surf = (NvBufSurface*)_map.data;
        params = _surf->surfaceList[0];
        _width = params.width;
        _height = params.height;
        _pitch = params.pitch;
        _byteLen = params.dataSize;
    }

    if(_buff != nullptr && _surf->memType == NVBUF_MEM_SURFACE_ARRAY) 
    {
        if(0 != NvBufSurfaceMapEglImage(_surf, -1)) 
        {
            LogErr("Failed to map surface to EGLImage.");
        }
        else 
        {
            eglImg = _surf->surfaceList[0].mappedAddr.eglImage;
        }

        if (eglImg == nullptr) 
        {
            LogErr("No EGLImage mapped from surface.");
        }
        else 
        {
            cudaFree(0);
            status = cuGraphicsEGLRegisterImage(
                &pResource, eglImg, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE
            );
            
            if (status != CUDA_SUCCESS) 
            {
                LogErr("cuGraphicsEGLRegisterImage failed.");
            }
            else 
            {
                status = cuGraphicsResourceGetMappedEglFrame(
                    &eglFrame, pResource, 0, 0
                );

                if(status != CUDA_SUCCESS) 
                {
                    LogErr("cuGraphicsResourceGetMappedEglFrame failed.");
                }
                else 
                {
                    status = cuCtxSynchronize();
                    
                    if(status != CUDA_SUCCESS) 
                    {
                        LogErr("cuCtxSynchronize failed.");
                    }
                    else if(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) 
                    {
                        if(eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR)
                            _cudaPtr = (u_char*)eglFrame.frame.pPitch[0];
                        else
                            LogErr("Image is not RGBA format!");
                    }
                    else 
                    {
                        LogErr("Incorrect egl frame type.");
                    }
                }
            }
        }
    }

    return _cudaPtr;
}

int NvCudaMapper::GetByteLen()
{
    return _byteLen;
}

int NvCudaMapper::GetPitch()
{
    return _pitch;
}

int NvCudaMapper::GetHeight()
{
    return _height;
}

int NvCudaMapper::GetWidth()
{
    return _width;
}

void NvCudaMapper::Unmap()
{
    if(_surf != nullptr)
    {
        NvBufSurfaceUnMapEglImage(_surf, 0);
        _surf = nullptr;
    }
    
    if(_buff != nullptr)
    {
        gst_buffer_unmap(_buff, &_map);
        _buff = nullptr;
    }

    _width = _height = _pitch = _byteLen = 0;
}
