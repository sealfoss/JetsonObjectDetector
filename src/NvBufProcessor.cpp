// Uncomment the following line if using JetPack4 (R32). Let it commented if using JP5 (R35)
//#define JP4

#include "NvBufProcessor.h"
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

#ifdef JP4    
static EGLDisplay egl_display;

static inline unsigned int GetPitch(unsigned int width, const unsigned int stride) {
    g_print("width=%d, stride=%d\n", width, stride);
    int num_strides = width/stride;
    if ((width % stride) != 0)
        ++num_strides;
    return num_strides*stride;
}
#endif /* JP4 */

// This example would just draw a 200x200 green rectangle starting at (100,100) so resolution is expected to be at least 500 for height and width  
static cv::Rect roi(100,100,200,200); 
static cv::Rect roi2(50,50,100,100); // for half size planes such as U&V in NV12 format


NvBufProcessor::NvBufProcessor()
{

}

NvBufProcessor::~NvBufProcessor()
{
    Unmap();
}

uchar* NvBufProcessor::MapNvmmToCuda(GstBuffer *buffer)
{
    std::string err;
    EGLImageKHR egl_image;
    NvBufSurfaceParams params;
    CUresult status;
    CUeglFrame eglFrame;
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

    if(_buff != nullptr && _surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
        if(0 != NvBufSurfaceMapEglImage(_surf, -1)) {
            LogErr("Failed to map surface to EGLImage.");
        }
        else {
            egl_image = _surf->surfaceList[0].mappedAddr.eglImage;
        }

        if (egl_image == nullptr) {
            LogErr("No EGLImage mapped from surface.");
        }
        else {
            cudaFree(0);
            status = cuGraphicsEGLRegisterImage(
                &pResource, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE
            );
            
            if (status != CUDA_SUCCESS) {
                LogErr("cuGraphicsEGLRegisterImage failed.");
            }
            else {
                status = cuGraphicsResourceGetMappedEglFrame(
                    &eglFrame, pResource, 0, 0
                );

                if(status != CUDA_SUCCESS) {
                    LogErr("cuGraphicsResourceGetMappedEglFrame failed.");
                }
                else {
                    status = cuCtxSynchronize();
                    
                    if(status != CUDA_SUCCESS) {
                        LogErr("cuCtxSynchronize failed.");
                    }
                    else if(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
                        if(eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR)
                            _cudaPtr = (u_char*)eglFrame.frame.pPitch[0];
                        else
                            LogErr("Image is not RGBA format!");
                    }
                    else {
                        LogErr("Incorrect egl frame type.");
                    }
                }
            }
        }
    }

    return _cudaPtr;
}

void NvBufProcessor::TestProcessRgbaCudaImg(cv::cuda::GpuMat& img)
{
    img(roi).setTo(cv::Scalar(128,0,128,255));
}


int NvBufProcessor::GetByteLen()
{
    return _byteLen;
}

int NvBufProcessor::GetPitch()
{
    return _pitch;
}

int NvBufProcessor::GetHeight()
{
    return _height;
}

int NvBufProcessor::GetWidth()
{
    return _width;
}

void NvBufProcessor::Unmap()
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

static void CvCudaProcessRGBA(const unsigned int height, const unsigned int pitch, uchar * bufPtr) {
    g_print("RGBA processing using pitch %d\n", pitch);
    cv::cuda::GpuMat d_mat(height, pitch, CV_8UC4, bufPtr);

    // Process
    d_mat(roi).setTo(cv::Scalar(0,255,0,255));  // R,G,B,A

    // Check
    if (d_mat.data != bufPtr)
        g_printerr("Error RGBA buffer reallocated\n");
}


static void CvCudaProcessI420(const unsigned int height, const unsigned int pitch, uchar * YbufPtr, uchar * UbufPtr, uchar * VbufPtr) {
    g_print("I420 processing using pitch %d\n", pitch);
    // Y is full size
    cv::cuda::GpuMat d_Mat_Y(height, pitch, CV_8UC1, YbufPtr);
    // U and V are half size
    cv::cuda::GpuMat d_Mat_U(height/2, pitch/2, CV_8UC1, UbufPtr);
    cv::cuda::GpuMat d_Mat_V(height/2, pitch/2, CV_8UC1, VbufPtr);

    // Process
    d_Mat_Y(roi).setTo(255);
    d_Mat_U(roi2).setTo(0);
    d_Mat_V(roi2).setTo(0);
    
    // Check
    if (d_Mat_Y.data != YbufPtr)
        g_printerr("Error Y buffer reallocated\n");
    if (d_Mat_U.data != UbufPtr)
        g_printerr("Error U buffer reallocated\n"); 
    if (d_Mat_V.data != VbufPtr)
        g_printerr("Error V buffer reallocated\n"); 
}

static void CvCudaProcessNV12(const unsigned int height, const unsigned int pitch, uchar * YbufPtr, uchar * UVbufPtr) {
    static std::vector<cv::cuda::GpuMat> uv(2);
    g_print("NV12 using pitch %d\n", pitch);
    // Y is full size
    cv::cuda::GpuMat d_Mat_Y(height, pitch, CV_8UC1, YbufPtr);
    // U and V are half size and interleaved 
    cv::cuda::GpuMat d_Mat_UV(height/2, pitch/2, CV_8UC2, UVbufPtr);
    cv::cuda::split(d_Mat_UV, uv);
    cv::cuda::GpuMat& d_Mat_U = uv[0];
    cv::cuda::GpuMat& d_Mat_V = uv[1];

    // Process
    d_Mat_Y(roi).setTo(255);
    d_Mat_U(roi2).setTo(0);
    d_Mat_V(roi2).setTo(0);

    // reinterleave U&V
    cv::cuda::merge(uv, d_Mat_UV);
    // Check
    if (d_Mat_Y.data != YbufPtr)
        g_printerr("Error Y buffer reallocated\n");
    if (d_Mat_UV.data != UVbufPtr)
        g_printerr("Error UV buffer reallocated\n");
}


/************************************************************************/
/* Maps an EGL frame and call CUDA processing depending on color format */
/************************************************************************/
static void CvCudaProcessEGLImage(EGLImageKHR& egl_image, unsigned int pitch)
{
    //printf("EglImage at %p\n", egl_image);
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;
    std::string err;

    cudaFree(0);
    status = cuGraphicsEGLRegisterImage(&pResource,
                                        egl_image,
                                        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS)
        LogErr("cuGraphicsEGLRegisterImage failed\n");
        //g_printerr("cuGraphicsEGLRegisterImage failed\n");

    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
        LogErr("cuGraphicsResourceGetMappedEglFrame failed\n");

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
        LogErr("cuCtxSynchronize failed\n");

    //printf("Mapped frame with width=%d and height=%d\n", eglFrame.width, eglFrame.height);
    if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        /********************/
        /* EGL Pitch frame  */
        /********************/
        //g_print("EGL frame type pitch: ");
        LogDebug("EGL frame type pitch: ");
        switch(eglFrame.eglColorFormat) {
            case CU_EGL_COLOR_FORMAT_ABGR:
                CvCudaProcessRGBA(eglFrame.height, eglFrame.width, (uchar*) eglFrame.frame.pPitch[0]);
                break;

            case CU_EGL_COLOR_FORMAT_YUV420_PLANAR:
#ifdef JP4
                pitch = GetPitch(eglFrame.width, 256);
#endif /* JP4 */
                CvCudaProcessI420(eglFrame.height, pitch, (uchar*)eglFrame.frame.pPitch[0], (uchar*)eglFrame.frame.pPitch[1], (uchar*)eglFrame.frame.pPitch[2]);
                break;
        
            case CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR:
#ifdef JP4
                pitch = GetPitch(eglFrame.width, 256);
#endif /* JP4 */
                CvCudaProcessNV12(eglFrame.height, pitch, (uchar*)eglFrame.frame.pPitch[0], (uchar*)eglFrame.frame.pPitch[1]);
                break;
        
            default:
            {
                err = "Unsupported eglcolorformat " + std::to_string(eglFrame.eglColorFormat) + " for CU_EGL_FRAME_TYPE_PITCH\n";
                LogErr(err);
            }
        }
    }
    else {
        /***********************/
        /* EGL CU Array frame  */
        /***********************/
        /* EGL CU Array frame  */
        /***********************/
        LogErr("EGL frame type array. Block linear format is not public AFAIK. Currently unsupported\n");
    }

    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
        LogErr("cuCtxSynchronize failed\n");

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS)
        LogErr("cuGraphicsUnregisterResource failed\n");

}






/********************************************************************/
/* This src (output) probe callback function would be called each   */
/* time that element with name "conv" has prepared an output buffer */
/* that we can process here before it goes further in pipeline      */
/********************************************************************/
static GstPadProbeReturn
conv_src_pad_buffer_probe (GstPad* pad, GstPadProbeInfo* info, gpointer u_data)
{
    GstBuffer *buffer = (GstBuffer *) info->data;
    GstMapInfo map    = {0};
    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    
    EGLImageKHR egl_image;
    
#ifdef JP4
    int dmabuf_fd = 0;
    if (-1 == ExtractFdFromNvBuffer((void *)map.data, &dmabuf_fd))
    {
        g_printerr("ExtractFdFromNvBuffer failed\n");
    }
    egl_image = NvEGLImageFromFd(egl_display, dmabuf_fd);
    CvCudaProcessEGLImage(egl_image, 0); // pitch will be later computed from eglFrame
    NvDestroyEGLImage(egl_display, egl_image);
    
#else /* JP5 */

    NvBufSurface* surf = (NvBufSurface*)map.data;
    g_print("Got NvBufSurface with %d surface%s  mem type %d\n", surf->numFilled, surf->numFilled > 1 ? "s":"", surf->memType);
    NvBufSurfaceParams params = surf->surfaceList[0];
    g_print("Surface has width=%u, height=%u, pitch=%u, memsize=%u at %p bufferDesc=%zu\n", 
            params.width, params.height, params.pitch, params.dataSize, params.dataPtr, params.bufferDesc);
            
    if (surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
   	    /***************************************************/    
        /* NVBUF_MEM_SURFACE_ARRAY is accessed through EGL */
    	/***************************************************/            
        g_print("Using EGL CUDA mapping: ");
        //int dmabuf_fd = surf->surfaceList[0].bufferDesc;

        if(0 != NvBufSurfaceMapEglImage(surf, -1))
            g_printerr("Failed to map surface to EGLImage\n");
        egl_image = surf->surfaceList[0].mappedAddr.eglImage; 
        if (egl_image == NULL)
            g_printerr("No EGLImage mapped from surface\n");

        CvCudaProcessEGLImage(egl_image, params.pitch);
        NvBufSurfaceUnMapEglImage (surf, 0);
    }
    else {  
        g_print("Not using EGL: ");
        switch(params.colorFormat) {
            case NVBUF_COLOR_FORMAT_RGBA:
                CvCudaProcessRGBA(params.height, params.pitch/4, (uchar *)surf->surfaceList[0].dataPtr);
                break;

            case NVBUF_COLOR_FORMAT_YUV420:
                CvCudaProcessI420(params.height, params.pitch, (uchar*)surf->surfaceList[0].dataPtr, (uchar *)surf->surfaceList[0].dataPtr + params.height*params.pitch, (uchar *)surf->surfaceList[0].dataPtr + params.height*params.pitch + params.height*params.pitch/4);
                break;

            case NVBUF_COLOR_FORMAT_NV12:
                CvCudaProcessNV12(params.height, params.pitch, (uchar*)surf->surfaceList[0].dataPtr, (uchar *)surf->surfaceList[0].dataPtr + params.height*params.pitch);
                break;

            default:
                g_printerr("Unsupported color format\n");
        }
    }
     
#endif /* JP5 */
    
    gst_buffer_unmap(buffer, &map);

    return GST_PAD_PROBE_OK;
}




void print_usage(void) {
    fprintf(stderr, "Usage: test_surface \"your pipeline string with your processing element having name=conv\"\n");
    return;
}


/************************************************************************************************/
/* This example builds a gstreamer app that will launch the pipeline string passed as argument. */
/* In that pipeline, the app will try to find an element with name=conv. Then it will add a src */
/* pad probe to it that will be able to post process its output buffers from OpenCV CUDA        */
/************************************************************************************************/
/*gint
main (gint   argc,
      gchar *argv[])
{
    if (argc != 2) {
        print_usage();
        exit(-2);
    }

    GMainLoop *loop;
    GstElement *pipeline;

    // Initialize GStreamer
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    // Try to create pipeline 
    pipeline = gst_parse_launch(argv[1], NULL);
    if (pipeline == NULL)
        g_error ("Failed to launch pipeline");

    // Try to find element with name=conv
    GstElement *conv = gst_bin_get_by_name(GST_BIN(pipeline), "conv");
    if (conv == NULL)
        g_error ("Failed to find conv in pipeline");

    // Get its src pad and add the probe to it
    GstPad *pad = gst_element_get_static_pad (conv, "src");
    //gulong probe_id =
    gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER,(GstPadProbeCallback) conv_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref (pad);
    gst_object_unref (conv);

    // Try running the pipeline and wait until it's up and running or failed
    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    if (gst_element_get_state (pipeline, NULL, NULL, -1) == GST_STATE_CHANGE_FAILURE) {
       g_error ("Failed to go into PLAYING state");
    }
    g_print ("Running ...\n");
    g_main_loop_run (loop);

    // Exit
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);

    return 0;
}*/


/* Built with:
gcc -Wall -O3 -o test_surface -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0  -I/usr/lib/aarch64-linux-gnu/glib-2.0/include -I/usr/local/cuda/include -I/usr/local/opencv-4.6.0/include/opencv4 -I/usr/src/jetson_multimedia_api/include/ test_surface.cpp -lstdc++ -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -L/usr/local/opencv-4.6.0/lib -lopencv_core -lopencv_cudaarithm -lEGL -lGLESv2 -L/usr/lib/aarch64-linux-gnu/tegra/ -lcuda -lnvbuf_utils -lnvbufsurface -L/usr/local/cuda/lib64/ -lcudart

export LD_LIBRARY_PATH=/usr/local/opencv-4.6.0/lib/:$LD_LIBRARY_PATH

# RGBA works 
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv ! video/x-raw(memory:NVMM),width=1920,height=1080,format=RGBA ! nv3dsink"

# I420 works
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv ! video/x-raw(memory:NVMM),width=1920,height=1080,format=I420 ! nv3dsink"

# NV12 with pitch linear works
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv bl-output=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nv3dsink"

# NV12 with block linear is not supported:
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv bl-output=1 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nv3dsink"

# If you have some decoder or else outputting BL NV12, there is a bug in nvvidconv for converting into NV12 pitch linear. DS nvvideoconvert works for the following case:
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvideoconvert  bl-output=1 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nvvideoconvert name=conv bl-output=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nv3dsink"

# ...but it doesn't work with nvv4l2decoder (at least with the following case), although bl-output=0 is set, BL format is reported on src pad probe. 
# So a workaround would be converting to I420:
./test_surface "filesrc location=/home/nvidia/Videos/bbb_sunflower_1080p_60fps_normal.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! queue ! nvvideoconvert ! video/x-raw(memory:NVMM),format=I420 ! nvvideoconvert bl-output=0 name=conv ! video/x-raw(memory:NVMM),format=NV12 ! nv3dsink"

# RGBA works from other memory types such as:
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv compute-hw=GPU nvbuf-memory-type=1 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=RGBA ! nv3dsink"

# I420 may work from other memory types such as:
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv compute-hw=GPU nvbuf-memory-type=1 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=I420 ! nv3dsink"

# NV12 may work from other memory types such as:
./test_surface "videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! nvvidconv name=conv compute-hw=GPU nvbuf-memory-type=1 bl-output=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nv3dsink"

*/