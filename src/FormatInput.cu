#include "FormatInput.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PLANE_FMT_SCALAR 255.0f

__global__ void RgbaToPlanarRgbAndNormalizeKernel(
    const unsigned char* rgbaData, 
    float* redPlane, 
    float* greenPlane, 
    float* bluePlane, 
    int width, 
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int rgbaIdx = (y * width + x) * 4;
    const int planarIdx = y * width + x;

    // Read interleaved RGBA data (assuming 8-bit unsigned chars)
    // and normalize to float values between 0.0 and 1.0.
    redPlane[planarIdx] = 
        static_cast<float>(rgbaData[rgbaIdx + 0]) / PLANE_FMT_SCALAR;
    greenPlane[planarIdx] = 
        static_cast<float>(rgbaData[rgbaIdx + 1]) / PLANE_FMT_SCALAR;
    bluePlane[planarIdx] = 
        static_cast<float>(rgbaData[rgbaIdx + 2]) / PLANE_FMT_SCALAR;
}

void ConvertRgbaToPlanarRgbAndNormalize(
    const unsigned char* rgbaData,
    float* redPlane, 
    float* greenPlane, 
    float* bluePlane, 
    int width,
    int height,
    cudaStream_t stream)
{

    // Set up kernel launch parameters
    dim3 threads_per_block(32, 32);
    dim3 num_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the new kernel
    RgbaToPlanarRgbAndNormalizeKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        rgbaData,
        redPlane,
        greenPlane,
        bluePlane,
        width,
        height);

    // Wait for the kernel to complete (optional)
    cudaStreamSynchronize(stream);
}
