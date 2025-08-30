cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(PROJECT_NAME JetsonObjectDetector)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Use at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")


find_package(CUDA REQUIRED) # If you need to explicitly find CUDA components
if (CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit found")
  message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()

# Find TensorRt
set(TENSORRT_LIB "/usr/lib/aarch64-linux-gnu/")
set(TENSORRT_INC "/usr/include/aarch64-linux-gnu/")
find_library(TENSORRT_NVINFER nvinfer PATHS ${TENSORRT_LIB} NO_DEFAULT_PATH)
find_library(TENSORRT_NVDISPATCH nvinfer_dispatch PATHS ${TENSORRT_LIB} NO_DEFAULT_PATH)
find_library(TENSORRT_NVPLUGIN nvinfer_plugin PATHS ${TENSORRT_LIB} NO_DEFAULT_PATH)
find_library(TENSORRT_NVONNX nvonnxparser PATHS ${TENSORRT_LIB} NO_DEFAULT_PATH)
find_path(TENSORRT_INCLUDE_DIR nvinfer.h PATHS ${TENSORRT_INC} NO_DEFAULT_PATH)

#Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTLIBS REQUIRED
    glib-2.0
    gobject-2.0
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-audio-1.0
    gstreamer-base-1.0
    gstreamer-video-1.0
)
message(STATUS "GStreamer library status:")
message(STATUS "    include directories: ${GSTLIBS_INCLUDE_DIRS}")
message(STATUS "    library directories: ${GSTLIBS_LIBRARY_DIRS}")
message(STATUS "    libraries: ${GSTLIBS_LIBRARIES}")

# Assuming DeepStream is installed in the default location
set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream") 
set(DEEPSTREAM_LIB ${DEEPSTREAM_DIR}/lib)
# Add the include directory for nvbufsurface.h
include_directories(${DEEPSTREAM_DIR}/sources/includes)
# Add the library directory where libnvbufsurface.so is located
link_directories(${DEEPSTREAM_DIR}/lib)
find_library(NVBUFSURFACE nvbufsurface PATHS ${DEEPSTREAM_LIB} NO_DEFAULT_PATH)
find_library(NVBUFSURFTANSFORM nvbufsurftransform PATHS ${DEEPSTREAM_LIB} NO_DEFAULT_PATH)
message(STATUS "NVBUF surface: ${NVBUFSURFACE}")
message(STATUS "NVBUF surface transform: ${NVBUFSURFTANSFORM}")

# Project sources
file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Add executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Include the project's public headers
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${GSTLIBS_INCLUDE_DIRS}
  ${TENSORRT_INC}
  ${DEEPSTREAM_DIR}/sources/includes
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES} cuda
    ${GSTLIBS_LIBRARIES}
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVDISPATCH}
    ${TENSORRT_NVPLUGIN}
    ${TENSORRT_NVONNX}
    ${NVBUFSURFACE}
    ${NVBUFSURFTANSFORM}
)



# Link OpenCV and CUDA runtime
#if(TARGET OpenCV::opencv_core)
  #target_link_libraries(${PROJECT_NAME} PRIVATE OpenCV::opencv_core OpenCV::opencv_imgproc OpenCV::opencv_imgcodecs OpenCV::opencv_highgui ${CUDA_TARGET_LIBS})
#else()
  # fallback to the older variable
  #target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_TARGET_LIBS})
#endif()

# Recommended compiler flags for Jetson (adjust as needed)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

# Install rules (optional)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
