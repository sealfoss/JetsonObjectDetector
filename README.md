# Jetson Orin C++ OpenCV CUDA Project

## Overview
This project is designed to leverage the capabilities of the Jetson Orin NX platform for image processing using OpenCV and CUDA. It provides a framework for loading, processing, and saving images efficiently.

## Project Structure
```
jetson-orin-cpp-opencv-cuda
├── src
│   ├── main.cpp            # Entry point of the application
│   └── image_processor.cpp  # Implementation of the ImageProcessor class
├── include
│   └── image_processor.h    # Declaration of the ImageProcessor class
├── cmake
│   └── modules
│       └── FindCUDA.cmake   # Custom CMake module for finding CUDA
├── CMakeLists.txt           # CMake configuration file
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

## Dependencies
- OpenCV: Ensure that OpenCV is installed on your system. You can install it using the package manager or build it from source.
- CUDA: Make sure CUDA is installed and properly configured on your Jetson Orin NX.

## Building the Project
1. Clone the repository:
   ```
   git clone <repository-url>
   cd jetson-orin-cpp-opencv-cuda
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

## Running the Application
After building the project, you can run the application using:
```
./your_executable_name <image_path>
```
Replace `<image_path>` with the path to the image you want to process.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Your contributions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for more details.