cmake_minimum_required(VERSION 3.10)
project(YOLOv8_System)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的库
find_package(OpenCV)
find_package(CUDA REQUIRED)
find_package(Eigen3)
find_package(nlohmann_json)

# 设置TensorRT路径
set(TENSORRT_PATH "/C/onnx/tensorrt86/TensorRT-8.6.1.6")

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
    /usr/local/include
)

# 链接目录
link_directories(
    ${CUDA_LIBRARIES}
    ${TENSORRT_PATH}/lib
    /usr/local/lib
)

# CUDA核函数
cuda_add_library(cuda_kernels src/cuda_kernels.cu)

# 主可执行文件
add_executable(yolov8_system
    src/main.cpp
    src/camera_producer.cpp
    src/detection_consumer.cpp
    src/prediction_manager.cpp
    src/ptz_controller.cpp
)




# 链接库
# 使用关键字签名（PRIVATE）以避免与后续的 keyword 调用冲突
target_link_libraries(yolov8_system PRIVATE
    cuda_kernels
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    cudart
    turbojpeg
    pthread
)

if(nlohmann_json_FOUND)
    target_link_libraries(yolov8_system PRIVATE nlohmann_json::nlohmann_json)
endif()

if(Eigen3_FOUND)
    target_link_libraries(yolov8_system PRIVATE Eigen3::Eigen)
endif()

# 设置CUDA架构
set_target_properties(cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})