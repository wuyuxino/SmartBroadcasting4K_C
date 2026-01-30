// detection_consumer.h
#pragma once

#include "common.h"
#include "ring_buffer.h"
#include "detection_queue.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <thread>
#include <chrono>

// 前向声明，避免包含 cuda_fp16.h 引起的问题
struct __half;

class InferenceEngine {
private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    nvinfer1::DataType input_type_;
    
    // GPU内存
    void* d_input_buffer_ = nullptr;
    void* d_output_buffer_ = nullptr;
    void* d_frame_buffer_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    size_t frame_size_ = 0;
    
    // CUDA流
    cudaStream_t stream_ = nullptr;
    
    // 输出缓冲区
    std::vector<float> output_float_;
    std::vector<float> output_half_converted_;  // 改为 float，用于存储转换后的值
    
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TRT] " << msg << std::endl;
            }
        }
    } logger_;
    
public:
    InferenceEngine();
    ~InferenceEngine();
    
    bool init(const std::string& engine_path);
    std::vector<DetectionBox> infer(const cv::Mat& frame);
    
private:
    void decodeYOLOv8Output(const std::vector<float>& output, 
                          std::vector<DetectionBox>& boxes);
    void postprocessNMS(std::vector<DetectionBox>& boxes);
    void cleanup();
};

class DetectionConsumer {
private:
    FrameRingBuffer& ring_buffer_;
    DetectionResultQueue& result_queue_;
    InferenceEngine engine_;
    std::atomic<bool> running_{true};
    std::thread consumer_thread_;
    
public:
    DetectionConsumer(FrameRingBuffer& ring_buffer, 
                     DetectionResultQueue& result_queue);
    ~DetectionConsumer();
    
    bool init(const std::string& engine_path);
    void start();
    void stop();
    float get_fps();
    
private:
    void consumerLoop();
};