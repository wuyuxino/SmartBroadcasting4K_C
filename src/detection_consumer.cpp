// detection_consumer.cpp
#include "detection_consumer.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <cuda_fp16.h>  // 添加这个头文件
#include <unistd.h>     // 添加这个头文件

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() { 
    cleanup(); 
}

bool InferenceEngine::init(const std::string& engine_path) {
    // 加载TensorRT引擎
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        std::cerr << "创建TensorRT Runtime失败" << std::endl;
        return false;
    }
    
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.is_open()) {
        std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
        return false;
    }
    
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();
    
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineSize);
    if (!engine_) {
        std::cerr << "引擎反序列化失败" << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "创建执行上下文失败" << std::endl;
        return false;
    }
    
    // 获取绑定信息
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* tensorName = engine_->getIOTensorName(i);
        nvinfer1::DataType dtype = engine_->getTensorDataType(tensorName);
        nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
        
        if (i == 0) {
            input_tensor_name_ = tensorName;
            input_dims_ = dims;
            input_type_ = dtype;
        } else {
            output_tensor_name_ = tensorName;
            output_dims_ = dims;
        }
    }
    
    // 分配GPU内存
    cudaSetDevice(0);
    cudaStreamCreate(&stream_);
    
    // 计算内存大小
    size_t input_elem_size = (input_type_ == nvinfer1::DataType::kHALF) ? 2 : 4;
    input_size_ = 1;
    for (int i = 0; i < input_dims_.nbDims; i++) {
        input_size_ *= input_dims_.d[i];
    }
    input_size_ *= input_elem_size;
    
    nvinfer1::DataType output_type = engine_->getTensorDataType(output_tensor_name_.c_str());
    size_t output_elem_size = (output_type == nvinfer1::DataType::kHALF) ? 2 : 4;
    output_size_ = 1;
    for (int i = 0; i < output_dims_.nbDims; i++) {
        output_size_ *= output_dims_.d[i];
    }
    output_size_ *= output_elem_size;
    
    // 分配内存
    frame_size_ = Config::CAM_HEIGHT * Config::CAM_WIDTH * 3 * sizeof(uint8_t);
    cudaMalloc(&d_frame_buffer_, frame_size_);
    cudaMalloc(&d_input_buffer_, input_size_);
    cudaMalloc(&d_output_buffer_, output_size_);
    
    // 初始化输出缓冲区
    int output_elements = output_dims_.d[0] * output_dims_.d[1] * output_dims_.d[2];
    output_float_.resize(output_elements);
    output_half_converted_.resize(output_elements);
    
    std::cout << "✅ 推理引擎初始化完成" << std::endl;
    std::cout << "   输入张量: " << input_tensor_name_ << std::endl;
    std::cout << "   输出张量: " << output_tensor_name_ << std::endl;
    
    return true;
}

std::vector<DetectionBox> InferenceEngine::infer(const cv::Mat& frame) {
    std::vector<DetectionBox> boxes;
    
    // 异步拷贝帧到GPU
    cudaMemcpyAsync(d_frame_buffer_, frame.data, frame_size_,
                   cudaMemcpyHostToDevice, stream_);
    
    // 预处理
    launch_preprocess_kernel(
        (const uint8_t*)d_frame_buffer_,
        d_input_buffer_,
        Config::CAM_WIDTH,
        Config::CAM_HEIGHT,
        Config::MODEL_WIDTH,
        Config::MODEL_HEIGHT,
        (input_type_ == nvinfer1::DataType::kHALF),
        stream_
    );
    
    // 设置TensorRT绑定
    context_->setTensorAddress(input_tensor_name_.c_str(), d_input_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), d_output_buffer_);
    
    // 执行推理
    void* bindings[] = {d_input_buffer_, d_output_buffer_};
    bool success = context_->executeV2(bindings);  // 使用 executeV2 而不是 enqueueV2
    
    if (!success) {
        std::cerr << "推理执行失败" << std::endl;
        return boxes;
    }
    
    // 拷贝结果到CPU
    nvinfer1::DataType output_type = engine_->getTensorDataType(output_tensor_name_.c_str());
    if (output_type == nvinfer1::DataType::kHALF) {
        std::vector<__half> output_half_temp(output_float_.size());
        cudaMemcpy(output_half_temp.data(), d_output_buffer_, 
                   output_size_, cudaMemcpyDeviceToHost);
        
        // 转换为float
        for (size_t i = 0; i < output_float_.size(); i++) {
            output_float_[i] = __half2float(output_half_temp[i]);
        }
    } else {
        cudaMemcpy(output_float_.data(), d_output_buffer_,
                   output_size_, cudaMemcpyDeviceToHost);
    }
    
    // 解码输出
    decodeYOLOv8Output(output_float_, boxes);
    
    // NMS后处理
    postprocessNMS(boxes);
    
    return boxes;
}

void InferenceEngine::decodeYOLOv8Output(const std::vector<float>& output, 
                                      std::vector<DetectionBox>& boxes) {
    boxes.clear();
    
    int channels = 5;
    int points = 171360;
    
    if (output.size() != (size_t)(channels * points)) {
        std::cout << "⚠️ 输出大小不匹配: 期望 " << channels * points 
                  << ", 实际 " << output.size() << std::endl;
        return;
    }
    
    int confidence_channel = 4;
    
    for (int p = 0; p < points; p++) {
        float conf = output[confidence_channel * points + p];
        
        if (conf > Config::CONF_THRESHOLD) {
            float cx = output[0 * points + p];
            float cy = output[1 * points + p];
            float w = output[2 * points + p];
            float h = output[3 * points + p];
            
            if (w <= 0 || h <= 0) continue;
            
            float x1 = cx - w/2.0f;
            float y1 = cy - h/2.0f;
            float x2 = cx + w/2.0f;
            float y2 = cy + h/2.0f;
            
            x1 = std::max(0.0f, std::min(x1, (float)Config::MODEL_WIDTH - 1));
            y1 = std::max(0.0f, std::min(y1, (float)Config::MODEL_HEIGHT - 1));
            x2 = std::max(x1 + 1.0f, std::min(x2, (float)Config::MODEL_WIDTH - 1));
            y2 = std::max(y1 + 1.0f, std::min(y2, (float)Config::MODEL_HEIGHT - 1));
            
            float box_width = x2 - x1;
            float box_height = y2 - y1;
            
            if (box_width < 10 || box_height < 10 || 
                box_width > Config::MODEL_WIDTH * 0.5f || 
                box_height > Config::MODEL_HEIGHT * 0.5f) {
                continue;
            }
            
            boxes.push_back(DetectionBox::fromRect(
                cv::Rect(x1, y1, box_width, box_height), conf));
        }
    }
}

void InferenceEngine::postprocessNMS(std::vector<DetectionBox>& boxes) {
    if (boxes.empty()) return;
    
    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    
    for (const auto& box : boxes) {
        rects.push_back(box.toRect());
        scores.push_back(box.confidence);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, scores, 0.001f, Config::NMS_THRESHOLD, indices);
    
    std::vector<DetectionBox> nms_boxes;
    for (int idx : indices) {
        nms_boxes.push_back(boxes[idx]);
    }
    
    boxes = nms_boxes;
}

void InferenceEngine::cleanup() {
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
    
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    if (d_frame_buffer_) {
        cudaFree(d_frame_buffer_);
        d_frame_buffer_ = nullptr;
    }
    
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
}

DetectionConsumer::DetectionConsumer(FrameRingBuffer& ring_buffer, 
                                   DetectionResultQueue& result_queue)
    : ring_buffer_(ring_buffer), result_queue_(result_queue) {}

DetectionConsumer::~DetectionConsumer() {
    stop();
}

bool DetectionConsumer::init(const std::string& engine_path) {
    return engine_.init(engine_path);
}

void DetectionConsumer::start() {
    running_ = true;
    consumer_thread_ = std::thread(&DetectionConsumer::consumerLoop, this);
}

void DetectionConsumer::stop() {
    running_ = false;
    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }
}

float DetectionConsumer::get_fps() {
    static int frame_count = 0;
    static auto last_time = std::chrono::steady_clock::now();
    static float fps = 0;
    
    frame_count++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_time).count() / 1000.0f;
    
    if (elapsed > 1.0f) {
        fps = frame_count / elapsed;
        frame_count = 0;
        last_time = now;
    }
    
    return fps;
}

void DetectionConsumer::consumerLoop() {
    while (running_) {
        FrameData frame_data;
        
        // 从环形缓冲区获取最新帧
        if (ring_buffer_.peek_latest(frame_data) && frame_data.valid) {
            // 执行推理
            auto start_time = std::chrono::steady_clock::now();
            
            std::vector<DetectionBox> boxes = engine_.infer(frame_data.frame);
            
            auto end_time = std::chrono::steady_clock::now();
            auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            // 推送到结果队列
            result_queue_.push(std::move(boxes));
            
            // 控制帧率（45FPS ≈ 22.22ms/帧）
            std::this_thread::sleep_until(start_time + std::chrono::milliseconds(22));
        } else {
            // 缓冲区空，短暂等待
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}