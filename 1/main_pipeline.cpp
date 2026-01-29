#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>

// 你的现有头文件
#include "yolov8_detector.h"      // YOLOv8检测器
#include "trajectory_predictor.h"  // 轨迹预测器
#include "gimbal_controller.h"     // 云台控制器
#include "frame_buffer.h"          // 帧缓冲区

// ====================== 全局配置 ======================
const int CAMERA_FPS = 60;
const int DETECTION_FPS = 45;      // 检测目标帧率
const int PREDICTION_FPS = 9;      // 预测帧率 (检测FPS/5)
const int BUFFER_SIZE = 3;         // 帧缓冲区大小
const int DETECTION_QUEUE_SIZE = 5; // 检测结果队列大小

// ====================== 全局状态 ======================
std::atomic<bool> running{true};

// ====================== 缓冲区定义 ======================
class FrameBuffer {
private:
    std::vector<cv::Mat> buffer;
    std::mutex mtx;
    int size;
    int write_idx = 0;
    
public:
    FrameBuffer(int size) : size(size) {
        buffer.resize(size);
    }
    
    void push(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mtx);
        buffer[write_idx] = frame.clone();
        write_idx = (write_idx + 1) % size;
    }
    
    cv::Mat getLatest() {
        std::lock_guard<std::mutex> lock(mtx);
        int read_idx = (write_idx - 1 + size) % size;
        return buffer[read_idx].clone();
    }
    
    cv::Mat getWithAge(int age) {
        std::lock_guard<std::mutex> lock(mtx);
        if (age >= size) age = size - 1;
        int read_idx = (write_idx - 1 - age + 2 * size) % size;
        return buffer[read_idx].clone();
    }
};

// 检测结果队列
template<typename T>
class LockedQueue {
private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;
    
public:
    LockedQueue(size_t max_size = 10) : max_size(max_size) {}
    
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (queue.size() >= max_size) {
            queue.pop();  // 丢弃最旧的
        }
        queue.push(item);
        cv.notify_one();
    }
    
    bool pop(T& item, int timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(mtx);
        if (timeout_ms > 0) {
            if (!cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [this]() { return !queue.empty(); })) {
                return false;
            }
        } else {
            cv.wait(lock, [this]() { return !queue.empty(); });
        }
        
        item = queue.front();
        queue.pop();
        return true;
    }
    
    std::vector<T> getAll() {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<T> result;
        while (!queue.empty()) {
            result.push_back(queue.front());
            queue.pop();
        }
        return result;
    }
    
    bool isFull() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size() >= max_size;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};

// ====================== 数据结构 ======================
struct DetectionResult {
    int64_t timestamp;  // 时间戳（微秒）
    std::vector<DetectionBox> boxes;
    cv::Mat frame;      // 对应的帧（可选）
    
    DetectionResult() : timestamp(0) {}
    DetectionResult(const std::vector<DetectionBox>& b, const cv::Mat& f = cv::Mat())
        : boxes(b), frame(f) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

struct PredictionResult {
    int64_t timestamp;
    float predicted_x;      // 预测的x坐标
    float predicted_y;      // 预测的y坐标
    float confidence;       // 预测置信度
    
    PredictionResult(float x, float y, float conf = 1.0)
        : predicted_x(x), predicted_y(y), confidence(conf) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// ====================== 线程函数 ======================
void capture_thread(FrameBuffer& frame_buffer) {
    std::cout << "📹 采集线程启动 (目标: " << CAMERA_FPS << " FPS)" << std::endl;
    
    // 初始化摄像头（使用你的现有代码）
    if (camera_init() != 0) {
        std::cerr << "❌ 摄像头初始化失败" << std::endl;
        return;
    }
    
    float target_interval_ms = 1000.0f / CAMERA_FPS;
    
    while (running) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 采集一帧
        cv::Mat frame;
        float capture_time, decode_time;
        int ret = camera_capture_frame(frame, capture_time, decode_time);
        
        if (ret == 0) {
            // 放入缓冲区
            frame_buffer.push(frame);
            
            // 统计
            static int frame_count = 0;
            static auto last_stat_time = std::chrono::steady_clock::now();
            frame_count++;
            
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - last_stat_time).count();
            
            if (elapsed >= 2.0f) {  // 每2秒打印一次
                float fps = frame_count / elapsed;
                std::cout << "📹 采集FPS: " << fps << " | 缓冲帧数: 持续更新" << std::endl;
                frame_count = 0;
                last_stat_time = now;
            }
        }
        
        // 控制帧率
        auto end_time = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        float sleep_time = target_interval_ms - elapsed;
        
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time)));
        }
    }
    
    camera_release();
    std::cout << "📹 采集线程结束" << std::endl;
}

void detection_thread(FrameBuffer& frame_buffer, 
                     LockedQueue<DetectionResult>& detection_queue,
                     YOLOv8Detector& detector) {
    std::cout << "🔍 检测线程启动 (目标: " << DETECTION_FPS << " FPS)" << std::endl;
    
    float target_interval_ms = 1000.0f / DETECTION_FPS;
    int frame_count = 0;
    auto last_stat_time = std::chrono::steady_clock::now();
    
    while (running) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. 从缓冲区获取最新帧
        cv::Mat frame = frame_buffer.getLatest();
        
        if (!frame.empty()) {
            // 2. 执行检测
            std::vector<DetectionBox> boxes = detector.detect(frame);
            
            // 3. 将结果放入队列
            if (!boxes.empty()) {
                detection_queue.push(DetectionResult(boxes, frame));
                frame_count++;
            }
        }
        
        // 统计
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_stat_time).count();
        
        if (elapsed >= 2.0f) {
            float fps = frame_count / elapsed;
            std::cout << "🔍 检测FPS: " << fps << " | 队列长度: " << detection_queue.size() << std::endl;
            frame_count = 0;
            last_stat_time = now;
        }
        
        // 控制帧率
        auto end_time = std::chrono::steady_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        float sleep_time = target_interval_ms - elapsed_ms;
        
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time)));
        }
    }
    
    std::cout << "🔍 检测线程结束" << std::endl;
}

void prediction_thread(LockedQueue<DetectionResult>& detection_queue,
                      LockedQueue<PredictionResult>& prediction_queue,
                      TrajectoryPredictor& predictor) {
    std::cout << "🎯 预测线程启动 (目标: " << PREDICTION_FPS << " FPS)" << std::endl;
    
    float target_interval_ms = 1000.0f / PREDICTION_FPS;
    std::vector<DetectionResult> recent_detections;
    
    while (running) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. 检查是否有足够的检测结果
        if (detection_queue.isFull()) {
            // 2. 获取所有检测结果
            std::vector<DetectionResult> detections = detection_queue.getAll();
            
            if (!detections.empty()) {
                // 3. 执行预测
                PredictionResult prediction = predictor.predict(detections);
                
                // 4. 将预测结果放入队列
                prediction_queue.push(prediction);
                
                // 打印预测结果
                std::cout << "🎯 预测位置: (" << prediction.predicted_x 
                          << ", " << prediction.predicted_y 
                          << ") 置信度: " << prediction.confidence << std::endl;
            }
        }
        
        // 控制帧率
        auto end_time = std::chrono::steady_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        float sleep_time = target_interval_ms - elapsed_ms;
        
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time)));
        }
    }
    
    std::cout << "🎯 预测线程结束" << std::endl;
}

void control_thread(LockedQueue<PredictionResult>& prediction_queue,
                   GimbalController& controller) {
    std::cout << "🎮 控制线程启动" << std::endl;
    
    while (running) {
        // 1. 从队列获取预测结果
        PredictionResult prediction;
        if (prediction_queue.pop(prediction, 100)) {  // 100ms超时
            // 2. 计算云台控制指令
            GimbalCommand command = controller.calculateCommand(prediction);
            
            // 3. 发送控制指令
            controller.sendCommand(command);
            
            // 打印控制指令
            std::cout << "🎮 发送指令: 俯仰=" << command.pitch 
                      << "°, 方位=" << command.yaw 
                      << "°, 速度=" << command.speed << std::endl;
        }
    }
    
    std::cout << "🎮 控制线程结束" << std::endl;
}

// ====================== 主函数 ======================
int main() {
    std::cout << "🚀 启动实时目标跟踪与控制系统" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "📊 系统配置:" << std::endl;
    std::cout << "  摄像头帧率: " << CAMERA_FPS << " FPS" << std::endl;
    std::cout << "  检测帧率: " << DETECTION_FPS << " FPS" << std::endl;
    std::cout << "  预测帧率: " << PREDICTION_FPS << " FPS" << std::endl;
    std::cout << "  帧缓冲区: " << BUFFER_SIZE << " 帧" << std::endl;
    std::cout << "  检测队列: " << DETECTION_QUEUE_SIZE << " 个结果" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // 初始化组件
    FrameBuffer frame_buffer(BUFFER_SIZE);
    LockedQueue<DetectionResult> detection_queue(DETECTION_QUEUE_SIZE);
    LockedQueue<PredictionResult> prediction_queue(3);  // 预测结果队列
    
    // 初始化检测器、预测器、控制器（需要你实现）
    YOLOv8Detector detector;
    TrajectoryPredictor predictor;
    GimbalController controller;
    
    // 创建线程
    std::thread capture_t(capture_thread, std::ref(frame_buffer));
    std::thread detection_t(detection_thread, std::ref(frame_buffer), 
                           std::ref(detection_queue), std::ref(detector));
    std::thread prediction_t(prediction_thread, std::ref(detection_queue),
                           std::ref(prediction_queue), std::ref(predictor));
    std::thread control_t(control_thread, std::ref(prediction_queue),
                         std::ref(controller));
    
    // 等待用户输入退出
    std::cout << "\n按回车键退出..." << std::endl;
    std::cin.get();
    running = false;
    
    // 等待线程结束
    capture_t.join();
    detection_t.join();
    prediction_t.join();
    control_t.join();
    
    std::cout << "✅ 系统正常退出" << std::endl;
    return 0;
}