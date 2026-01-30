#include "common.h"
#include "ring_buffer.h"
#include "detection_queue.h"
#include "camera_producer.h"
#include "detection_consumer.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <signal.h>

// 全局变量
std::atomic<bool> running(true);
FrameRingBuffer frame_buffer(Config::RING_BUFFER_SIZE);
DetectionResultQueue detection_queue(Config::DETECTION_QUEUE_SIZE);

void signal_handler(int sig) {
    std::cout << "\n收到停止信号，正在清理资源..." << std::endl;
    running = false;
}

void displayThread() {
    cv::namedWindow("YOLOv8 Detection", cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow("YOLOv8 Detection", Config::SHOW_WIDTH, Config::SHOW_HEIGHT);
    cv::moveWindow("YOLOv8 Detection", 100, 100);
    
    while (running) {
        std::vector<DetectionBox> boxes;
        
        // 获取最新检测结果
        if (detection_queue.peek_latest(boxes) && !boxes.empty()) {
            // 获取最新帧用于显示
            FrameData frame_data;
            if (frame_buffer.peek_latest(frame_data) && frame_data.valid) {
                if (frame_data.frame.empty()) continue;
                cv::Mat display_frame;
                cv::resize(frame_data.frame, display_frame, 
                          cv::Size(Config::SHOW_WIDTH, Config::SHOW_HEIGHT));
                if (display_frame.empty()) continue;
                
                // 绘制检测结果
                for (const auto& box : boxes) {
                    cv::rectangle(display_frame,
                                cv::Point(box.x1 * Config::SHOW_WIDTH / Config::MODEL_WIDTH,
                                         box.y1 * Config::SHOW_HEIGHT / Config::MODEL_HEIGHT),
                                cv::Point(box.x2 * Config::SHOW_WIDTH / Config::MODEL_WIDTH,
                                         box.y2 * Config::SHOW_HEIGHT / Config::MODEL_HEIGHT),
                                cv::Scalar(0, 255, 0), 2);
                    
                    std::string label = box.class_name + ": " + 
                                      std::to_string(box.confidence).substr(0, 4);
                    
                    cv::putText(display_frame, label,
                              cv::Point(box.x1 * Config::SHOW_WIDTH / Config::MODEL_WIDTH + 5,
                                       box.y1 * Config::SHOW_HEIGHT / Config::MODEL_HEIGHT - 5),
                              cv::FONT_HERSHEY_SIMPLEX, 0.5,
                              cv::Scalar(0, 255, 0), 2);
                }
                
                cv::imshow("YOLOv8 Detection", display_frame);
            }
        }
        
        // 检查退出键
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            running = false;
            break;
        }
        
        // 控制显示帧率（60FPS）
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    
    std::cout << "🚀 启动YOLOv8多线程检测系统" << std::endl;
    std::cout << "📷 生产者帧率: 60 FPS" << std::endl;
    std::cout << "🔍 消费者帧率: 45 FPS" << std::endl;
    std::cout << "🔄 环形缓冲区大小: " << Config::RING_BUFFER_SIZE << std::endl;
    std::cout << "📊 检测结果队列大小: " << Config::DETECTION_QUEUE_SIZE << std::endl;
    
    // 初始化生产者（摄像头）
    CameraProducer producer(frame_buffer);
    if (!producer.init()) {
        std::cerr << "❌ 摄像头初始化失败" << std::endl;
        return -1;
    }
    
    // 初始化消费者（检测）
    DetectionConsumer consumer(frame_buffer, detection_queue);
    if (!consumer.init(Config::ENGINE_PATH)) {
        std::cerr << "❌ 推理引擎初始化失败" << std::endl;
        return -1;
    }
    
    // 启动各个线程
    producer.start();
    consumer.start();
    
    // 启动显示线程
    std::thread display_thread(displayThread);
    
    // 主循环：监控性能
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        frame_count++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time).count() / 1000.0f;
        
        if (elapsed >= 1.0f) {
            std::cout << "\r📊 系统状态: ";
            std::cout << "缓冲区: " << frame_buffer.size() << "/" << frame_buffer.capacity();
            std::cout << " | 检测队列: " << detection_queue.size() << "/" 
                     << Config::DETECTION_QUEUE_SIZE;
            std::cout << " | FPS: " << frame_count / elapsed << "          ";
            std::cout.flush();
            
            frame_count = 0;
            start_time = now;
        }
    }
    
    // 停止所有线程
    producer.stop();
    consumer.stop();
    running = false;
    
    if (display_thread.joinable()) {
        display_thread.join();
    }
    
    std::cout << "\n\n✅ 系统正常退出" << std::endl;
    return 0;
}