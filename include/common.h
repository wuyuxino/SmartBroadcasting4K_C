#pragma once

#include <cstdint>
#if defined(__has_include)
  #if __has_include(<opencv2/opencv.hpp>)
    #include <opencv2/opencv.hpp>
  #else
    // Minimal OpenCV stubs when OpenCV is not available (allows lightweight testing)
    namespace cv {
        class Mat {};
        struct Rect { int x; int y; int width; int height; Rect(int x_=0,int y_=0,int w_=0,int h_=0):x(x_),y(y_),width(w_),height(h_){} };
        struct Point2f { float x; float y; Point2f(float _x=0, float _y=0):x(_x),y(_y){} };
    }
  #endif
#else
  #include <opencv2/opencv.hpp>
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

// common.h 中添加
#ifdef __cplusplus
extern "C" {
#endif

void launch_preprocess_kernel(
    const uint8_t* src,
    void* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    bool is_fp16,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

// 通用配置
struct Config {
    // 摄像头配置
    static constexpr const char* CAMERA_DEVICE = "/dev/video0";
    static constexpr int CAM_WIDTH = 3840;
    static constexpr int CAM_HEIGHT = 2160;
    
    // 模型配置
    static constexpr int MODEL_WIDTH = 3840;
    static constexpr int MODEL_HEIGHT = 2176;
    static constexpr const char* ENGINE_PATH = "best.engine";
    
    // 显示配置
    static constexpr int SHOW_WIDTH = 1920;
    static constexpr int SHOW_HEIGHT = 1088;
    static constexpr bool ENABLE_DISPLAY_THREAD = false; // 是否启用显示线程（默认关闭，减少资源占用）
    
    // 阈值配置
    static constexpr float CONF_THRESHOLD = 0.6f;
    static constexpr float NMS_THRESHOLD = 0.3f;
    
    // 缓冲区配置
    static constexpr int RING_BUFFER_SIZE = 3;      // 环形缓冲区大小
    static constexpr int DETECTION_QUEUE_SIZE = 5;  // 检测结果队列大小

    // 启动时跳过的帧数（避免前若干帧不稳定）
    static constexpr int SKIP_INITIAL_FRAMES = 50;

    // 发送命令控制设置
    static constexpr int WINDOW = 5; // 平滑窗口大小（帧数）
    static constexpr int CHECK_MS = 16; // 检查间隔（毫秒）
    static constexpr int COOLDOWN_MS = 16; // 控制冷却时间（毫秒）
    static constexpr double SUM_MOVE_THRESH = 40.0; // 累积移动阈值（像素）
    static constexpr int PREDICTION_HORIZON = 2; // 预测时间步长（帧数）默认使用第三帧（index=2）
};

// 检测框结构
struct DetectionBox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
    int frame_id;            // 对应推理帧ID（由 consumer 填充）
    uint64_t timestamp_ms;   // 可选：推理时的时间戳

    // 序列化方法
    cv::Rect toRect() const {
        return cv::Rect(x1, y1, x2-x1, y2-y1);
    }
    
    // 从Rect转换
    static DetectionBox fromRect(const cv::Rect& rect, float conf, int class_id = 0, int frame_id = 0, uint64_t ts = 0) {
        DetectionBox box;
        box.x1 = rect.x;
        box.y1 = rect.y;
        box.x2 = rect.x + rect.width;
        box.y2 = rect.y + rect.height;
        box.confidence = conf;
        box.class_id = class_id;
        box.class_name = "football";
        box.frame_id = frame_id;
        box.timestamp_ms = ts;
        return box;
    }
};

// 帧数据结构
struct FrameData {
    cv::Mat frame;              // 原始帧
    uint64_t timestamp;         // 时间戳
    int frame_id;               // 帧ID
    bool valid;                 // 是否有效
    
    FrameData() : valid(false), frame_id(0), timestamp(0) {}
    explicit FrameData(cv::Mat&& mat, int id = 0) 
        : frame(std::move(mat)), frame_id(id), valid(true) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// CUDA核函数声明
extern "C" void launch_preprocess_kernel(
    const uint8_t* src,
    void* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    bool is_fp16,
    cudaStream_t stream
);