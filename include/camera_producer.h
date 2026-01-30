#pragma once

#include "common.h"
#include "ring_buffer.h"
#include <turbojpeg.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <atomic>
#include <thread>
#include <chrono>

class CameraProducer {
private:
    int cam_fd_ = -1;
    struct buffer {
        void* start;
        size_t length;
    } *buffers_ = nullptr;
    int buffer_count_ = 0;
    
    FrameRingBuffer& ring_buffer_;
    std::atomic<bool> running_{true};
    std::thread producer_thread_;
    tjhandle tj_handle_ = nullptr;

    // 统计数据
    std::atomic<int> frames_seen_{0};                  // 包括被跳过的帧
    std::atomic<int> skipped_frames_{0};               // 已跳过的帧数
    std::atomic<int> frames_pushed_{0};                // 已推送用于检测的帧数
    std::atomic<uint64_t> total_capture_decode_time_ms_{0}; // 累计获取+解码时间（毫秒）
    std::atomic<uint64_t> frames_pushed_total_{0};     // 总推送计数（用于计算fps）
    std::chrono::steady_clock::time_point last_fps_check_time_;
    uint64_t last_frames_pushed_count_ = 0;
    
public:
    explicit CameraProducer(FrameRingBuffer& ring_buffer);
    ~CameraProducer();
    
    bool init();
    void start();
    void stop();
    float get_fps();

    // 统计查询
    int get_skipped_frames() const { return skipped_frames_.load(); }
    double get_avg_capture_decode_time_ms() const {
        uint64_t frames = frames_pushed_.load();
        return frames ? (double)total_capture_decode_time_ms_.load() / frames : 0.0;
    }
    
private:
    void producerLoop();
    cv::Mat captureFrame();
    int decodeMJPG(const uint8_t* mjpeg_data, size_t size, cv::Mat& rgb_mat);
    void cleanup();
};