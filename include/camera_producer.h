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
    
public:
    explicit CameraProducer(FrameRingBuffer& ring_buffer);
    ~CameraProducer();
    
    bool init();
    void start();
    void stop();
    float get_fps();
    
private:
    void producerLoop();
    cv::Mat captureFrame();
    int decodeMJPG(const uint8_t* mjpeg_data, size_t size, cv::Mat& rgb_mat);
    void cleanup();
};