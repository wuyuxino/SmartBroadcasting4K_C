// camera_producer.cpp
#include "camera_producer.h"
#include <turbojpeg.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <unistd.h>  // 添加这个头文件

CameraProducer::CameraProducer(FrameRingBuffer& ring_buffer) 
    : ring_buffer_(ring_buffer) {}

CameraProducer::~CameraProducer() {
    stop();
}

bool CameraProducer::init() {
    // 初始化摄像头
    cam_fd_ = open(Config::CAMERA_DEVICE, O_RDWR | O_NONBLOCK);
    if (cam_fd_ < 0) {
        perror("摄像头打开失败");
        return false;
    }
    
    // 设置摄像头参数
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = Config::CAM_WIDTH;
    fmt.fmt.pix.height = Config::CAM_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (ioctl(cam_fd_, VIDIOC_S_FMT, &fmt) == -1) {
        perror("格式设置失败");
        close(cam_fd_);
        return false;
    }
    
    // 分配缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 8;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(cam_fd_, VIDIOC_REQBUFS, &req) == -1) {
        perror("缓冲区请求失败");
        close(cam_fd_);
        return false;
    }
    
    buffer_count_ = req.count;
    buffers_ = new buffer[req.count];
    
    // 映射缓冲区
    for (int i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) == -1) {
            perror("缓冲区查询失败");
            cleanup();
            return false;
        }
        
        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length, 
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED, cam_fd_, buf.m.offset);
        
        if (buffers_[i].start == MAP_FAILED) {
            perror("缓冲区映射失败");
            cleanup();
            return false;
        }
        
        if (ioctl(cam_fd_, VIDIOC_QBUF, &buf) == -1) {
            perror("缓冲区入队失败");
            cleanup();
            return false;
        }
    }
    
    // 启动流
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(cam_fd_, VIDIOC_STREAMON, &type) == -1) {
        perror("流启动失败");
        cleanup();
        return false;
    }
    
    // 初始化TurboJPEG
    tj_handle_ = tjInitDecompress();
    if (!tj_handle_) {
        std::cerr << "TurboJPEG初始化失败" << std::endl;
        cleanup();
        return false;
    }
    
    std::cout << "✅ 摄像头初始化完成" << std::endl;
    return true;
}

void CameraProducer::start() {
    running_ = true;
    producer_thread_ = std::thread(&CameraProducer::producerLoop, this);
}

void CameraProducer::stop() {
    running_ = false;
    if (producer_thread_.joinable()) {
        producer_thread_.join();
    }
    cleanup();
}

float CameraProducer::get_fps() {
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_fps_check_time_).count() / 1000.0f;

    uint64_t total = frames_pushed_total_.load();
    uint64_t delta = total - last_frames_pushed_count_;

    float fps = 0.0f;
    if (elapsed > 0.0001f) {
        fps = delta / elapsed;
    }

    last_frames_pushed_count_ = total;
    last_fps_check_time_ = now;
    return fps;
} 

void CameraProducer::producerLoop() {
    int frame_id = 0;

    // 初始化 FPS 计时点
    last_fps_check_time_ = std::chrono::steady_clock::now();

    while (running_) {
        auto t0 = std::chrono::steady_clock::now();
        // 采集一帧
        cv::Mat frame = captureFrame();
        auto t1 = std::chrono::steady_clock::now();

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        int seen = frames_seen_.fetch_add(1) + 1;
        if (seen <= Config::SKIP_INITIAL_FRAMES) {
            skipped_frames_.fetch_add(1);
            // 不推送、不统计这些前导帧
            continue;
        }

        // 记录获取+解码时间（毫秒）
        uint64_t dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total_capture_decode_time_ms_.fetch_add(dt_ms);

        // 推送到环形缓冲区
        FrameData frame_data(std::move(frame), frame_id++);
        ring_buffer_.push_nonblock(std::move(frame_data));
        frames_pushed_.fetch_add(1);
        frames_pushed_total_.fetch_add(1);

        // 控制帧率（60FPS ≈ 16.67ms/帧）
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
} 

cv::Mat CameraProducer::captureFrame() {
    fd_set fds;
    struct timeval tv;
    struct v4l2_buffer buf;
    
    FD_ZERO(&fds);
    FD_SET(cam_fd_, &fds);
    tv.tv_sec = 0;
    tv.tv_usec = 10000;  // 10ms超时
    
    int r = select(cam_fd_ + 1, &fds, NULL, NULL, &tv);
    if (r <= 0) return cv::Mat();
    
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(cam_fd_, VIDIOC_DQBUF, &buf) == -1) {
        return cv::Mat();
    }
    
    // 解码MJPG
    cv::Mat rgb_frame(Config::CAM_HEIGHT, Config::CAM_WIDTH, CV_8UC3);
    if (decodeMJPG((uint8_t*)buffers_[buf.index].start, 
                  buf.bytesused, rgb_frame) != 0) {
        // 重新入队缓冲区
        ioctl(cam_fd_, VIDIOC_QBUF, &buf);
        return cv::Mat();
    }
    
    // 重新入队缓冲区
    ioctl(cam_fd_, VIDIOC_QBUF, &buf);
    return rgb_frame;
}

int CameraProducer::decodeMJPG(const uint8_t* mjpeg_data, size_t size, cv::Mat& rgb_mat) {
    int width, height, subsamp, colorspace;
    
    if (tjDecompressHeader3(tj_handle_, mjpeg_data, size, 
                           &width, &height, &subsamp, &colorspace) != 0) {
        return -1;
    }
    
    if (width != Config::CAM_WIDTH || height != Config::CAM_HEIGHT) {
        return -1;
    }
    
    int flags = TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE;
    if (tjDecompress2(tj_handle_, mjpeg_data, size, 
                     rgb_mat.data, width, 0, height,
                     TJPF_RGB, flags) != 0) {
        return -1;
    }
    
    return 0;
}

void CameraProducer::cleanup() {
    if (cam_fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(cam_fd_, VIDIOC_STREAMOFF, &type);
        
        if (buffers_) {
            for (int i = 0; i < buffer_count_; i++) {
                if (buffers_[i].start) {
                    munmap(buffers_[i].start, buffers_[i].length);
                }
            }
            delete[] buffers_;
            buffers_ = nullptr;
        }
        
        close(cam_fd_);
        cam_fd_ = -1;
    }
    
    if (tj_handle_) {
        tjDestroy(tj_handle_);
        tj_handle_ = nullptr;
    }
}