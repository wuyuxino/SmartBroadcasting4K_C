#include "ptz_tcp_controller.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
    using socklen_t = int;
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netinet/tcp.h>
    #include <fcntl.h>
#endif

static unsigned long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// 小工具：角度到4字节（VISCA格式低4位有效）
static void angle_to_bytes(double angle_deg, unsigned char out[4]) {
    const double ANGLE_COEFF = 0.075; // 与 python 版本保持一致
    double target_value = angle_deg / ANGLE_COEFF;
    if (target_value < 0) {
        target_value = 0x10000 + target_value;
    }
    int tv = static_cast<int>(std::llround(target_value));
    unsigned int p = (tv >> 12) & 0x0F;
    unsigned int q = (tv >> 8) & 0x0F;
    unsigned int r = (tv >> 4) & 0x0F;
    unsigned int s = tv & 0x0F;
    out[0] = static_cast<unsigned char>(0x00 | p);
    out[1] = static_cast<unsigned char>(0x00 | q);
    out[2] = static_cast<unsigned char>(0x00 | r);
    out[3] = static_cast<unsigned char>(0x00 | s);
}

// Zoom mapping：参考 python 代码（线性映射 raw）
static int zoom_multi_to_raw(double zoom_multi) {
    const double ZOOM_RANGE_MIN = 1.0;
    const double ZOOM_RANGE_MAX = 20.0;
    const int ZOOM_RAW_MAX = 16384;
    zoom_multi = std::max(ZOOM_RANGE_MIN, std::min(ZOOM_RANGE_MAX, zoom_multi));
    double zoom_raw = (zoom_multi - 1.0) * (ZOOM_RAW_MAX / (ZOOM_RANGE_MAX - ZOOM_RANGE_MIN));
    return static_cast<int>(std::llround(zoom_raw));
}

PTZTcpController::PTZTcpController(const std::string& ip, uint16_t port,
                                   int control_timeout_ms, int query_timeout_ms,
                                   int send_interval_ms, bool debug)
    : ip_(ip), port_(port), control_timeout_ms_(control_timeout_ms),
      query_timeout_ms_(query_timeout_ms), send_interval_ms_(send_interval_ms), debug_(debug) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
    running_ = true;
    worker_ = std::thread(&PTZTcpController::run, this);
}

PTZTcpController::~PTZTcpController() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

void PTZTcpController::stop() {
    if (!running_) return;
    running_ = false;
    cmd_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    closeSocket();
}

void PTZTcpController::setSendIntervalMs(int ms) {
    send_interval_ms_ = ms;
}

void PTZTcpController::sendPanTilt(double pan_deg, double tilt_deg, double zoom) {
    std::lock_guard<std::mutex> lk(cmd_mu_);
    latest_cmd_.pan_deg = pan_deg;
    latest_cmd_.tilt_deg = tilt_deg;
    latest_cmd_.zoom = zoom;
    latest_cmd_.ts_ms = now_ms();
    latest_cmd_.speed = 0x18;
    has_new_cmd_ = true;
    cmd_cv_.notify_one();
    if (debug_) std::cout << "PTZ: queued pan=" << pan_deg << " tilt=" << tilt_deg << " zoom=" << zoom << " ts=" << latest_cmd_.ts_ms << std::endl;
}

bool PTZTcpController::ensureConnected() {
    std::lock_guard<std::mutex> lk(sock_mu_);
    if (sockfd_ >= 0) return true;

    // 创建socket
    int s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s < 0) {
        if (debug_) std::cerr << "PTZ: socket() failed" << std::endl;
        return false;
    }

    // 设置 TCP_NODELAY
    int flag = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));

    // 设置超时
#ifdef _WIN32
    DWORD timeout_ms = static_cast<DWORD>(control_timeout_ms_);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv;
    tv.tv_sec = control_timeout_ms_ / 1000;
    tv.tv_usec = (control_timeout_ms_ % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = inet_addr(ip_.c_str());

    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        if (debug_) std::cerr << "PTZ: connect() failed to " << ip_ << ":" << port_ << std::endl;
#ifdef _WIN32
        closesocket(s);
#else
        close(s);
#endif
        return false;
    }

    sockfd_ = s;
    if (debug_) std::cout << "✅ PTZ TCP connected: " << ip_ << ":" << port_ << std::endl;
    return true;
}

void PTZTcpController::closeSocket() {
    std::lock_guard<std::mutex> lk(sock_mu_);
    if (sockfd_ >= 0) {
#ifdef _WIN32
        closesocket(sockfd_);
#else
        close(sockfd_);
#endif
        sockfd_ = -1;
        if (debug_) std::cout << "PTZ: socket closed" << std::endl;
    }
}

void PTZTcpController::run() {
    const int reconnect_backoff_ms = 500; // 0.5s
    while (running_) {
        // 等待命令或超时唤醒以便保持周期发送
        std::unique_lock<std::mutex> lk(cmd_mu_);
        if (!has_new_cmd_) {
            cmd_cv_.wait_for(lk, std::chrono::milliseconds(send_interval_ms_));
        }

        if (!running_) break;

        // 如果没有命令，继续循环（保持低 CPU）
        if (!has_new_cmd_) continue;

        // 取走最新命令
        PTZCommand cmd = latest_cmd_;
        has_new_cmd_ = false;
        lk.unlock();

        // 确保连接
        if (!ensureConnected()) {
            // 失败则睡眠再重连
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_backoff_ms));
            continue;
        }

        // 组装命令：0x81 0x01 0x06 0x02 speed speed h0 h1 h2 h3 v0 v1 v2 v3 0xFF
        unsigned char hbytes[4];
        unsigned char vbytes[4];
        angle_to_bytes(cmd.pan_deg, hbytes);
        angle_to_bytes(cmd.tilt_deg, vbytes);

        unsigned char buf[13];
        buf[0] = 0x81;
        buf[1] = 0x01;
        buf[2] = 0x06;
        buf[3] = 0x02;
        buf[4] = static_cast<unsigned char>(cmd.speed);
        buf[5] = static_cast<unsigned char>(cmd.speed);
        buf[6] = hbytes[0];
        buf[7] = hbytes[1];
        buf[8] = hbytes[2];
        buf[9] = hbytes[3];
        buf[10] = vbytes[0];
        buf[11] = vbytes[1];
        buf[12] = 0xFF;

        // 发送
        ssize_t bytes_sent = 0;
        {
            std::lock_guard<std::mutex> sl(sock_mu_);
            if (sockfd_ >= 0) {
#ifdef _WIN32
                int rc = send(sockfd_, reinterpret_cast<const char*>(buf), sizeof(buf), 0);
                bytes_sent = rc;
#else
                bytes_sent = ::send(sockfd_, buf, sizeof(buf), 0);
#endif
            }
        }

        if (bytes_sent <= 0) {
            if (debug_) std::cerr << "PTZ: send failed, closing socket and will reconnect" << std::endl;
            closeSocket();
            continue;
        }

        // 记录发送成功
        sent_count_.fetch_add(1);
        if (debug_) std::cout << "PTZ: sent pan=" << cmd.pan_deg << " tilt=" << cmd.tilt_deg << " ts=" << cmd.ts_ms << " total_sent=" << sent_count_.load() << std::endl;

        // 变焦：如果需要较复杂的格式（Direct Zoom），可以发送单独的指令
        // 这里我们以 Direct Zoom 为例，和 Python 一致
        int zoom_raw = zoom_multi_to_raw(cmd.zoom);
        unsigned char zoom_b1 = static_cast<unsigned char>((zoom_raw >> 12) & 0x0F);
        unsigned char zoom_b2 = static_cast<unsigned char>((zoom_raw >> 8) & 0x0F);
        unsigned char zoom_b3 = static_cast<unsigned char>((zoom_raw >> 4) & 0x0F);
        unsigned char zoom_b4 = static_cast<unsigned char>(zoom_raw & 0x0F);
        unsigned char zcmd[9];
        zcmd[0] = 0x81; zcmd[1] = 0x01; zcmd[2] = 0x04; zcmd[3] = 0x47;
        zcmd[4] = zoom_b1; zcmd[5] = zoom_b2; zcmd[6] = zoom_b3; zcmd[7] = zoom_b4; zcmd[8] = 0xFF;

        {
            std::lock_guard<std::mutex> sl(sock_mu_);
            if (sockfd_ >= 0) {
#ifdef _WIN32
                int rc = send(sockfd_, reinterpret_cast<const char*>(zcmd), sizeof(zcmd), 0);
                (void)rc;
#else
                ::send(sockfd_, zcmd, sizeof(zcmd), 0);
#endif
            }
        }

        if (debug_) std::cout << "PTZ: sent zoom raw=0x" << std::hex << zoom_raw << std::dec << std::endl;

        // 小延迟以便控制频率（避免 tight-loop）
        std::this_thread::sleep_for(std::chrono::milliseconds(send_interval_ms_));
    }
}
