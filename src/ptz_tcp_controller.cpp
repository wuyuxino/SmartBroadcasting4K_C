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
    // 简短打印：仅在 verbose 模式下打印排队日志，避免高频产生大量日志
    if (debug_ && verbose_debug_) std::cout << "PTZ: queued pan=" << pan_deg << " tilt=" << tilt_deg << " zoom=" << zoom << " ts=" << latest_cmd_.ts_ms << std::endl;
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

// Helper: send raw bytes and try to receive a response (short blocking, returns hex string and vector of hex byte strings)
static std::pair<std::string, std::vector<std::string>> send_and_recv_raw(int sockfd, const unsigned char* buf, size_t len, int timeout_ms) {
    std::pair<std::string, std::vector<std::string>> empty{"", {}};
#ifdef _WIN32
    int rc = send(sockfd, reinterpret_cast<const char*>(buf), (int)len, 0);
    if (rc <= 0) return empty;
#else
    ssize_t rc = ::send(sockfd, buf, len, 0);
    if (rc <= 0) return empty;
#endif
    // set recv timeout
#ifdef _WIN32
    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    unsigned char rbuf[32];
    ssize_t rlen = ::recv(sockfd, (char*)rbuf, sizeof(rbuf), 0);
    if (rlen <= 0) return empty;
    std::string hexstr;
    std::vector<std::string> list;
    for (ssize_t i = 0; i < rlen; ++i) {
        char tmp[8];
        sprintf(tmp, "%02X", rbuf[i]);
        if (!hexstr.empty()) hexstr += " ";
        hexstr += tmp;
        char tmp2[8];
        sprintf(tmp2, "0x%02X", rbuf[i]);
        list.emplace_back(tmp2);
    }
    return {hexstr, list};
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

        // ensure socket is in blocking mode for simple send/recv; any recv used for queries will set short timeout locally
        // (no-op on most platforms)


        // 组装命令：0x81 0x01 0x06 0x02 speed speed h0 h1 h2 h3 v0 v1 v2 v3 0xFF
        unsigned char hbytes[4];
        unsigned char vbytes[4];
        angle_to_bytes(cmd.pan_deg, hbytes);
        angle_to_bytes(cmd.tilt_deg, vbytes);

        // VISCA Absolute Position command must include 4 pan bytes and 4 tilt bytes (total 15 bytes: header(4)+speeds(2)+pan(4)+tilt(4)+term(1))
        unsigned char buf[15];
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
        buf[12] = vbytes[2];
        buf[13] = vbytes[3];
        buf[14] = 0xFF;

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
        if (debug_) {
            // 简短的发送确认（每次打印），便于追踪发送频率和成功计数
            std::cout << "PTZ: sent pan=" << cmd.pan_deg << " tilt=" << cmd.tilt_deg << " total_sent=" << sent_count_.load() << std::endl;
            // 仅在 verbose 模式下输出完整原始字节，便于抓包比对
            if (verbose_debug_) {
                std::cout << "PTZ: raw pan bytes: ";
                for (size_t i = 0; i < sizeof(buf); ++i) std::cout << std::hex << std::uppercase << (int)buf[i] << " ";
                std::cout << std::dec << std::nouppercase << std::endl;
            }
        }

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

        // 只有在 verbose 模式或 zoom_raw 非 0 时打印变焦详情，减少无意义日志
        if (debug_ && (verbose_debug_ || zoom_raw != 0)) {
            if (verbose_debug_) {
                std::cout << "PTZ: sent zoom raw=0x" << std::hex << zoom_raw << std::dec << std::endl;
                std::cout << "PTZ: raw zoom bytes: ";
                for (size_t i = 0; i < sizeof(zcmd); ++i) std::cout << std::hex << std::uppercase << (int)zcmd[i] << " ";
                std::cout << std::dec << std::nouppercase << std::endl;
            } else {
                std::cout << "PTZ: sent zoom raw=0x" << std::hex << zoom_raw << std::dec << std::endl;
            }
        }

        // 小延迟以便控制频率（避免 tight-loop）
        std::this_thread::sleep_for(std::chrono::milliseconds(send_interval_ms_));
    }
}

// 将 zoom 原始位置（0..ZOOM_RAW_MAX）映射为倍数（1..20），与 Python 的 zoom_pos_to_multiple 对齐
static double zoom_pos_to_multiple(int zoom_pos) {
    const double ZOOM_MIN_MULTIPLE = 1.0;
    const double ZOOM_MAX_MULTIPLE = 20.0;
    const int ZOOM_RAW_MIN = 0;
    const int ZOOM_RAW_MAX = 16384;
    if (zoom_pos < ZOOM_RAW_MIN) zoom_pos = ZOOM_RAW_MIN;
    if (zoom_pos > ZOOM_RAW_MAX) zoom_pos = ZOOM_RAW_MAX;
    double mul = ZOOM_MIN_MULTIPLE + (double)(zoom_pos - ZOOM_RAW_MIN) * (ZOOM_MAX_MULTIPLE - ZOOM_MIN_MULTIPLE) / (double)(ZOOM_RAW_MAX - ZOOM_RAW_MIN);
    return std::round(mul * 100.0) / 100.0; // 两位小数
}

std::optional<std::tuple<double,double,double>> PTZTcpController::queryPosition(bool debug) {
    if (!ensureConnected()) return std::nullopt;

    // Build pan/tilt query: 0x81 0x09 0x06 0x12 0xFF
    unsigned char ptq[] = {0x81, 0x09, 0x06, 0x12, 0xFF};
    std::string hexstr;
    std::vector<std::string> list;
    {
        std::lock_guard<std::mutex> sl(sock_mu_);
#ifdef _WIN32
        auto r = send_and_recv_raw(sockfd_, ptq, sizeof(ptq), query_timeout_ms_);
#else
        auto r = send_and_recv_raw(sockfd_, ptq, sizeof(ptq), query_timeout_ms_);
#endif
        hexstr = r.first;
        list = r.second;
    }

    double pan_angle = NAN, tilt_angle = NAN, zoom_multi = NAN;

    if (!hexstr.empty() && list.size() >= 11) {
        // Expecting response starting with 0x90 0x50
        // parse low4 for p,q,r,s
        auto get_low = [&](size_t idx)->int{
            if (idx >= list.size()) return 0;
            std::string s = list[idx]; // like "0x00"
            int v = std::stoi(s.substr(2), nullptr, 16);
            return v & 0x0F;
        };
        int p = get_low(2), q = get_low(3), r = get_low(4), s = get_low(5);
        char pan_hex_buf[5];
        sprintf(pan_hex_buf, "%01x%01x%01x%01x", p, q, r, s);
        int pan_val = std::stoi(std::string(pan_hex_buf), nullptr, 16);
        if (pan_val > 0x7FFF) pan_val -= 0x10000;
        pan_angle = pan_val * 0.075;
        pan_angle = std::max(-171.0, std::min(171.0, pan_angle));

        int t = get_low(6), u = get_low(7), v = get_low(8), w = get_low(9);
        char tilt_hex_buf[5];
        sprintf(tilt_hex_buf, "%01x%01x%01x%01x", t, u, v, w);
        int tilt_val = std::stoi(std::string(tilt_hex_buf), nullptr, 16);
        if (tilt_val > 0x7FFF) tilt_val -= 0x10000;
        tilt_angle = tilt_val * 0.075;
        tilt_angle = std::max(-30.0, std::min(120.0, tilt_angle));

        if (debug && verbose_debug_) {
            std::cout << "PTZ QUERY PT response: " << hexstr << std::endl;
            std::cout << "    parsed pan=" << pan_angle << " tilt=" << tilt_angle << std::endl;
        }
    } else if (verbose_debug_) {
        std::cout << "PTZ: no valid PT response (hex=" << hexstr << ")" << std::endl;
    }

    // Zoom query: 0x81 0x09 0x04 0x47 0xFF
    unsigned char zq[] = {0x81, 0x09, 0x04, 0x47, 0xFF};
    std::string zhex;
    std::vector<std::string> zlist;
    {
        std::lock_guard<std::mutex> sl(sock_mu_);
        auto r = send_and_recv_raw(sockfd_, zq, sizeof(zq), query_timeout_ms_);
        zhex = r.first;
        zlist = r.second;
    }
    if (!zhex.empty() && zlist.size() >= 7) {
        auto get_low = [&](size_t idx)->int{
            if (idx >= zlist.size()) return 0;
            std::string s = zlist[idx];
            int v = std::stoi(s.substr(2), nullptr, 16);
            return v & 0x0F;
        };
        int a = get_low(2), b = get_low(3), c = get_low(4), d = get_low(5);
        char zhexbuf[5];
        sprintf(zhexbuf, "%01x%01x%01x%01x", a, b, c, d);
        int zoom_pos = std::stoi(std::string(zhexbuf), nullptr, 16);
        zoom_multi = zoom_pos_to_multiple(zoom_pos);
        if (debug) {
            std::cout << "PTZ QUERY ZOOM response: " << zhex << std::endl;
            std::cout << "    parsed zoom_pos=" << zoom_pos << " zoom_mult=" << zoom_multi << std::endl;
        }
    } else if (debug) {
        std::cout << "PTZ: no valid Zoom response (hex=" << zhex << ")" << std::endl;
    }

    if (!std::isnan(pan_angle) || !std::isnan(tilt_angle) || !std::isnan(zoom_multi)) {
        double p = std::isnan(pan_angle) ? 0.0 : pan_angle;
        double t = std::isnan(tilt_angle) ? 0.0 : tilt_angle;
        double z = std::isnan(zoom_multi) ? 1.0 : zoom_multi;
        return std::optional<std::tuple<double,double,double>>(std::make_tuple(p,t,z));
    }
    return std::nullopt;
}
