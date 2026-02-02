#pragma once

#include "ptz_controller.h"
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

// 简单命令结构：合并高频命令仅保留最新一条
struct PTZCommand {
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
    double zoom = 1.0;
    int speed = 0x18;
    unsigned long long ts_ms = 0;
};

class PTZTcpController : public IPTZController {
public:
    // ip: camera ip, port: visca-over-tcp port (通常5678)
    // control_timeout_ms / query_timeout_ms: socket 超时
    // send_interval_ms: 工作线程发送频率（ms），高频场景可调小到 5-10ms
    PTZTcpController(const std::string& ip, uint16_t port,
                     int control_timeout_ms = 100, int query_timeout_ms = 500,
                     int send_interval_ms = 10, bool debug = true);
    ~PTZTcpController() override;

    // 异步高频安全接口：仅将命令合并入最新命令并通知发送线程
    void sendPanTilt(double pan_deg, double tilt_deg, double zoom) override;

    // 优雅停止（析构会调用）
    void stop();

    // 可选：调整发送间隔
    void setSendIntervalMs(int ms);

    // 调试与状态接口
    void setDebug(bool d) { debug_ = d; }
    uint64_t getSentCount() const { return sent_count_.load(); }

private:
    void run();
    bool ensureConnected();
    void closeSocket();

    std::string ip_;
    uint16_t port_;
    int control_timeout_ms_;
    int query_timeout_ms_;
    int send_interval_ms_;
    bool debug_;

    // 命令合并
    std::mutex cmd_mu_;
    std::condition_variable cmd_cv_;
    PTZCommand latest_cmd_;
    bool has_new_cmd_ = false;

    // 线程管理
    std::thread worker_;
    std::atomic<bool> running_{false};

    // 发送计数（用于外部查询，确认发送成功）
    std::atomic<uint64_t> sent_count_{0};

    // 平台相关 socket（实现文件中定义类型别名）
    int sockfd_ = -1;
    std::mutex sock_mu_;
};