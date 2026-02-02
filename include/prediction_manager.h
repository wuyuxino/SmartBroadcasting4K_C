#pragma once

#include "detection_queue.h"
#include "ptz_controller.h"
// 前向声明 KalmanPredictor，避免在缺少外部库时强依赖头文件
class KalmanPredictor;
#include "common.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <functional>
#include <cctype>
#include <memory>



class PredictionManager {
public:
    // sum_move_thresh: 累积位移阈值（像素），默认 8.0
    // prediction_horizon: index of predicted frame to use (0 = next frame, 2 = third frame)
    PredictionManager(DetectionResultQueue& q,
                      IPTZController* ptz,
                      const std::string& kalman_json,
                      const std::string& norm_json,
                      double sum_move_thresh = 8.0,
                      int prediction_horizon = 2);
    ~PredictionManager();

    void set_prediction_horizon(int h) { prediction_horizon_ = h; }
    int get_prediction_horizon() const { return prediction_horizon_; }

    void start();
    void stop();

    // 用于单元测试 / 调试，公开对外检查函数
    bool debug_hasSignificantChange(const std::vector<std::vector<DetectionBox>>& frames) {
        return hasSignificantChange(frames);
    }

    void set_sum_move_thresh(double t) { sum_move_thresh_ = t; }
    double get_sum_move_thresh() const { return sum_move_thresh_; }

    // 预测时间统计（ms）
    double get_avg_prediction_time_ms() const;

private:
    void loop();
    bool hasSignificantChange(const std::vector<std::vector<DetectionBox>>& frames);
    static cv::Point2f centroid(const DetectionBox& b);
    static bool isBall(const std::string& s);

private:
    DetectionResultQueue& queue_;
    IPTZController* ptz_;
    std::unique_ptr<KalmanPredictor> predictor_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::chrono::steady_clock::time_point last_fire_;
    std::mutex mutex_;

    static constexpr int WINDOW = 5;  // 最近多少帧用于“累积位移”计算
    static constexpr int CHECK_MS = 16; // 预测器的轮询间隔（毫秒）
    static constexpr int COOLDOWN_MS = 16;   // 预测器触发PTZ控制的冷却时间（毫秒）
    static constexpr float MOVE_THRESH_PX = 30.0f; // 保留旧阈值（按需）

    double sum_move_thresh_ = 40.0; // 默认累积位移阈值

    int prediction_horizon_ = 2; // 默认使用第三帧（index=2）

    // 预测时间统计（以微秒累计以提高精度）
    std::atomic<uint64_t> total_prediction_time_us_{0};
    std::atomic<int> prediction_count_{0};
};