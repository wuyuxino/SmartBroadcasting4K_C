#ifndef KALMAN_PREDICTOR_H
#define KALMAN_PREDICTOR_H

#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>  // 用于解析json配置文件（轻量级json库）
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// 简化json命名空间
using json = nlohmann::json;

// 坐标结构体（输入输出均使用，简单直观）
struct FrameCoord {
    int frame_id;
    double x1;
    double y1;

    FrameCoord(int fid = 0, double x = 0.0, double y = 0.0)
        : frame_id(fid), x1(x), y1(y) {}
};

// 卡尔曼预测器类（封装所有逻辑，预加载参数，对外提供简单调用接口）
class KalmanPredictor {
public:
    // 构造函数：加载模型参数（Q/R）和归一化统计量
    KalmanPredictor(const std::string& model_path, const std::string& stats_path) {
        // 初始化固定矩阵（F、H、初始P）
        initFixedMatrices();

        // 加载归一化统计量
        loadNormStats(stats_path);

        // 加载卡尔曼参数（Q、R，从pth文件提取，注：原Python的pth是torch序列化文件，C++直接解析复杂，建议先转换为json格式）
        // 【简化方案】：先从Python提取Q/R的数值，保存为json（下文会提供提取脚本），C++直接解析该json
        loadKalmanParams(model_path);

        // 标记初始化完成
        is_initialized_ = true;
    }

    // 对外核心调用函数：传入历史帧坐标，返回未来3帧预测结果
    std::vector<FrameCoord> predict(const std::vector<FrameCoord>& history_frames) {
        // 检查初始化状态
        if (!is_initialized_) {
            throw std::runtime_error("KalmanPredictor not initialized successfully.");
        }

        // 检查输入数据有效性
        if (history_frames.empty()) {
            throw std::invalid_argument("History frames cannot be empty.");
        }

        // 步骤1：预处理历史数据（排序、取最后5帧、提取坐标）
        auto processed_coords = preprocessHistory(history_frames);
        Eigen::MatrixXd history_coords(processed_coords.size(), 2);
        for (size_t i = 0; i < processed_coords.size(); ++i) {
            history_coords(i, 0) = processed_coords[i].x1;
            history_coords(i, 1) = processed_coords[i].y1;
        }

        // 步骤2：归一化坐标
        auto norm_coords = normalizeCoords(history_coords);

        // 步骤3：卡尔曼滤波核心计算
        auto [current_state, current_velocity] = runKalmanFilter(norm_coords);

        // 步骤4：反归一化，计算未来3帧坐标
        auto future_coords = calculateFutureCoords(current_state, current_velocity);

        // 步骤5：构造返回结果（填充未来帧ID）
        int max_frame_id = std::max_element(history_frames.begin(), history_frames.end(),
            [](const FrameCoord& a, const FrameCoord& b) { return a.frame_id < b.frame_id; })->frame_id;

        std::vector<FrameCoord> result;
        for (size_t i = 0; i < 3; ++i) {
            result.emplace_back(
                max_frame_id + 1 + static_cast<int>(i),
                std::round(future_coords(i, 0) * 100) / 100,  // 保留2位小数，与Python一致
                std::round(future_coords(i, 1) * 100) / 100
            );
        }

        return result;
    }

    // 检查是否初始化成功
    bool isInitialized() const { return is_initialized_; }

private:
    // ------------- 固定参数与成员变量 -------------
    Eigen::Matrix4d F_;  // 状态转移矩阵
    Eigen::Matrix<double, 2, 4> H_;  // 观测矩阵
    Eigen::Matrix4d init_P_;  // 初始协方差矩阵

    Eigen::Matrix4d Q_;  // 过程噪声协方差矩阵
    Eigen::Matrix2d R_;  // 观测噪声协方差矩阵

    double mean_x_ = 0.0, mean_y_ = 0.0;
    double std_x_ = 1.0, std_y_ = 1.0;

    bool is_initialized_ = false;

    // ------------- 辅助函数：初始化固定矩阵 -------------
    void initFixedMatrices() {
        // 初始化F矩阵（与Python一致）
        F_ << 1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0,
              0, 0, 0, 1;

        // 初始化H矩阵（与Python一致）
        H_ << 1, 0, 0, 0,
              0, 1, 0, 0;

        // 初始化初始P矩阵（与Python一致，1000*单位矩阵）
        init_P_ = Eigen::Matrix4d::Identity() * 1000.0;
    }

    // ------------- 辅助函数：加载归一化统计量（json文件） -------------
    void loadNormStats(const std::string& stats_path) {
        std::ifstream stats_file(stats_path);
        if (!stats_file.is_open()) {
            throw std::runtime_error("Cannot open norm stats file: " + stats_path);
        }

        json stats_json;
        stats_file >> stats_json;

        // 提取归一化参数
        mean_x_ = stats_json["mean_x"];
        mean_y_ = stats_json["mean_y"];
        std_x_ = stats_json["std_x"];
        std_y_ = stats_json["std_y"];
    }

    // ------------- 辅助函数：加载卡尔曼参数（Q/R，json格式） -------------
    void loadKalmanParams(const std::string& model_path) {
        std::ifstream model_file(model_path);
        if (!model_file.is_open()) {
            throw std::runtime_error("Cannot open kalman params file: " + model_path);
        }

        json model_json;
        model_file >> model_json;

        // 提取Q矩阵（4x4）
        auto q_data = model_json["Q"];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Q_(i, j) = q_data[i][j];
            }
        }

        // 提取R矩阵（2x2）
        auto r_data = model_json["R"];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                R_(i, j) = r_data[i][j];
            }
        }
    }

    // ------------- 辅助函数：预处理历史数据（排序、取最后5帧） -------------
    std::vector<FrameCoord> preprocessHistory(const std::vector<FrameCoord>& history_frames) {
        // 按frame_id排序
        std::vector<FrameCoord> sorted_frames = history_frames;
        std::sort(sorted_frames.begin(), sorted_frames.end(),
            [](const FrameCoord& a, const FrameCoord& b) { return a.frame_id < b.frame_id; });

        // 取最后5帧（不足则取全部）
        size_t take_count = std::min(sorted_frames.size(), static_cast<size_t>(5));
        std::vector<FrameCoord> result(sorted_frames.end() - take_count, sorted_frames.end());

        return result;
    }

    // ------------- 辅助函数：归一化坐标 -------------
    Eigen::MatrixXd normalizeCoords(const Eigen::MatrixXd& coords) {
        Eigen::MatrixXd norm_coords = coords;
        for (int i = 0; i < coords.rows(); ++i) {
            norm_coords(i, 0) = (coords(i, 0) - mean_x_) / std_x_;
            norm_coords(i, 1) = (coords(i, 1) - mean_y_) / std_y_;
        }
        return norm_coords;
    }

    // ------------- 辅助函数：反归一化坐标 -------------
    Eigen::MatrixXd denormalizeCoords(const Eigen::MatrixXd& norm_coords) {
        Eigen::MatrixXd coords = norm_coords;
        for (int i = 0; i < norm_coords.rows(); ++i) {
            coords(i, 0) = norm_coords(i, 0) * std_x_ + mean_x_;
            coords(i, 1) = norm_coords(i, 1) * std_y_ + mean_y_;
        }
        return coords;
    }

    // ------------- 核心函数：运行卡尔曼滤波 -------------
    std::pair<Eigen::Vector2d, Eigen::Vector2d> runKalmanFilter(const Eigen::MatrixXd& norm_history) {
        // 初始化状态向量X（4x1）和协方差矩阵P
        Eigen::Vector4d X;
        X << norm_history(0, 0), norm_history(0, 1), 0.0, 0.0;

        Eigen::Matrix4d P = init_P_;

        // 遍历历史观测值，更新卡尔曼滤波状态
        for (int i = 0; i < norm_history.rows(); ++i) {
            // 1. 预测步骤
            X = F_ * X;
            P = F_ * P * F_.transpose() + Q_;

            // 2. 更新步骤
            Eigen::Vector2d z;
            z << norm_history(i, 0), norm_history(i, 1);

            Eigen::Matrix2d S = H_ * P * H_.transpose() + R_;
            Eigen::Matrix<double, 4, 2> K = P * H_.transpose() * S.inverse();

            Eigen::Vector2d z_pred = H_ * X;
            X = X + K * (z - z_pred);

            Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
            P = (I - K * H_) * P;
        }

        // 提取当前位置和速度
        Eigen::Vector2d current_state(X(0), X(1));
        Eigen::Vector2d current_velocity(X(2), X(3));

        return { current_state, current_velocity };
    }

    // ------------- 辅助函数：计算未来3帧坐标 -------------
    Eigen::MatrixXd calculateFutureCoords(const Eigen::Vector2d& current_state, const Eigen::Vector2d& current_velocity) {
        // 构建未来3帧的归一化坐标
        Eigen::MatrixXd norm_future(3, 2);
        for (int k = 1; k <= 3; ++k) {
            double pred_x = current_state(0) + k * current_velocity(0);
            double pred_y = current_state(1) + k * current_velocity(1);
            norm_future(k - 1, 0) = pred_x;
            norm_future(k - 1, 1) = pred_y;
        }

        // 反归一化
        return denormalizeCoords(norm_future);
    }
};

#endif // KALMAN_PREDICTOR_H