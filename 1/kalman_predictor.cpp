#include <opencv2/video/tracking.hpp>

class KalmanPredictor : public TrajectoryPredictor {
private:
    cv::KalmanFilter kf;
    cv::Mat measurement;
    bool initialized;
    float prediction_horizon;  // 预测时间（秒）
    
public:
    KalmanPredictor(float horizon = 0.1f) 
        : prediction_horizon(horizon), initialized(false) {
        
        // 4维状态: [x, y, vx, vy]
        // 2维测量: [x, y]
        kf.init(4, 2, 0);
        
        // 转移矩阵 (匀速模型)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        
        // 测量矩阵
        cv::setIdentity(kf.measurementMatrix);
        
        // 过程噪声协方差
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
        
        // 测量噪声协方差
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        
        // 后验误差协方差
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.1));
        
        measurement = cv::Mat::zeros(2, 1, CV_32F);
    }
    
    PredictionResult predict(const std::vector<DetectionResult>& detections) override {
        if (detections.empty()) {
            return PredictionResult(0, 0, 0);
        }
        
        // 获取最新检测的目标中心（假设只有一个目标）
        const auto& latest = detections.back();
        if (latest.boxes.empty()) {
            return PredictionResult(0, 0, 0);
        }
        
        // 取第一个检测框的中心
        const auto& box = latest.boxes[0];
        float center_x = (box.x1 + box.x2) / 2.0f;
        float center_y = (box.y1 + box.y2) / 2.0f;
        
        // 更新卡尔曼滤波器
        measurement.at<float>(0) = center_x;
        measurement.at<float>(1) = center_y;
        
        if (!initialized) {
            kf.statePost.at<float>(0) = center_x;
            kf.statePost.at<float>(1) = center_y;
            kf.statePost.at<float>(2) = 0;
            kf.statePost.at<float>(3) = 0;
            initialized = true;
        } else {
            kf.predict();
            kf.correct(measurement);
        }
        
        // 预测未来位置
        cv::Mat prediction = kf.predict();
        
        // 应用预测时间
        float pred_x = prediction.at<float>(0) + prediction.at<float>(2) * prediction_horizon;
        float pred_y = prediction.at<float>(1) + prediction.at<float>(3) * prediction_horizon;
        
        // 计算置信度（基于检测置信度和速度稳定性）
        float confidence = box.confidence;
        float speed = sqrt(pow(prediction.at<float>(2), 2) + pow(prediction.at<float>(3), 2));
        
        // 速度越稳定，置信度越高
        if (speed < 100) {  // 像素/秒
            confidence *= 1.2f;
        }
        
        return PredictionResult(pred_x, pred_y, confidence);
    }
    
    void setPredictionHorizon(float seconds) override {
        prediction_horizon = seconds;
    }
    
    void setSmoothingFactor(float factor) override {
        // 调整过程噪声协方差
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(factor));
    }
};