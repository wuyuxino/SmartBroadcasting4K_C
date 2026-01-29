// YOLOv8检测器接口
class YOLOv8Detector {
public:
    virtual ~YOLOv8Detector() = default;
    
    // 检测一帧图像
    virtual std::vector<DetectionBox> detect(const cv::Mat& frame) = 0;
    
    // 获取检测时间统计
    virtual float getAverageInferenceTime() const = 0;
};

// 轨迹预测器接口
class TrajectoryPredictor {
public:
    virtual ~TrajectoryPredictor() = default;
    
    // 基于历史检测结果预测目标位置
    virtual PredictionResult predict(const std::vector<DetectionResult>& detections) = 0;
    
    // 设置预测参数
    virtual void setPredictionHorizon(float seconds) = 0;
    virtual void setSmoothingFactor(float factor) = 0;
};

// 云台控制器接口
struct GimbalCommand {
    float pitch;    // 俯仰角（度）
    float yaw;      // 方位角（度）
    float speed;    // 转动速度（度/秒）
    bool fire;      // 是否发射（如果适用）
    
    GimbalCommand() : pitch(0), yaw(0), speed(10), fire(false) {}
};

class GimbalController {
public:
    virtual ~GimbalController() = default;
    
    // 根据预测结果计算控制指令
    virtual GimbalCommand calculateCommand(const PredictionResult& prediction) = 0;
    
    // 发送控制指令到云台
    virtual bool sendCommand(const GimbalCommand& command) = 0;
    
    // 连接/断开云台
    virtual bool connect(const std::string& device) = 0;
    virtual void disconnect() = 0;
};