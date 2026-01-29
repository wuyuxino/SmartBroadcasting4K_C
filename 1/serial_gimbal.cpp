#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

class SerialGimbalController : public GimbalController {
private:
    int serial_fd;
    bool connected;
    
public:
    SerialGimbalController() : serial_fd(-1), connected(false) {}
    
    bool connect(const std::string& device) override {
        serial_fd = open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (serial_fd < 0) {
            perror("打开串口失败");
            return false;
        }
        
        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        
        if (tcgetattr(serial_fd, &tty) != 0) {
            perror("获取串口属性失败");
            close(serial_fd);
            return false;
        }
        
        // 设置波特率
        cfsetospeed(&tty, B115200);
        cfsetispeed(&tty, B115200);
        
        // 8N1
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        
        // 无流控
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;
        
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_oflag &= ~OPOST;
        
        // 设置超时
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;
        
        if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
            perror("设置串口属性失败");
            close(serial_fd);
            return false;
        }
        
        connected = true;
        std::cout << "✅ 云台连接成功: " << device << std::endl;
        return true;
    }
    
    void disconnect() override {
        if (connected) {
            close(serial_fd);
            connected = false;
            std::cout << "📴 云台断开连接" << std::endl;
        }
    }
    
    GimbalCommand calculateCommand(const PredictionResult& prediction) override {
        GimbalCommand cmd;
        
        // 将图像坐标转换为云台角度
        // 假设图像中心对应云台中心
        // 这里需要根据你的摄像头和云台标定参数调整
        
        // 简化示例：假设图像分辨率3840x2160，云台范围±30度
        float center_x = 1920;  // 图像中心X
        float center_y = 1080;  // 图像中心Y
        
        // 计算偏移量（像素）
        float dx = prediction.predicted_x - center_x;
        float dy = prediction.predicted_y - center_y;
        
        // 转换为角度（假设100像素 = 1度）
        cmd.yaw = -dx / 100.0f;    // 左右偏移控制方位角
        cmd.pitch = -dy / 100.0f;  // 上下偏移控制俯仰角
        
        // 限制角度范围
        cmd.yaw = std::max(-30.0f, std::min(30.0f, cmd.yaw));
        cmd.pitch = std::max(-30.0f, std::min(30.0f, cmd.pitch));
        
        // 根据预测置信度调整速度
        cmd.speed = 10.0f + prediction.confidence * 10.0f;
        
        return cmd;
    }
    
    bool sendCommand(const GimbalCommand& command) override {
        if (!connected) {
            std::cerr << "❌ 云台未连接" << std::endl;
            return false;
        }
        
        // 构造控制指令（示例：自定义协议）
        char buffer[32];
        int len = snprintf(buffer, sizeof(buffer), "P%.1f Y%.1f S%.1f\n",
                          command.pitch, command.yaw, command.speed);
        
        // 发送指令
        ssize_t written = write(serial_fd, buffer, len);
        
        if (written != len) {
            std::cerr << "❌ 发送云台指令失败" << std::endl;
            return false;
        }
        
        // 可选：读取响应
        char response[64];
        usleep(10000);  // 等待10ms
        int n = read(serial_fd, response, sizeof(response) - 1);
        
        if (n > 0) {
            response[n] = '\0';
            // 解析响应...
        }
        
        return true;
    }
    
    ~SerialGimbalController() {
        disconnect();
    }
};