#!/bin/bash

echo "🚀 构建完整流水线系统..."

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib:/usr/local/lib:$LD_LIBRARY_PATH

# 编译
g++ -std=c++17 -O3 -pthread \
    main_pipeline.cpp \
    yolov8_detector.cpp \
    kalman_predictor.cpp \
    serial_gimbal.cpp \
    -o tracking_system \
    -I/usr/local/cuda-11.7/include \
    -I/C/onnx/tensorrt86/TensorRT-8.6.1.6/include \
    -I/usr/local/include/opencv4 \
    -L/usr/local/cuda-11.7/lib64 \
    -L/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib \
    -L/usr/local/lib \
    -lnvinfer -lnvonnxparser -lcudart \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video \
    -lopencv_imgcodecs -lturbojpeg -lpthread -lm

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 构建成功！"
    echo ""
    echo "📊 系统架构："
    echo "   线程1: 采集 (60 FPS)"
    echo "   线程2: 检测 (45 FPS)"
    echo "   线程3: 预测 (9 FPS)"
    echo "   线程4: 控制 (实时)"
    echo ""
    echo "🚀 运行命令："
    echo "   export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib:/usr/local/lib:\$LD_LIBRARY_PATH"
    echo "   ./tracking_system"
else
    echo "❌ 构建失败"
    exit 1
fi