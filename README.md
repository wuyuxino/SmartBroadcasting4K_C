#### 文件夹说明
```txt
文件结构
/
├── include/
│   ├── ring_buffer.h
│   ├── detection_queue.h
│   └── common.h
├── src/
│   ├── camera_producer.cpp
│   ├── detection_consumer.cpp
│   ├── inference_engine.cpp
│   ├── cuda_kernels.cu
│   └── main.cpp
├── build/
│   ├── best.engine
│   └── yolov8_system
├── CMakeLists.txt
└── README.md

系统架构设计如下

摄像头(60FPS, 4K)
    ↓ (生产)
[环形缓冲区] ← 实时填充，保留最新3帧
    ↓ (消费者1)
检测线程(45FPS) → [检测结果队列(长度5)] ← 保留最新5个检测结果
    ↓ (消费者2)      ↓ (触发条件: 队列满)
预测线程(9FPS)   → [预测结果]
    ↓ (消费者3)
控制线程(9FPS)   → 发送云台指令
```

#### 编译
```bash
mkdir build && cd build
cmake ..
make -j4
```

#### 运行
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib:/usr/local/lib:$LD_LIBRARY_PATH
./yolov8_system
```

#### 上传电脑目录
```bash
scp -r ./* user@192.168.31.149:/C/ONNX/cuda_onnx/finall

```