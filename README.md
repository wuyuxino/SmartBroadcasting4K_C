#### 文件夹说明（目录结构）
```txt
SmartBroadcasting4K_C/
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
├── transformation/
│   └── pth_to_json.py
├── build/
│   ├── best.engine
│   └── yolov8_system
├── CMakeLists.txt
└── README.md
```

#### 需要模型转化（转化代码在transformation目录下）
```bash
# 检测模型 best 格式转化 onnx
$ python ./transformation/pt_to_onnx.py
# 检测模型 onnx 格式转化 engine
$ trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
# 预测模型 pth 格式转化 json
$ python ./transformation/pth_to_json.py
# 以前手动编译程序（可以查看部分软件安装路径，后续直接用下面的编译模块命令进行编译即可。）
$ g++ -std=c++17 yolov8_trt.cpp -o yolov8_trt -Wno-deprecated-declarations -I/usr/local/cuda-11.7/targets/x86_64-linux/include -I/C/onnx/tensorrt86/TensorRT-8.6.1.6/include -I/usr/local/include/opencv4/ -L/usr/local/cuda-11.7/targets/x86_64-linux/lib/ -L/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib -L/usr/local/lib/ -Wl,-rpath=/usr/local/cuda-11.7/targets/x86_64-linux/lib/ -Wl,-rpath=/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib -Wl,-rpath=/usr/local/lib/ -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaarithm -lturbojpeg -lpthread -lm -ldl
```

#### 编译
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j4
```

#### 运行
```bash
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/C/onnx/tensorrt86/TensorRT-8.6.1.6/lib:/usr/local/lib:$LD_LIBRARY_PATH
$ ./yolov8_system
```

#### 上传电脑目录
```bash
$ scp -r ./* user@192.168.31.149:/C/ONNX/cuda_onnx/finall
$ 123qweasd!
```

#### 目前阶段延时时间分布
```txt
4K60帧
摄像头延迟（25ms）+ 获取、解码（8ms）+ 检测（22ms）+ 预测（1ms）+ 云台控制 （3ms）= 总时间（59ms）
备注：云台控制命令发送延迟时间有线或者无线网桥的情况下是1~3ms，通过wifi会不稳定有高有底最高可达50ms延迟；查询命令等待返回会增加延迟10~20ms（以前测试15ms原因，发送控制命令理论是很快的。）
```