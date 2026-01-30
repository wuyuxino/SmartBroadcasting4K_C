// cuda_kernels.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void preprocess_kernel_uint8_to_fp32(
    const uint8_t* src,      // 输入：NHWC布局 [height, width, 3]
    float* dst,              // 输出：NCHW布局 [3, height, width]
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const float scale_x,
    const float scale_y,
    const float norm_factor
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // 计算源图像坐标（双线性插值）
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float wx1 = src_x - x0;
    float wx0 = 1.0f - wx1;
    float wy1 = src_y - y0;
    float wy0 = 1.0f - wy1;

    // 对每个通道进行双线性插值
    for (int c = 0; c < 3; c++) {
        // 获取4个源像素值
        uint8_t v00 = src[(y0 * src_width + x0) * 3 + c];
        uint8_t v01 = src[(y0 * src_width + x1) * 3 + c];
        uint8_t v10 = src[(y1 * src_width + x0) * 3 + c];
        uint8_t v11 = src[(y1 * src_width + x1) * 3 + c];

        // 双线性插值
        float interpolated =
            (v00 * wx0 * wy0) + (v01 * wx1 * wy0) +
            (v10 * wx0 * wy1) + (v11 * wx1 * wy1);

        // 归一化 (0-1)
        float normalized = interpolated * norm_factor;

        // 计算目标索引 (CHW布局)
        int dst_idx = c * dst_height * dst_width + dst_y * dst_width + dst_x;
        dst[dst_idx] = normalized;
    }
}

__global__ void preprocess_kernel_uint8_to_fp16(
    const uint8_t* src,      // 输入：NHWC布局 [height, width, 3]
    half* dst,               // 输出：NCHW布局 [3, height, width]
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const float scale_x,
    const float scale_y,
    const float norm_factor
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height) return;

    // 计算源图像坐标（双线性插值）
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float wx1 = src_x - x0;
    float wx0 = 1.0f - wx1;
    float wy1 = src_y - y0;
    float wy0 = 1.0f - wy1;

    // 对每个通道进行双线性插值
    for (int c = 0; c < 3; c++) {
        // 获取4个源像素值
        uint8_t v00 = src[(y0 * src_width + x0) * 3 + c];
        uint8_t v01 = src[(y0 * src_width + x1) * 3 + c];
        uint8_t v10 = src[(y1 * src_width + x0) * 3 + c];
        uint8_t v11 = src[(y1 * src_width + x1) * 3 + c];

        // 双线性插值
        float interpolated =
            (v00 * wx0 * wy0) + (v01 * wx1 * wy0) +
            (v10 * wx0 * wy1) + (v11 * wx1 * wy1);

        // 归一化 (0-1) 并转换为fp16
        half normalized = __float2half(interpolated * norm_factor);

        // 计算目标索引 (CHW布局)
        int dst_idx = c * dst_height * dst_width + dst_y * dst_width + dst_x;
        dst[dst_idx] = normalized;
    }
}

// 包装函数，在C++中调用
extern "C" void launch_preprocess_kernel(
    const uint8_t* src,
    void* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    bool is_fp16,
    cudaStream_t stream
) {
    // 计算缩放比例
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    // 配置CUDA核函数
    dim3 block(16, 16);  // 256个线程/block
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);

    if (is_fp16) {
        preprocess_kernel_uint8_to_fp16<<<grid, block, 0, stream>>>(
            src, (half*)dst, src_width, src_height,
            dst_width, dst_height, scale_x, scale_y, 1.0f/255.0f);
    } else {
        preprocess_kernel_uint8_to_fp32<<<grid, block, 0, stream>>>(
            src, (float*)dst, src_width, src_height,
            dst_width, dst_height, scale_x, scale_y, 1.0f/255.0f);
    }
}