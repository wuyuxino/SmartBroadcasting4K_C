# bash!/bin/bash
trtexec \
--onnx=best.onnx \
--saveEngine=best.engine \
--fp16 \
--workspace=16384 \
--inputIOFormats=fp16:chw \
--outputIOFormats=fp16:chw \
--explicitBatch \
--device=0 \
--verbose


# 基础命令：将 best.onnx 转换为 FP16 精度的 TensorRT engine
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16