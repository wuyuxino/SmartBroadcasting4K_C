import torch
import json

# 直接把 KFDeepLearningModel 类复制到这里（仅保留核心必要部分）
class KFDeepLearningModel(torch.nn.Module):
    def __init__(self):
        super(KFDeepLearningModel, self).__init__()
        self.Q_log = torch.nn.Parameter(torch.log(torch.eye(4, dtype=torch.float32) * 0.1))
        self.R_log = torch.nn.Parameter(torch.log(torch.eye(2, dtype=torch.float32) * 1.0))

        self.F = torch.tensor([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float32)
        self.H = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=torch.float32)
        self.init_P = torch.eye(4, dtype=torch.float32) * 1000.0

    @property
    def Q(self):
        return torch.exp(self.Q_log) + 1e-6 * torch.eye(4, dtype=torch.float32).to(self.Q_log.device)

    @property
    def R(self):
        return torch.exp(self.R_log) + 1e-6 * torch.eye(2, dtype=torch.float32).to(self.R_log.device)

# 配置路径（你的目录下已经有 trained_kf_model.pth，无需修改）
MODEL_PATH = "./trained_kf_model.pth"
OUTPUT_JSON_PATH = "./kalman_params.json"

def main():
    try:
        # 加载模型
        model = KFDeepLearningModel()
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 提取 Q/R 矩阵（转换为 numpy 数组，再转列表）
        Q_np = model.Q.detach().cpu().numpy().tolist()
        R_np = model.R.detach().cpu().numpy().tolist()

        # 保存为 json
        kalman_params = {
            "Q": Q_np,
            "R": R_np
        }

        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(kalman_params, f, indent=4)

        print(f"✅ 成功！Q/R 参数已保存到：{OUTPUT_JSON_PATH}")
        print(f"📄 生成的文件在当前目录下，后续 C++ 代码直接加载该文件即可")

    except Exception as e:
        print(f"❌ 出错了：{str(e)}")
        print(f"💡 检查：是否存在 {MODEL_PATH} 文件？是否是合法的 PyTorch 模型文件？")

if __name__ == "__main__":
    main()