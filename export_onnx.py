import torch
import torch.nn as nn
import os
import sys
import copy

# 1. 尝试导入模块
try:
    from utils.get_model import getmodel
except ImportError:
    sys.path.append(os.getcwd())
    try:
        from utils.get_model import getmodel
    except ImportError as e:
        print(f"导入失败: {e}")
        exit(1)

class FrozenBatchNorm3d(nn.Module):
    """
    将 BatchNorm3d 转换为等价的乘法和加法操作。
    这能彻底消除 ONNX 中的 BatchNorm3D 算子，完美适配昇腾 ATC，无需依赖图融合。
    """
    def __init__(self, bn):
        super().__init__()
        # 提取原始 BN 的参数
        if bn.weight is not None:
            weight = bn.weight.data
            bias = bn.bias.data
        else:
            weight = torch.ones(bn.num_features, device=bn.running_mean.device)
            bias = torch.zeros(bn.num_features, device=bn.running_mean.device)
        
        running_mean = bn.running_mean.data
        running_var = bn.running_var.data
        eps = bn.eps
        
        # 预计算缩放系数 scale 和偏移量 shift
        scale = weight / torch.sqrt(running_var + eps)
        shift = bias - running_mean * scale
        
        # 注册为 buffer，将形状重塑为 (1, C, 1, 1, 1) 以支持 3D 张量(B, C, D, H, W)的广播机制
        self.register_buffer('scale', scale.view(1, -1, 1, 1, 1))
        self.register_buffer('shift', shift.view(1, -1, 1, 1, 1))

    def forward(self, x):
        # 仅执行基础的乘法和加法操作
        return x * self.scale + self.shift

def replace_bn3d_with_math(model):
    """
    递归深度遍历模型，将所有的 nn.BatchNorm3d 替换为自定义的 FrozenBatchNorm3d。
    """
    model_replaced = copy.deepcopy(model)
    for name, module in model_replaced.named_children():
        if isinstance(module, nn.BatchNorm3d):
            # 将其实例化为等价的数学运算模块
            setattr(model_replaced, name, FrozenBatchNorm3d(module))
        else:
            # 继续递归深入子模块
            setattr(model_replaced, name, replace_bn3d_with_math(module))
    return model_replaced

def convert_to_onnx():
    # --- 配置路径 ---
    weight_path = 'weights/best094nophone.pt'
    output_onnx_path = 'best094nophone.onnx'
    
    print(f">>> 正在准备转换模型...")

    # 2. 加载 PyTorch 模型
    try:
        model = getmodel(weight_path)
        model.eval()  # 必须先执行 eval，确保 BN 使用 running_mean/var
        print(">>> 原始模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 替换 BatchNorm3D (核心步骤)
    print(">>> 正在将 BatchNorm3D 替换为等价的乘加操作 (消除ATC不支持的算子)...")
    try:
        model = replace_bn3d_with_math(model)
        print(">>> 替换完成，所有 BatchNorm3D 已被安全转换为基础数学算子。")
    except Exception as e:
        print(f"算子替换失败: {e}")
        return

    # 4. 定义输入
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    # 5. 执行导出
    print(">>> 开始导出 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        training=torch.onnx.TrainingMode.EVAL, # 强制评估模式
        export_params=True,
        opset_version=12,  # 12 对 3D 算子支持较稳
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f">>> 转换完成！输出文件: {output_onnx_path}")

    # 6. 验证并简化 (进一步确保兼容性)
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(output_onnx_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_onnx_path)
            print(">>> ONNX 简化成功，结构已优化。")
        else:
            print(">>> 警告: ONNX 简化检查未通过，请注意。")
    except Exception as e:
        print(f">>> 跳过简化步骤 (可能未安装 onnxsim 或简化过程出错): {e}")

if __name__ == "__main__":
    convert_to_onnx()