import torch
import numpy as np
from models.actionclassifier import uniformer_small

# ================= 兼容性处理 =================
try:
    from mmengine.logging.history_buffer import HistoryBuffer
    torch.serialization.add_safe_globals([HistoryBuffer])
except ImportError:
    pass

# 注册 numpy 基础重建函数
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
# =============================================

def getmodel(pretained_weights):
    print(f"Loading Uniformer Small Model from: {pretained_weights}")
    
    # 1. 初始化模型 (UniFormer)
    # 内部默认 num_classes=7
    model = uniformer_small(num_classes=6)

    # 2. 加载权重文件
    # 使用 weights_only=False 彻底解决自定义类加载问题
    try:
        checkpoint = torch.load(pretained_weights, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(pretained_weights, map_location='cpu')

    # 3. 提取权重字典
    # 3. 提取权重字典
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # 适配你现在这个 best0851.pt 文件的格式
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # 兼容以前的其他格式
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 4. 核心修复：重命名键名以匹配 UniFormer 结构
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # a. 处理分布式训练的 module. 前缀
        if name.startswith("module."):
            name = name[7:]
            
        # b. 处理 MMAction2 的 backbone 前缀
        if name.startswith("backbone."):
            name = name[9:]  # 去掉 "backbone."
            
        # c. 处理分类头名称映射
        # MMAction2 的 I3DHead 使用 fc_cls，而你的 UniFormer 使用 head
        if name.startswith("cls_head.fc_cls."):
            name = name.replace("cls_head.fc_cls.", "head.")
        
        # d. 过滤掉不属于 UniFormer 的键（例如 cls_head 中其他的参数）
        # 只有在 new_state_dict 匹配当前模型参数时才加入
        new_state_dict[name] = v

    # 5. 将权重加载到模型中
    # 建议先使用 strict=False 观察是否还有微小不匹配，如果一切正常再改回 True
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print("--------------------------------------------------")
    if len(msg.missing_keys) > 0:
        print(f"Warning: Missing keys during loading: {msg.missing_keys}")
    if len(msg.unexpected_keys) > 0:
        print(f"Warning: Unexpected keys during loading: {msg.unexpected_keys}")
    print(f"Successfully loaded state dict. Strict check passed: {len(msg.missing_keys) == 0}")
    print("--------------------------------------------------")
    
    return model