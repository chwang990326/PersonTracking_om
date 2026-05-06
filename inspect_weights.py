import torch
from models.actionclassifier import uniformer_small

def inspect_pth_keys(weights_path):
    # 1. 初始化本地模型
    model = uniformer_small(num_classes=7)
    local_keys = set(model.state_dict().keys())

    # 2. 加载权重文件
    print(f"--- 正在分析权重文件: {weights_path} ---")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # 获取 state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        pth_state_dict = checkpoint['state_dict']
    else:
        pth_state_dict = checkpoint
    
    pth_keys = set(pth_state_dict.keys())

    # 3. 分析分类头 (ClsHead) 相关的 Key
    print("\n[1. 分类头权重搜索]")
    head_related = [k for k in pth_keys if 'head' in k.lower() or 'fc_cls' in k.lower()]
    if head_related:
        print("权重文件中疑似分类头的键名:")
        for k in head_related:
            print(f"  - {k} (Shape: {pth_state_dict[k].shape})")
    else:
        print("未在权重文件中找到包含 'head' 或 'fc_cls' 的键。")

    # 4. 统计匹配情况
    # 尝试模拟 get_model.py 中的初步清洗逻辑
    cleaned_pth_keys = set()
    for k in pth_keys:
        name = k
        if name.startswith("module."): name = name[7:]
        if name.startswith("backbone."): name = name[9:]
        if "cls_head.fc_cls." in name: name = name.replace("cls_head.fc_cls.", "head.")
        cleaned_pth_keys.add(name)

    missing = local_keys - cleaned_pth_keys
    unexpected = cleaned_pth_keys - local_keys

    print("\n[2. 匹配分析 (基于当前映射逻辑)]")
    print(f"本地模型总键数: {len(local_keys)}")
    print(f"权重文件清洗后总键数: {len(cleaned_pth_keys)}")
    
    if len(missing) == 0:
        print("✅ 完美匹配！所有本地模型参数在权重文件中都有对应。")
    else:
        print(f"❌ 缺失的键 (模型需要但权重里没有): {len(missing)} 个")
        # 列出前 5 个作为参考
        for k in sorted(list(missing))[:5]:
            print(f"  - {k}")
        if len(missing) > 5: print("  ... 等等")

    if len(unexpected) > 0:
        print(f"⚠️ 冗余的键 (权重里有但模型不需要): {len(unexpected)} 个")
        for k in sorted(list(unexpected))[:5]:
            print(f"  - {k}")
        if len(unexpected) > 5: print("  ... 等等")

if __name__ == "__main__":
    # 请替换为您实际的权重路径
    WEIGHTS_PATH = 'config/best_acc_top1_epoch_31.pth'
    inspect_pth_keys(WEIGHTS_PATH)