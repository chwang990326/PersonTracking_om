from ultralytics import YOLO

def export_to_onnx(model_path, imgsz=640, opset=11):
    """
    将 YOLO 模型导出为固定输入的 ONNX 格式
    """
    print(f"========== 开始处理模型: {model_path} ==========")
    
    # 1. 加载模型
    model = YOLO(model_path)
    
    # 2. 执行导出
    print(f"正在导出为 ONNX (输入尺寸锁定为 {imgsz}x{imgsz}, opset={opset})...")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,      # 绝对锁定输入 Shape：[1, 3, imgsz, imgsz]
        simplify=True,      # 开启 ONNX 简化，优化 NPU 部署
        opset=opset         # 匹配 Ascend ATC 的推荐版本
    )
    
    print(f"导出成功！ONNX 模型已保存至: {export_path}\n")
    return export_path

if __name__ == "__main__":
    # 定义需要转换的模型列表
    models_to_convert = [
        "./weights/yolo11s.pt",
        "./weights/yolo11s-pose.pt"
    ]
    
    # 批量执行转换
    for model_name in models_to_convert:
        export_to_onnx(model_name)
        
    print("所有模型转换完毕，准备进行 ATC 转换！")