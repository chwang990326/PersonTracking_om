import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

class ImageCheckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ReID 预处理效果查看器 (Gray World + CLAHE)")
        self.root.geometry("1200x700")

        # 顶部控制区
        control_frame = Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = Button(control_frame, text="选择图片", command=self.load_image, font=("Arial", 12), bg="#ddd")
        self.btn_load.pack()

        # 图片显示区
        display_frame = Frame(root)
        display_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 左侧：原图
        self.frame_left = Frame(display_frame)
        self.frame_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        Label(self.frame_left, text="原始图像 (Original)", font=("Arial", 14, "bold")).pack(side=tk.TOP, pady=5)
        self.lbl_img_orig = Label(self.frame_left, bg="gray")
        self.lbl_img_orig.pack(expand=True, fill=tk.BOTH)

        # 右侧：处理后
        self.frame_right = Frame(display_frame)
        self.frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        Label(self.frame_right, text="预处理后 (Preprocessed)", font=("Arial", 14, "bold"), fg="blue").pack(side=tk.TOP, pady=5)
        self.lbl_img_proc = Label(self.frame_right, bg="gray")
        self.lbl_img_proc.pack(expand=True, fill=tk.BOTH)

    def _preprocess_image(self, image):
        """
        [核心逻辑] 严格复用 personReID.py 中的预处理代码
        Gray World 白平衡 + CLAHE 直方图均衡化
        """
        if image is None or image.size == 0:
            return image

        # 1. Gray World Algorithm (白平衡)
        b, g, r = cv2.split(image)
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        
        # 避免除零
        b_avg = b_avg if b_avg > 1e-5 else 1.0
        g_avg = g_avg if g_avg > 1e-5 else 1.0
        r_avg = r_avg if r_avg > 1e-5 else 1.0
        
        avg_gray = (b_avg + g_avg + r_avg) / 3
        
        k_b = avg_gray / b_avg
        k_g = avg_gray / g_avg
        k_r = avg_gray / r_avg
        
        b = np.clip(b * k_b, 0, 255).astype(np.uint8)
        g = np.clip(g * k_g, 0, 255).astype(np.uint8)
        r = np.clip(r * k_r, 0, 255).astype(np.uint8)
        
        image_gw = cv2.merge([b, g, r])

        # 2. CLAHE (限制对比度自适应直方图均衡化)
        lab = cv2.cvtColor(image_gw, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image_clahe

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return

        # 读取图片
        # [修改] 使用 np.fromfile + cv2.imdecode 以支持中文路径
        try:
            img_array = np.fromfile(file_path, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取出错: {e}")
            return

        if img_bgr is None:
            print("无法读取图片")
            return

        # 执行预处理
        img_processed_bgr = self._preprocess_image(img_bgr)

        # 显示图片
        self.display_image(img_bgr, self.lbl_img_orig)
        self.display_image(img_processed_bgr, self.lbl_img_proc)

    def display_image(self, cv_img, label_widget):
        # 转换颜色空间 BGR -> RGB
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 获取 Label 的尺寸，进行自适应缩放
        # 注意：这里为了简单，固定一个最大显示尺寸，避免图片过大撑爆窗口
        max_w, max_h = 550, 550
        h, w = img_rgb.shape[:2]
        
        scale = min(max_w/w, max_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # 转换为 PIL 格式并显示
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label_widget.config(image=img_tk)
        label_widget.image = img_tk  # 保持引用，防止被垃圾回收

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCheckApp(root)
    root.mainloop()