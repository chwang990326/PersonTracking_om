import cv2
import os
import numpy as np
from natsort import natsorted

"""
用于将图片序列合成为视频的脚本。

假设图片存放在 'image' 文件夹中，输出视频为 'output_video.mp4'。
"""

def imread_zh(file_path):
    """
    用于读取包含中文字符路径的图片的函数。
    """
    # 使用numpy读取文件，然后用OpenCV解码
    frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame

def create_video_from_images(image_folder, video_name, fps=1):
    """
    从图片文件夹创建视频。

    :param image_folder: 包含图片的文件夹路径。
    :param video_name: 输出视频文件的名称（例如 'output.mp4'）。
    :param fps: 视频的帧率。
    """
    # 获取文件夹中所有的png图片
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    # 如果没有找到图片，则退出
    if not images:
        print(f"在文件夹 '{image_folder}' 中没有找到PNG图片。")
        return

    # 使用自然排序对图片文件名进行排序
    images = natsorted(images)

    # 读取第一张图片以获取视频尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = imread_zh(first_image_path) # 使用新的读取函数
    if frame is None:
        print(f"无法读取第一张图片: {first_image_path}")
        return
    height, width, layers = frame.shape
    size = (width, height)

    # 初始化VideoWriter
    # 'mp4v' 是 MP4 格式的编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_name, fourcc, fps, size)

    print("开始合成视频...")
    # 逐一将图片写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = imread_zh(image_path) # 使用新的读取函数
        if frame is not None:
            # 确保所有帧的尺寸都与第一帧相同
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            out.write(frame)
        else:
            print(f"警告: 无法读取图片 {image_path}，已跳过。")

    # 释放资源
    out.release()
    print(f"视频 '{video_name}' 已成功创建。")

if __name__ == '__main__':
    # 设置图片文件夹和输出视频文件名
    image_dir = 'image'
    output_video_file = 'output_video.mp4'
    
    # 调用函数创建视频，帧率设置为1（即每秒播放一张图片）
    create_video_from_images(image_dir, output_video_file, fps=1)