import cv2
import os

def split_video_to_frames(video_path, output_folder):
    """
    将视频文件拆分为单帧图像并保存到指定文件夹。

    参数:
    video_path (str): 输入视频文件的路径。
    output_folder (str): 保存帧图像的文件夹路径。
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建文件夹: {output_folder}")

    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        print(f"错误: 视频文件不存在于 {video_path}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    frame_count = 0
    print(f"开始处理视频: {video_path}")

    while True:
        # 逐帧读取
        success, frame = cap.read()

        # 如果读取失败（视频结束），则退出循环
        if not success:
            break

        # 构建输出文件名 (例如: frame_000001.jpg)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")

        # 保存当前帧
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"处理完成。总共保存了 {frame_count} 帧到文件夹 '{output_folder}'。")

if __name__ == '__main__':
    # --- 配置 ---
    # 将 'your_video.mp4' 替换为你的视频文件名
    input_video_path = 'results\point_CAD_foot_output_0927.mp4'
    # 输出文件夹的名称
    output_frames_folder = 'frames'
    # --- 配置结束 ---

    def combine_video_frames(video_path1, video_path2, output_folder):
        """
        将两个视频的帧按顺序水平合并并保存。

        参数:
        video_path1 (str): 第一个输入视频文件的路径 (将显示在左侧)。
        video_path2 (str): 第二个输入视频文件的路径 (将显示在右侧)。
        output_folder (str): 保存合并后帧图像的文件夹路径。
        """
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"已创建文件夹: {output_folder}")

        # 打开两个视频文件
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)

        # 检查视频是否成功打开
        if not cap1.isOpened():
            print(f"错误: 无法打开视频文件 {video_path1}")
            return
        if not cap2.isOpened():
            print(f"错误: 无法打开视频文件 {video_path2}")
            cap1.release()
            return

        frame_count = 0
        print(f"开始合并视频帧: {video_path1} 和 {video_path2}")

        while True:
            # 从每个视频中逐帧读取
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()

            # 如果任一视频读取结束，则退出循环
            if not success1 or not success2:
                break

            # 确保两个帧的高度相同以便拼接，如果不同则调整
            h1, _, _ = frame1.shape
            h2, _, _ = frame2.shape
            if h1 != h2:
                # 将两个帧的高度调整为两者中较小的高度
                target_height = min(h1, h2)
                # 计算新的宽度以保持纵横比
                w1_new = int(frame1.shape[1] * target_height / h1)
                w2_new = int(frame2.shape[1] * target_height / h2)
                frame1 = cv2.resize(frame1, (w1_new, target_height))
                frame2 = cv2.resize(frame2, (w2_new, target_height))

            # 水平合并两个帧 (frame1在左, frame2在右)
            combined_frame = cv2.hconcat([frame1, frame2])

            # 构建输出文件名
            frame_filename = os.path.join(output_folder, f"combined_frame_{frame_count:06d}.jpg")

            # 保存合并后的帧
            cv2.imwrite(frame_filename, combined_frame)

            frame_count += 1

        # 释放视频捕获对象
        cap1.release()
        cap2.release()
        print(f"处理完成。总共合并并保存了 {frame_count} 帧到文件夹 '{output_folder}'。")


    # --- 新的配置 ---
    # 替换为你的两个视频文件名
    input_video_path1 = 'results/output_annotated.mp4'  # 左侧视频
    input_video_path2 = "results/normal.mp4"  # 右侧视频

    # 调用新函数来合并视频帧
    def combine_video_frames_vertical(video_path1, video_path2, output_folder):
        """
        将两个视频的帧按顺序垂直合并并保存。

        参数:
        video_path1 (str): 第一个输入视频文件的路径 (将显示在上半部分)。
        video_path2 (str): 第二个输入视频文件的路径 (将显示在下半部分)。
        output_folder (str): 保存合并后帧图像的文件夹路径。
        """
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"已创建文件夹: {output_folder}")

        # 打开两个视频文件
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)

        # 检查视频是否成功打开
        if not cap1.isOpened():
            print(f"错误: 无法打开视频文件 {video_path1}")
            return
        if not cap2.isOpened():
            print(f"错误: 无法打开视频文件 {video_path2}")
            cap1.release()
            return

        frame_count = 0
        print(f"开始合并视频帧: {video_path1} 和 {video_path2}")

        while True:
            # 从每个视频中逐帧读取
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()

            # 如果任一视频读取结束，则退出循环
            if not success1 or not success2:
                break

            # 确保两个帧的宽度相同以便拼接，如果不同则调整
            _, w1, _ = frame1.shape
            _, w2, _ = frame2.shape
            if w1 != w2:
                # 将两个帧的宽度调整为两者中较小的宽度
                target_width = min(w1, w2)
                # 计算新的高度以保持纵横比
                h1_new = int(frame1.shape[0] * target_width / w1)
                h2_new = int(frame2.shape[0] * target_width / w2)
                frame1 = cv2.resize(frame1, (target_width, h1_new))
                frame2 = cv2.resize(frame2, (target_width, h2_new))

            # 垂直合并两个帧 (frame1在上, frame2在下)
            combined_frame = cv2.vconcat([frame1, frame2])

            # 构建输出文件名
            frame_filename = os.path.join(output_folder, f"combined_frame_{frame_count:06d}.jpg")

            # 保存合并后的帧
            cv2.imwrite(frame_filename, combined_frame)

            frame_count += 1

        # 释放视频捕获对象
        cap1.release()
        cap2.release()
        print(f"处理完成。总共合并并保存了 {frame_count} 帧到文件夹 '{output_folder}'。")


    # --- 新的配置 ---
    # 替换为你的两个视频文件名
    input_video_path1 = 'results/output_cad.mp4'  # 上半部分视频
    input_video_path2 = "video/input.mp4"  # 下半部分视频

    # 调用新函数来合并视频帧
    combine_video_frames_vertical(input_video_path1, input_video_path2, 'frames')