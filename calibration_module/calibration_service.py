import cv2
import numpy as np
import time

class CalibrationService:
    def __init__(self):
        # ========= 配置参数 (可根据实际情况调整) =========
        self.BOARD_COLS = 8   # 棋盘格列数
        self.BOARD_ROWS = 11  # 棋盘格行数
        self.SQUARE_SIZE_MM = 60.0
        
        # 房间坐标系原点 (对应 calibrate_new.py 中的 ROOM_ORIGIN_MM)
        # 实际项目中，这里可能需要根据入参 calibration_world_coordinates 动态计算，
        # 但目前我们保留原脚本逻辑，使用固定值或默认值
        self.ROOM_ORIGIN_MM = np.array([1246072.0, 578116.0, 0.0], dtype=np.float64)
        self.R_room_from_board = np.eye(3, dtype=np.float64)

    def _normalize_chessboard_corners(self, corners, pattern_size, direction=None):
        """(保留原脚本逻辑) 标准化棋盘格角点顺序"""
        if corners is None: return None
        corners_2d = corners.reshape(-1, 2)
        width, height = pattern_size
        
        diff = np.diff(corners_2d, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        median_dist = np.median(distances)
        breaks = np.where(distances > median_dist * 1.5)[0]
        
        if len(breaks) > 0:
            first_row_or_col_length = breaks[0] + 1
            if first_row_or_col_length == width:
                is_row_major = True
                rows, cols = height, width
            else:
                is_row_major = False
                rows, cols = width, height
        else:
            is_row_major = True
            rows, cols = height, width
        
        if is_row_major:
            grid = corners_2d.reshape(rows, cols, 2)
        else:
            grid = corners_2d.reshape(rows, cols, 2)
            grid = grid.transpose(1, 0, 2)
            rows, cols = cols, rows
        
        top_left = grid[0, 0]
        top_right = grid[0, -1]
        bottom_left = grid[-1, 0]
        bottom_right = grid[-1, -1]
        
        corners_dict = {
            'top_left': (0, 0, top_left),
            'top_right': (0, -1, top_right),
            'bottom_left': (-1, 0, bottom_left),
            'bottom_right': (-1, -1, bottom_right)
        }
        
        min_sum = float('inf')
        target_corner = None
        
        # 定义计算权重的方法：默认寻找左上角 (x+y 最小)
        def get_score_default(point): return point[0] + point[1]
        
        # 定义计算权重的方法：寻找最左侧 (x 最小)
        def get_score_min_x(point): return point[0]
        
        # 根据拍摄方向选择不同的评价函数
        # 需求：X-Y+单独特异化处理，其他保留结构但沿用默认逻辑
        
        if direction == "X-Y+":
            # 规则：选择像素坐标中最偏左侧的点 (x 最小)
            score_func = get_score_min_x
        elif direction == "X+":
            score_func = get_score_default
        elif direction == "X-":
            score_func = get_score_default
        elif direction == "Y+":
            score_func = get_score_default
        elif direction == "Y-": # 注意：API中可能出现 X+Y- 等，需确保覆盖所有case
             score_func = get_score_default
        elif direction == "X+Y+":
            score_func = get_score_default
        elif direction == "X-Y-":
            score_func = get_score_default
        elif direction == "X+Y-":
            score_func = get_score_default
        else:
            # 兜底默认逻辑
            score_func = get_score_default

        # 执行选择逻辑
        for name, (_, _, point) in corners_dict.items():
            val = score_func(point)
            if val < min_sum:
                min_sum = val
                target_corner = name
        
        if target_corner == 'top_left': normalized_grid = grid
        elif target_corner == 'top_right': normalized_grid = np.fliplr(grid)
        elif target_corner == 'bottom_left': normalized_grid = np.flipud(grid)
        elif target_corner == 'bottom_right': normalized_grid = np.flipud(np.fliplr(grid))
        
        return normalized_grid.reshape(-1, 1, 2)

    def _find_corners(self, img, direction=None):
        """(保留原脚本逻辑) 查找棋盘角点"""
        pattern_size = (self.BOARD_COLS, self.BOARD_ROWS)
        ret, corners = cv2.findChessboardCornersSB(
            img, pattern_size, 
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if not ret:
            ret, corners = cv2.findChessboardCorners(img, pattern_size)
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = cv2.cornerSubPix(
                    gray, corners, (11,11), (-1,-1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        if ret:
            corners = self._normalize_chessboard_corners(corners, pattern_size, direction)
        return ret, corners

    def _build_chessboard_object_points(self, direction):
        """(保留原脚本逻辑) 生成与检测结果匹配的3D角点"""
        cols, rows = self.BOARD_COLS, self.BOARD_ROWS
        square = self.SQUARE_SIZE_MM
        
        objp = np.zeros((rows * cols, 3), np.float32)
        
        # 简化逻辑：直接复用原脚本的 direction 判断
        # 注意：此处必须确保 direction 字符串与 API 传入的 camera_orientation 一致
        if direction == "X+":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [(7 - c) * square, r * square, 0]
        elif direction == "X+Y+":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [(cols - 1 - c) * square, (rows - 1 - r) * square, 0]
        elif direction == "X-Y+":
             for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [(cols - 1 - c) * square, (rows - 1 - r) * square, 0]
        elif direction == "X-":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [c * square, (10 - r) * square, 0]
        elif direction == "X-Y-":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [c * square, (10 - r) * square, 0]
        elif direction == "Y+":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [c * square, r * square, 0]
        elif direction == "X+Y-":
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [c * square, r * square, 0]
        else:
            # 留给y-方向及默认情况
            for r in range(rows):
                for c in range(cols):
                    objp[r * cols + c] = [c * square, r * square, 0]
                    
        return objp

    def _select_origin_from_world_corners(self, corners):
        """
        从4个世界坐标中选择原点
        业务背景：
        1. 棋盘格轴与世界坐标系轴方向一致。
        2. 原点是所有点在棋盘格坐标系第一象限的基础（即它是左下角）。
        
        逻辑：在此约束下，(x最小且y最小) 等价于 (x+y最小)。
        """
        if not corners or len(corners) == 0:
            return self.ROOM_ORIGIN_MM 
        
        # 你的场景下，这就是寻找几何上的“左下角”
        origin = min(corners, key=lambda p: p[0] + p[1])
        
        # === 新增逻辑：确保是 3D 坐标 ===
        if len(origin) == 2:
            # 如果只有 [x, y]，补充 z=0.0
            origin = [origin[0], origin[1], 0.0]
        
        return np.array(origin, dtype=np.float64)

    def calibrate(self, intrinsic_images, anchor_images, direction, calibration_world_coordinates=None):
        """
        核心标定函数
        :param intrinsic_images: 用于内参标定的图片列表
        :param anchor_images: 用于外参标定的图片列表
        :param direction: 相机朝向
        :param calibration_world_coordinates: 标定板4角世界坐标 [[x,y,z],...]
        """
        start_time = time.time()
        
        # 0. 确定房间坐标系原点 (ROOM_ORIGIN_MM)
        room_origin = self._select_origin_from_world_corners(calibration_world_coordinates)
        
        # 1. 内参标定
        objp = self._build_chessboard_object_points(direction)
        objpoints = []
        imgpoints = []
        imsize = None
        
        print(f"正在进行内参标定，图片数量: {len(intrinsic_images)}")
        
        for img in intrinsic_images:
            if imsize is None:
                imsize = (img.shape[1], img.shape[0])
            
            # 传递 direction 参数
            found, corners = self._find_corners(img, direction)
            if found:
                imgpoints.append(corners.reshape(-1, 1, 2).astype(np.float32))
                # 简单判断方向以匹配 object points 顺序 (复用原脚本逻辑)
                if corners.reshape(-1, 2)[0][0] <= corners.reshape(-1, 2)[1][0]:
                    objpoints.append(objp)
                else:
                    objpoints.append(objp[::-1])
        
        if len(objpoints) < 3: # OpenCV 建议至少3张，脚本里是5张，这里放宽一点防止报错
            raise ValueError(f"有效标定板图片不足 (至少需要3张检测到角点的图片)")
            
        # 使用图像高度作为焦距的初始猜测 (Focal Length Guess)
        focal_length = imsize[1] 
        cx, cy = imsize[0] / 2, imsize[1] / 2

        init_camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        ret, K, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, imsize, init_camera_matrix, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        
        # 2. 外参标定
        # 通常使用第一张有效的 anchor_image 作为基准
        rvec_final, tvec_final = None, None
        
        print(f"正在进行外参标定，图片数量: {len(anchor_images)}")
        
        for img in anchor_images:
            # 传递 direction 参数
            found, corners = self._find_corners(img, direction)
            if not found:
                continue
                
            current_objp = objp
            if corners.reshape(-1, 2)[0][0] > corners.reshape(-1, 2)[1][0]:
                current_objp = objp[::-1]
                
            success, rvec, tvec = cv2.solvePnP(
                current_objp, corners.reshape(-1, 1, 2), K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rvec_final = rvec
                tvec_final = tvec
                break # 找到一张有效的即可
        
        if rvec_final is None:
            raise ValueError("外参标定失败：无法在原点图片(origin_images)中检测到棋盘格")

        cost_time = time.time() - start_time
        
        return {
            "camera_matrix": K,
            "dist_coeffs": dist,
            "image_width": imsize[0],
            "image_height": imsize[1],
            "rvec": rvec_final,
            "tvec": tvec_final,
            "room_origin": room_origin,  # 返回计算出的原点
            "cost_time": cost_time
        }