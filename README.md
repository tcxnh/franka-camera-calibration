# franka-camera-calibration
# How to use 
1. create conda environment with packages in requirement.txt
2. fill your own config setting
   
   '''
   python 
   self.config = {
            # 标定板参数
            'checkerboard_size': (10, 8),  # 标定板内角点数量
            'square_size': 0.015,  # 标定板方格尺寸(米)
            
            # 相机参数
            'camera_frames': ['cam_right', 'cam_left'],  # 相机名称
            
            # 目标参数 - 自定义配置
            'sensor_cam_eye_pos': [0.3, 0, 0.6],
            'sensor_cam_target_pos': [-0.1, 0, 0.1],
            'human_cam_eye_pos': [0.6, 0.7, 0.6],
            'human_cam_target_pos': [0.0, 0.0, 0.35],
            
            # 标定参数
            'num_calibration_images': 20,  # 每个相机采集的图像数量, 推荐15张以上
            'calibration_delay': 0.5,  # 捕获图像之间的延迟(秒)
            
            # 参考坐标系
            'reference_frame': 'world',  # 世界坐标系作为参考
        }
   python
   '''
   
   3. run the program and repalce camera with the feedback
