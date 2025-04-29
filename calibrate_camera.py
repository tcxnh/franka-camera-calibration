#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机外参校准代码 - 针对直连相机系统
使用标定板对两个相机进行校准，并计算它们之间的相对变换关系
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import yaml
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建保存图像和结果的目录
CALIB_DIR = "calibration_data"
os.makedirs(CALIB_DIR, exist_ok=True)


class CameraCalibrator:
    def __init__(self, config_file=None):
        """
        初始化相机校准器
        Args:
            config_file: 可选的配置文件路径
        """
        self.config = {
            # 标定板参数
            'checkerboard_size': (10, 8),  # 标定板内角点数量
            'square_size': 0.015,  # 标定板方格尺寸(米)
            
            # 相机参数
            'camera_frames': ['cam_right', 'cam_left'],  # 相机名称
            
            # 目标参数 - 从用户提供的配置中提取
            'sensor_cam_eye_pos': [0.3, 0, 0.6],
            'sensor_cam_target_pos': [-0.1, 0, 0.1],
            'human_cam_eye_pos': [0.6, 0.7, 0.6],
            'human_cam_target_pos': [0.0, 0.0, 0.35],
            
            # 标定参数
            'num_calibration_images': 10,  # 每个相机采集的图像数量
            'calibration_delay': 0.5,  # 捕获图像之间的延迟(秒)
            
            # 参考坐标系
            'reference_frame': 'world',  # 世界坐标系作为参考
        }
        
        # 如果提供了配置文件，则加载它
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                # 更新配置
                self.config.update(user_config)
        
        # 创建标定板角点的3D坐标
        self.prepare_object_points()
    
    def prepare_object_points(self):
        """
        准备标定板角点的3D坐标
        """
        board_size = self.config['checkerboard_size']
        square_size = self.config['square_size']
        
        # 创建3D点，这里假设标定板放在Z=0的平面上
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # 准备寻找角点的终止标准
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def initialize_realsense_cameras(self):
        """
        初始化两个RealSense相机
        """
        try:
            # 创建RealSense上下文
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) < 2:
                raise Exception(f"只检测到 {len(devices)} 个RealSense设备，需要2个")
            
            print(f"检测到 {len(devices)} 个RealSense设备:")
            for i, dev in enumerate(devices):
                print(f"  设备 {i}: {dev.get_info(rs.camera_info.name)}, 序列号: {dev.get_info(rs.camera_info.serial_number)}")
            
            # 创建两个pipeline和配置
            pipelines = []
            for i in range(2):
                pipeline = rs.pipeline()
                config = rs.config()
                
                # 获取设备序列号
                serial = devices[i].get_info(rs.camera_info.serial_number)
                config.enable_device(serial)
                
                # 启用彩色流
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                # 启动pipeline
                print(f"正在启动相机 {i} (序列号: {serial})...")
                profile = pipeline.start(config)
                
                # 打印活动流信息
                for stream in profile.get_streams():
                    if stream.stream_type() == rs.stream.color:
                        stream_profile = stream.as_video_stream_profile()
                        print(f"  相机 {i} 活动流: {stream.stream_name()}, " 
                            f"分辨率: {stream_profile.width()}x{stream_profile.height()}, "
                            f"FPS: {stream_profile.fps()}")
                
                pipelines.append(pipeline)
            
            # 预热相机
            print("正在预热相机...")
            for _ in range(30):
                for pipeline in pipelines:
                    pipeline.wait_for_frames(1000)
            
            return pipelines
            
        except Exception as e:
            print(f"初始化RealSense相机时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def capture_calibration_images(self, pipelines):
        """
        从两个相机捕获标定图像
        """
        # 初始化存储标定数据的变量
        calibration_data = {cam_name: {'objpoints': [], 'imgpoints': [], 'images': []} 
                        for cam_name in self.config['camera_frames']}
        
        # 检查是否已经有足够的标定图像
        num_images = self.config['num_calibration_images']
        print(f"将为每个相机捕获 {num_images} 张标定图像")
        
        # 创建窗口以显示相机画面
        for cam_name in self.config['camera_frames']:
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, 640, 480)
        
        captured_count = 0
        while captured_count < num_images:
            # 获取两个相机的帧
            frames = []
            frame_valid = []  # 新增一个列表来记录每一帧是否有效
            
            for pipeline in pipelines:
                try:
                    frame = pipeline.wait_for_frames(1000)
                    color_frame = frame.get_color_frame()
                    if color_frame:
                        frames.append(np.asanyarray(color_frame.get_data()))
                        frame_valid.append(True)
                    else:
                        frames.append(None)
                        frame_valid.append(False)
                except Exception as e:
                    print(f"获取相机帧时出错: {e}")
                    frames.append(None)
                    frame_valid.append(False)
            
            # 检查是否成功获取两个相机的帧
            if len(frames) != 2 or False in frame_valid:  # 修改了这行代码
                print("未能从两个相机获取有效帧，重试...")
                continue
            
            # 显示相机画面
            for i, cam_name in enumerate(self.config['camera_frames']):
                if i < len(frames) and frames[i] is not None:
                    # 在图像上绘制指示
                    img_display = frames[i].copy()
                    cv2.putText(img_display, f"将标定板放在两个相机视野中，按's'保存图像 ({captured_count}/{num_images})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 尝试检测角点
                    gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, self.config['checkerboard_size'], None)
                    
                    if ret:
                        # 绘制检测到的角点
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                        cv2.drawChessboardCorners(img_display, self.config['checkerboard_size'], corners2, ret)
                        cv2.putText(img_display, "检测到标定板!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(cam_name, img_display)
            
            # 等待键盘输入
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # q或ESC键退出
                print("用户中止标定过程")
                break
            elif key == ord('s'):  # s键保存当前帧
                # 处理两个相机的图像
                all_corners_found = True
                
                for i, cam_name in enumerate(self.config['camera_frames']):
                    if i < len(frames) and frames[i] is not None:
                        # 转为灰度图
                        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                        
                        # 寻找标定板角点
                        ret, corners = cv2.findChessboardCorners(gray, self.config['checkerboard_size'], None)
                        
                        if ret:
                            # 角点精细化
                            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                            
                            # 保存标定数据
                            calibration_data[cam_name]['objpoints'].append(self.objp)
                            calibration_data[cam_name]['imgpoints'].append(corners2)
                            calibration_data[cam_name]['images'].append(frames[i])
                            
                            # 保存带有检测到角点的图像
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_with_corners = frames[i].copy()
                            cv2.drawChessboardCorners(img_with_corners, self.config['checkerboard_size'], 
                                                    corners2, ret)
                            img_path = os.path.join(CALIB_DIR, f"{cam_name}_calib_{captured_count}_{timestamp}.jpg")
                            cv2.imwrite(img_path, img_with_corners)
                            print(f"保存标定图像: {img_path}")
                        else:
                            all_corners_found = False
                            print(f"在相机 {cam_name} 的图像中未检测到标定板角点")
                
                if all_corners_found:
                    captured_count += 1
                    print(f"已保存标定图像 {captured_count}/{num_images}")
                    time.sleep(self.config['calibration_delay'])
                else:
                    print("请调整标定板位置使两个相机都能清晰看到所有角点")
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        return calibration_data

    
    def calibrate_cameras(self, calibration_data):
        """
        使用捕获的图像校准相机
        """
        camera_parameters = {}
        
        for cam_name, data in calibration_data.items():
            if not data['objpoints'] or not data['imgpoints'] or len(data['images']) == 0:
                print(f"相机 {cam_name} 没有足够的标定数据")
                continue
            
            print(f"正在校准相机 {cam_name}...")
            
            # 获取图像尺寸
            h, w = data['images'][0].shape[:2]
            
            # 相机校准
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                data['objpoints'], data['imgpoints'], (w, h), None, None)
            
            if ret:
                # 计算重投影误差
                mean_error = 0
                for i in range(len(data['objpoints'])):
                    imgpoints2, _ = cv2.projectPoints(data['objpoints'][i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(data['imgpoints'][i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                
                if len(data['objpoints']) > 0:
                    mean_error /= len(data['objpoints'])
                
                # 保存相机参数
                camera_parameters[cam_name] = {
                    'camera_matrix': mtx.tolist(),
                    'distortion_coefficients': dist.tolist(),
                    'rotation_vectors': [r.tolist() for r in rvecs],
                    'translation_vectors': [t.tolist() for t in tvecs],
                    'reprojection_error': float(mean_error)
                }
                
                print(f"相机 {cam_name} 校准成功，重投影误差: {mean_error}")
            else:
                print(f"相机 {cam_name} 校准失败")
        
        return camera_parameters
    
    def estimate_camera_poses(self, calibration_data, camera_parameters):
        """
        估计相机相对于标定板的位姿
        """
        camera_poses = {}
        
        try:
            # 使用最后一组标定图像估计标定板在相机坐标系中的位姿
            for cam_name, params in camera_parameters.items():
                if cam_name not in calibration_data:
                    continue
                
                data = calibration_data[cam_name]
                if not data['objpoints'] or not data['imgpoints']:
                    continue
                
                # 获取相机内参和畸变系数
                camera_matrix = np.array(params['camera_matrix'])
                dist_coeffs = np.array(params['distortion_coefficients'])
                
                # 使用最后一张图像的角点估计标定板位姿
                _, rvec, tvec = cv2.solvePnP(
                    data['objpoints'][-1], 
                    data['imgpoints'][-1], 
                    camera_matrix, 
                    dist_coeffs
                )
                
                # 将旋转向量转换为旋转矩阵
                rmat, _ = cv2.Rodrigues(rvec)
                
                # 计算标定板到相机的变换矩阵
                checkerboard_to_camera = np.eye(4)
                checkerboard_to_camera[:3, :3] = rmat
                checkerboard_to_camera[:3, 3] = tvec.reshape(-1)
                
                # 相机到标定板的变换（取逆）
                camera_to_checkerboard = np.linalg.inv(checkerboard_to_camera)
                
                # 保存估计的相机位姿
                camera_poses[cam_name] = {
                    'checkerboard_to_camera': checkerboard_to_camera.tolist(),
                    'camera_to_checkerboard': camera_to_checkerboard.tolist(),
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist()
                }
                
                print(f"已估计 {cam_name} 相对于标定板的位姿")
            
            return camera_poses
            
        except Exception as e:
            print(f"估计相机位姿时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_stereo_relationship(self, camera_poses):
        """
        计算立体相机之间的相对变换关系
        """
        if len(camera_poses) < 2:
            print("没有足够的相机位姿数据来计算立体关系")
            return None
        
        try:
            cam_names = list(camera_poses.keys())
            
            # 假设第一个相机作为参考坐标系
            ref_cam = cam_names[0]
            other_cam = cam_names[1]
            
            # 获取两个相机到标定板的变换
            ref_to_checker = np.array(camera_poses[ref_cam]['camera_to_checkerboard'])
            other_to_checker = np.array(camera_poses[other_cam]['camera_to_checkerboard'])
            
            # 计算第二个相机相对于第一个相机的变换
            # other_T_ref = other_T_checker * checker_T_ref
            other_to_ref = other_to_checker @ np.linalg.inv(ref_to_checker)
            
            # 从变换矩阵中提取位置和姿态
            position = other_to_ref[:3, 3]
            rotation = R.from_matrix(other_to_ref[:3, :3])
            quaternion = rotation.as_quat()  # x, y, z, w
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            stereo_relationship = {
                'reference_camera': ref_cam,
                'other_camera': other_cam,
                'transform_matrix': other_to_ref.tolist(),
                'position': position.tolist(),
                'quaternion': quaternion.tolist(),
                'euler_angles': euler_angles.tolist()
            }
            
            print(f"\n相机 {other_cam} 相对于相机 {ref_cam} 的位置:")
            print(f"  位置向量: {position}")
            print(f"  欧拉角(XYZ): {euler_angles} 度")
            
            return stereo_relationship
            
        except Exception as e:
            print(f"计算立体相机关系时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_desired_transforms(self, camera_parameters):
        """
        计算符合目标配置的相机变换
        """
        # 获取目标配置
        sensor_eye = self.config['sensor_cam_eye_pos']
        sensor_target = self.config['sensor_cam_target_pos']
        human_eye = self.config['human_cam_eye_pos']
        human_target = self.config['human_cam_target_pos']
        
        # 计算相机到参考坐标系的所需变换
        desired_poses = {}
        
        # 处理第一个相机（传感器相机）
        cam_name = self.config['camera_frames'][0]
        desired_poses[cam_name] = self.compute_camera_transform(sensor_eye, sensor_target)
        
        # 处理第二个相机（人类视角相机）
        cam_name = self.config['camera_frames'][1]
        desired_poses[cam_name] = self.compute_camera_transform(human_eye, human_target)
        
        return desired_poses
    
    def compute_camera_transform(self, eye_pos, target_pos):
        """
        计算相机的变换矩阵
        """
        # 将列表转换为numpy数组
        eye = np.array(eye_pos)
        target = np.array(target_pos)
        
        # 计算相机Z轴方向（相机看向目标的方向）
        z_axis = target - eye
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 假设相机Y轴向上（世界坐标系中的+Z方向）
        up = np.array([0, 0, 1])
        
        # 计算相机X轴（右方向）
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            # 如果相机直接向上或向下看，选择另一个参考向量
            up = np.array([0, 1, 0])
            x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 计算相机Y轴（相机的上方向）
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建旋转矩阵
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        
        # 创建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = eye
        
        # 计算位置和四元数
        position = eye
        rotation_matrix = rotation
        quaternion = R.from_matrix(rotation_matrix).as_quat()  # x, y, z, w
        
        return {
            'transform': transform.tolist(),
            'position': position.tolist(),
            'quaternion': quaternion.tolist()
        }
    
    def generate_camera_adjustment(self, current_poses, desired_poses):
        """
        生成相机调整指南
        """
        adjustment_guides = {}
        
        for cam_name in self.config['camera_frames']:
            if cam_name in current_poses and cam_name in desired_poses:
                # 从当前位姿中获取相机到标定板的变换
                current = current_poses[cam_name]
                desired = desired_poses[cam_name]
                
                # 获取当前和所需的位置
                # 注意：这里我们假设当前位置是通过标定板估计的，可能需要调整
                current_pos = np.array(current['tvec']).reshape(-1)  # 使用相机到标定板的平移向量
                desired_pos = np.array(desired['position'])
                
                # 计算位置差异
                position_diff = desired_pos - current_pos
                
                # 获取当前和所需的旋转
                current_rvec = np.array(current['rvec'])
                current_rmat, _ = cv2.Rodrigues(current_rvec)
                current_rot = R.from_matrix(current_rmat)
                
                desired_quat = np.array(desired['quaternion'])
                desired_rot = R.from_quat(desired_quat)
                
                # 计算相对旋转
                relative_rot = desired_rot * current_rot.inv()
                
                # 计算欧拉角（以度为单位）
                euler_angles = relative_rot.as_euler('xyz', degrees=True)
                
                adjustment_guides[cam_name] = {
                    'position_adjustment': position_diff.tolist(),
                    'rotation_adjustment_euler': euler_angles.tolist(),
                    'current_position': current_pos.tolist(),
                    'desired_position': desired_pos.tolist()
                }
                
                print(f"\n相机 {cam_name} 的调整指南:")
                print(f"  当前位置: {current_pos}")
                print(f"  目标位置: {desired_pos}")
                print(f"  位置调整: {position_diff} 米")
                print(f"  旋转调整: {euler_angles} 度 (XYZ欧拉角)")
        
        return adjustment_guides
    
    def save_calibration_results(self, camera_parameters, camera_poses, stereo_relationship, 
                                desired_poses=None, adjustment_guides=None):
        """
        保存校准结果
        """
        # 创建完整的结果字典
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config,
            'camera_parameters': camera_parameters,
            'camera_poses': camera_poses,
            'stereo_relationship': stereo_relationship
        }
        
        # 如果有所需的位姿和调整指南，也保存它们
        if desired_poses:
            results['desired_camera_poses'] = desired_poses
        
        if adjustment_guides:
            results['adjustment_guides'] = adjustment_guides
        
        # 保存为YAML文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(CALIB_DIR, f"camera_calibration_results_{timestamp}.yaml")
        
        with open(result_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"校准结果已保存到: {result_file}")
        
        return result_file
    
    def visualize_camera_setup(self, camera_poses, stereo_relationship=None, desired_poses=None):
        """
        可视化相机设置
        """
        try:
            # 创建3D图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制参考坐标系
            origin = [0, 0, 0]
            axis_length = 0.2
            ax.quiver(*origin, axis_length, 0, 0, color='r', label='X axis')
            ax.quiver(*origin, 0, axis_length, 0, color='g', label='Y axis')
            ax.quiver(*origin, 0, 0, axis_length, color='b', label='Z axis')
            
            # 绘制标定板 (假设在原点)
            board_size = self.config['checkerboard_size']
            square_size = self.config['square_size']
            board_width = board_size[0] * square_size
            board_height = board_size[1] * square_size
            
            # 标定板四个角点
            board_corners = np.array([
                [0, 0, 0],
                [board_width, 0, 0],
                [board_width, board_height, 0],
                [0, board_height, 0]
            ])
            
            # 绘制标定板
            ax.plot([0, board_width, board_width, 0, 0], 
                    [0, 0, board_height, board_height, 0], 
                    [0, 0, 0, 0, 0], 'k-', label='Checkerboard')
            
            # 绘制相机位置
            colors = ['red', 'blue']
            for i, (cam_name, pose) in enumerate(camera_poses.items()):
                if 'camera_to_checkerboard' in pose:
                    # 获取相机到标定板的变换
                    cam_to_board = np.array(pose['camera_to_checkerboard'])
                    
                    # 相机位置是变换矩阵的平移部分
                    pos = cam_to_board[:3, 3]
                    
                    # 绘制相机位置
                    ax.scatter(pos[0], pos[1], pos[2], color=colors[i % len(colors)], 
                              s=80, label=f"{cam_name}")
                    
                    # 绘制相机坐标系
                    rot = cam_to_board[:3, :3]
                    
                    # X, Y, Z轴
                    ax.quiver(pos[0], pos[1], pos[2], 
                             rot[0, 0]*0.1, rot[1, 0]*0.1, rot[2, 0]*0.1, 
                             color='r')
                    ax.quiver(pos[0], pos[1], pos[2], 
                             rot[0, 1]*0.1, rot[1, 1]*0.1, rot[2, 1]*0.1, 
                             color='g')
                    ax.quiver(pos[0], pos[1], pos[2], 
                             rot[0, 2]*0.1, rot[1, 2]*0.1, rot[2, 2]*0.1, 
                             color='b')
                    
                    # 绘制相机到标定板的连线
                    ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0], 
                           color=colors[i % len(colors)], linestyle='--', alpha=0.5)
            
            # 如果提供了立体关系，绘制两个相机之间的连线
            if stereo_relationship:
                ref_cam = stereo_relationship['reference_camera']
                other_cam = stereo_relationship['other_camera']
                
                if ref_cam in camera_poses and other_cam in camera_poses:
                    ref_pos = np.array(camera_poses[ref_cam]['camera_to_checkerboard'][:3, 3])
                    other_pos = np.array(camera_poses[other_cam]['camera_to_checkerboard'][:3, 3])
                    
                    # 绘制相机之间的连线
                    ax.plot([ref_pos[0], other_pos[0]], 
                           [ref_pos[1], other_pos[1]], 
                           [ref_pos[2], other_pos[2]], 
                           'g-', linewidth=2, label='Stereo Baseline')
            
            # 绘制理想相机位置（如果提供）
            if desired_poses:
                for i, (cam_name, pose) in enumerate(desired_poses.items()):
                    if 'position' in pose:
                        pos = pose['position']
                        ax.scatter(pos[0], pos[1], pos[2], color=colors[i % len(colors)], 
                                  s=80, alpha=0.3, label=f"{cam_name} (Desired)")
                        
                        # 如果有当前位置，绘制从当前位置到理想位置的向量
                        if cam_name in camera_poses and 'camera_to_checkerboard' in camera_poses[cam_name]:
                            current_pos = camera_poses[cam_name]['camera_to_checkerboard'][:3, 3]
                            ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                                     pos[0]-current_pos[0], pos[1]-current_pos[1], pos[2]-current_pos[2],
                                     color=colors[i % len(colors)], alpha=0.5, 
                                     label=f"{cam_name} Adjustment")
            
            # 设置图形属性
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('相机位置可视化')
            ax.legend()
            
            # 设置轴比例相同
            max_range = max([
                np.max([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]) - 
                np.min([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
            ])
            mid_x = np.mean(ax.get_xlim())
            mid_y = np.mean(ax.get_ylim())
            mid_z = np.mean(ax.get_zlim())
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            # 保存图形
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(CALIB_DIR, f"camera_visualization_{timestamp}.png")
            plt.savefig(fig_path)
            
            # 显示图形
            plt.show()
            
            print(f"相机位置可视化已保存到: {fig_path}")
            
        except ImportError:
            print("无法导入可视化所需的库 (matplotlib)")
        except Exception as e:
            print(f"可视化相机设置时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def run_calibration(self):
        """
        运行完整的相机校准流程
        """
        # 1. 初始化RealSense相机
        print("初始化RealSense相机...")
        pipelines = self.initialize_realsense_cameras()
        
        if not pipelines or len(pipelines) != 2:
            print("相机初始化失败，无法继续校准")
            return False
        
        try:
            # 2. 捕获标定图像
            print("\n开始捕获标定图像")
            print("请将标定板放在两个相机都能看到的位置")
            print("按's'保存图像，按'q'或ESC退出")
            calibration_data = self.capture_calibration_images(pipelines)
            
            # 3. 校准相机
            print("\n开始相机校准...")
            camera_parameters = self.calibrate_cameras(calibration_data)
            
            # 4. 估计相机位姿
            print("\n估计相机位姿...")
            camera_poses = self.estimate_camera_poses(calibration_data, camera_parameters)
            
            # 5. 计算立体相机关系
            print("\n计算立体相机关系...")
            stereo_relationship = self.compute_stereo_relationship(camera_poses)
            
            # 6. 计算所需的相机变换（可选）
            desired_poses = None
            adjustment_guides = None
            if 'sensor_cam_eye_pos' in self.config and 'human_cam_eye_pos' in self.config:
                print("\n计算目标相机位姿...")
                desired_poses = self.compute_desired_transforms(camera_parameters)
                
                # 7. 生成相机调整指南（可选）
                if desired_poses:
                    print("\n生成相机调整指南...")
                    adjustment_guides = self.generate_camera_adjustment(camera_poses, desired_poses)
            
            # 8. 保存结果
            print("\n保存校准结果...")
            result_file = self.save_calibration_results(
                camera_parameters, camera_poses, stereo_relationship, 
                desired_poses, adjustment_guides)
            
            # 9. 可视化相机设置
            print("\n可视化相机设置...")
            self.visualize_camera_setup(camera_poses, stereo_relationship, desired_poses)
            
            print("\n相机校准完成!")
            print(f"详细结果保存在: {result_file}")
            
            if adjustment_guides:
                print("\n请根据调整指南调整相机位置")
            
            return True
            
        except Exception as e:
            print(f"校准过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 停止所有pipeline
            for pipeline in pipelines:
                pipeline.stop()


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='相机外参校准工具')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--load', type=str, help='加载之前的校准结果')
    parser.add_argument('--visualize', action='store_true', help='可视化相机设置')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_arguments()
    
    # 创建校准器实例
    calibrator = CameraCalibrator(config_file=args.config)
    
    # 如果提供了加载文件
    if args.load:
        if os.path.exists(args.load):
            print(f"正在加载校准结果: {args.load}")
            with open(args.load, 'r') as f:
                results = yaml.safe_load(f)
            
            camera_poses = results.get('camera_poses', {})
            stereo_relationship = results.get('stereo_relationship', None)
            desired_poses = results.get('desired_camera_poses', None)
            
            # 可视化相机设置
            if args.visualize:
                calibrator.visualize_camera_setup(camera_poses, stereo_relationship, desired_poses)
                
        else:
            print(f"找不到校准结果文件: {args.load}")
        
        return
    
    # 运行校准
    success = calibrator.run_calibration()
    
    if success:
        print("校准成功完成")


if __name__ == "__main__":
    main()