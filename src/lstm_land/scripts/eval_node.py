#!/usr/bin/env python3
import rospy
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, PointStamped
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图支持
class DynamicLandingEvaluator:
    def __init__(self):
        rospy.init_node('dynamic_landing_evaluator')

        # 初始化存储路径
        self.result_dir = os.path.join(os.path.expanduser("~"), "landing_evaluation")
        os.makedirs(self.result_dir, exist_ok=True)

        # 系统参数
        self.buffer_size = 50  # 数据缓冲队列大小
        self.traj_window = 7   # 轨迹平滑窗口

        # 系统状态
        self.takeoff_time = None
        self.land_time = None
        self.recording = False
        self.target = None

        # 数据存储
        self.uav_buffer = []
        self.ugv_buffer = []
        self.uav_traj = []
        self.ugv_traj = []
        self.timestamps = []
        self.traj_yaw = []

        # 评价指标
        self.metrics = {
            'success': False,
            'avg_pos_err': 0.0,
            'max_pos_err': 0.0,
            'term_pos_err': 0.0,
            'avg_vel_err': 0.0,
            'term_vel_err': 0.0,
            'avg_yaw_change': 0.0,
            'max_yaw_change': 0.0,
            'curvature': 0.0,
            'landing_time': 0.0
        }

        # ROS订阅器
        rospy.Subscriber('/status/takeoff_complete', Bool, self.takeoff_cb)
        rospy.Subscriber('/status/land_complete', Bool, self.land_cb)
        rospy.Subscriber('/ugv_0/odom', Odometry, self.ugv_odom_cb)
        rospy.Subscriber('/uav0/mavros/local_position/odom', Odometry, self.uav_odom_cb)
        rospy.Subscriber('/output/converted_point', PointStamped, self.target_cb)

    def takeoff_cb(self, msg):
        if msg.data and not self.takeoff_time:
            self.takeoff_time = rospy.Time.now().to_sec()
            self.recording = True
            rospy.loginfo("🚁 检测到起飞，开始记录数据...")

    def target_cb(self, msg):
        self.target = msg
        # rospy.loginfo(f"🎯 目标点更新：({msg.point.x:.2f}, {msg.point.y:.2f})")

    def land_cb(self, msg):
        if msg.data and not self.land_time:
            self.land_time = rospy.Time.now().to_sec()
            self.recording = False
            self.calculate_metrics()
            self.save_results()
            self.visualize_results()
            rospy.loginfo("评估完成，结果已保存")
            rospy.signal_shutdown("评估结束")

    def ugv_odom_cb(self, msg):
        if self.recording:
            current_time = rospy.Time.now().to_sec()
            self.ugv_buffer.append({
                'time': current_time,
                'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
                'velocity': (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            })
            self.ugv_traj.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def uav_odom_cb(self, msg):
        if self.recording:
            current_time = rospy.Time.now().to_sec()
            self.uav_buffer.append({
                'time': current_time,
                'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
                'velocity': (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            })
            self.uav_traj.append([msg.pose.pose.position.x+3, msg.pose.pose.position.y, msg.pose.pose.position.z])
            self.timestamps.append(current_time)

    def calculate_trajectory_yaw(self):
        """通过轨迹微分计算航向角"""
        self.traj_yaw = []
        if len(self.uav_traj) < 2:
            return

        # 轨迹平滑处理
        smoothed = []
        for i in range(len(self.uav_traj)):
            start = max(0, i - self.traj_window)
            end = min(len(self.uav_traj), i + self.traj_window + 1)
            window = self.uav_traj[start:end]
            avg_x = np.mean([p[0] for p in window])
            avg_y = np.mean([p[1] for p in window])
            smoothed.append((avg_x, avg_y))

        # 计算航向变化
        prev_point = smoothed[0]
        for point in smoothed[1:]:
            dx = point[0] - prev_point[0]
            dy = point[1] - prev_point[1]
            
            if dx == 0 and dy == 0:
                self.traj_yaw.append(self.traj_yaw[-1] if self.traj_yaw else 0.0)
                continue
                
            yaw = math.atan2(dy, dx)
            
            # 处理角度跳变
            if self.traj_yaw:
                last_yaw = self.traj_yaw[-1]
                delta = yaw - last_yaw
                if abs(delta) > math.pi:
                    yaw -= 2*math.pi * np.sign(delta)
            
            self.traj_yaw.append(yaw)
            prev_point = point

        # 填充第一个值
        if len(self.traj_yaw) < len(smoothed):
            self.traj_yaw.insert(0, self.traj_yaw[0])

    def find_nearest_ugv_data(self, uav_time):
        """寻找时间最近的UGV数据"""
        if not self.ugv_buffer:
            return None
            
        # 二分查找最近时间戳
        times = [d['time'] for d in self.ugv_buffer]
        idx = np.searchsorted(times, uav_time, side='left')
        
        if idx == 0:
            return self.ugv_buffer[0]
        elif idx == len(times):
            return self.ugv_buffer[-1]
        else:
            before = times[idx-1]
            after = times[idx]
            return self.ugv_buffer[idx] if after - uav_time < uav_time - before else self.ugv_buffer[idx-1]

    def calculate_metrics(self):
        """计算所有评估指标"""
        try:
            # 基础时间指标
            self.metrics['landing_time'] = self.land_time - self.takeoff_time
            self.metrics['pos_errors'] = []
            self.metrics['vel_errors'] = []
            self.metrics['curvatures'] = []
            self.metrics['yaw_change'] = []
            
            # 计算轨迹航向
            self.calculate_trajectory_yaw()

            # 初始化数据容器
            pos_errors = []
            vel_errors = []
            yaw_changes = []
            curvatures = []

            # 处理每个数据点
            for i, uav_data in enumerate(self.uav_buffer):
                # 位置误差计算（添加多层空值判断）
                if self.target and hasattr(self.target, 'point'):
                    target_x = getattr(self.target.point, 'x', None)
                    target_y = getattr(self.target.point, 'y', None)
                    uav_x = uav_data['position'][0] + 3.0 if uav_data['position'] else None
                    uav_y = uav_data['position'][1] if uav_data['position'] else None
                    
                    if None not in [target_x, target_y, uav_x, uav_y]:
                        dx = uav_x - target_x
                        dy = uav_y - target_y
                        pos_errors.append(math.hypot(dx, dy))
                        self.metrics['pos_errors'].append(math.hypot(dx, dy))

                # 速度误差计算（添加空值保护）
                ugv_data = self.find_nearest_ugv_data(uav_data['time'])
                if ugv_data and 'velocity' in ugv_data and 'velocity' in uav_data:
                    try:
                        ugv_vx = float(ugv_data['velocity'][0])
                        ugv_vy = float(ugv_data['velocity'][1])
                        uav_vx = float(uav_data['velocity'][0])
                        uav_vy = float(uav_data['velocity'][1])
                        dvx = uav_vx - ugv_vx
                        dvy = uav_vy - ugv_vy
                        vel_errors.append(math.hypot(dvx, dvy))
                        self.metrics['vel_errors'].append(math.hypot(dvx, dvy))
                    except (TypeError, IndexError) as e:
                        rospy.logwarn(f"速度数据格式异常: {str(e)}")

                # 航向变化率计算（添加索引保护）
                if i > 0 and i < len(self.traj_yaw):
                    try:
                        current_yaw = float(self.traj_yaw[i])
                        prev_yaw = float(self.traj_yaw[i-1])
                        delta = abs(current_yaw - prev_yaw)
                        yaw_changes.append(min(delta, 2*math.pi - delta))
                        self.metrics['yaw_change'].append(min(delta, 2*math.pi - delta))
                    except (TypeError, IndexError) as e:
                        rospy.logwarn(f"航向数据异常: {str(e)}")

            # 曲率计算（添加范围检查）
            if len(self.traj_yaw) >= 3:
                for i in range(1, len(self.traj_yaw)-1):
                    try:
                        yaw_prev = float(self.traj_yaw[i-1])
                        yaw_curr = float(self.traj_yaw[i])
                        yaw_next = float(self.traj_yaw[i+1])
                        curv = abs(yaw_next - 2*yaw_curr + yaw_prev)
                        curvatures.append(curv)
                        self.metrics['curvatures'].append(curv)
                    except (TypeError, IndexError) as e:
                        rospy.logwarn(f"曲率计算异常: {str(e)}")

            # 更新指标（添加空列表保护）
            if pos_errors:
                self.metrics.update({
                    'avg_pos_err': np.mean(pos_errors) if pos_errors else 0.0,
                    'max_pos_err': np.max(pos_errors) if pos_errors else 0.0,
                    'term_pos_err': pos_errors[-1] if pos_errors else 0.0
                })
            else:
                rospy.logwarn("位置误差数据为空，可能目标点未设置")

            if vel_errors:
                self.metrics.update({
                    'avg_vel_err': np.mean(vel_errors) if vel_errors else 0.0,
                    'term_vel_err': vel_errors[-1] if vel_errors else 0.0
                })

            if yaw_changes:
                self.metrics.update({
                    'avg_yaw_change': np.degrees(np.mean(yaw_changes)) if yaw_changes else 0.0,
                    'max_yaw_change': np.degrees(np.max(yaw_changes)) if yaw_changes else 0.0
                })

            if curvatures:
                self.metrics['curvature'] = np.mean(curvatures) if curvatures else 0.0

            # 成功条件判定（添加默认值）
            success_conds = [
                self.metrics.get('term_pos_err', 999) < 0.5,
                self.metrics.get('term_vel_err', 999) < 0.3,
                self.metrics.get('landing_time', 999) < 60,
                self.metrics.get('max_pos_err', 999) < 1.5,
                self.metrics.get('max_yaw_change', 999) < 30.0
            ]
            self.metrics['success'] = all(success_conds)

        except Exception as e:
            rospy.logerr(f"指标计算错误: {str(e)}")
            # 初始化安全值
            self.metrics.update({
                'success': False,
                'avg_pos_err': -1,
                'max_pos_err': -1,
                'term_pos_err': -1,
                'avg_vel_err': -1,
                'term_vel_err': -1,
                'avg_yaw_change': -1,
                'max_yaw_change': -1,
                'curvature': -1
            })

    def save_results(self):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存文本报告
        report_path = os.path.join(self.result_dir, f"report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("无人机动态着陆评估报告\n")
            f.write("==========================\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"着陆结果: {'成功' if self.metrics['success'] else '失败'}\n\n")
            # 只保存标量指标
            valid_metrics = {
                k: v for k, v in self.metrics.items()
                if not isinstance(v, (list, dict))  # 过滤所有非标量数据
            }
            f.write("关键指标:\n")
            for k, v in valid_metrics.items():
                if k == 'success':
                    continue  # 已单独处理
                f.write(f"{k.replace('_', ' ').title():<20}: {v:.4f}\n")

        # 保存原始数据
        data_path = os.path.join(self.result_dir, f"raw_data_{timestamp}.npz")
        np.savez(data_path,
                 timestamps=np.array(self.timestamps),
                 uav_traj=np.array(self.uav_traj),
                 ugv_traj=np.array(self.ugv_traj),
                 traj_yaw=np.array(self.traj_yaw),
                 target_pos=np.array([self.target.point.x, self.target.point.y]) if self.target else None)

    def visualize_results(self):
        """生成可视化图表"""
        plt.figure(figsize=(18, 12))
        
        # 轨迹对比图
        ax = plt.subplot(2, 2, 1,projection='3d')
        uav_x = [p[0] for p in self.uav_traj]
        uav_y = [p[1] for p in self.uav_traj]
        uav_z = [p[2] for p in self.uav_traj]
        ugv_x = [p[0] for p in self.ugv_traj]
        ugv_y = [p[1] for p in self.ugv_traj]
        ugv_z = [p[2] for p in self.ugv_traj]
        
        ax.plot(uav_x, uav_y, uav_z, 'b-', label='uav')
        ax.plot(ugv_x, ugv_y, ugv_z, 'r--', label='ugv')
        # 绘制目标点（使用scatter方法）
        if self.target:
            ax.scatter(
                [self.target.point.x], 
                [self.target.point.y], 
                [self.target.point.z], 
                c='g', 
                marker='*', 
                s=200,  # 控制标记大小
                label='Target'
        )
        # ax.axis('equal')
        plt.title("Landing trajectory")
        ax.legend()



        # 误差变化图
        plt.subplot(2, 2, 2)
        time_axis = np.array(self.timestamps) - self.takeoff_time
        plt.plot(time_axis, self.metrics.get('pos_errors', []), label='position error')
        plt.xlabel("time (s)")
        plt.ylabel("m")
        plt.title("position error")
        plt.legend()
        plt.grid(True)

        # 误差变化图
        plt.subplot(2, 2, 3)
        time_axis = np.array(self.timestamps) - self.takeoff_time
        plt.plot(time_axis, self.metrics.get('vel_errors', []), label='velocity error')
        plt.xlabel("time (s)")
        plt.ylabel("m/s")
        plt.title("velocity error")
        plt.legend()
        plt.grid(True)

        # 误差变化图
        plt.subplot(2, 2, 4)
        time_axis = np.array(self.timestamps) - self.takeoff_time
        plt.plot(time_axis[1:], self.metrics.get('yaw_change', []), label='yaw_change')
        plt.xlabel("time (s)")
        plt.ylabel("rad/s")
        plt.title("yaw_change")
        plt.legend()
        plt.grid(True)


        # 保存图表
        plot_path = os.path.join(self.result_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

if __name__ == '__main__':
    try:
        evaluator = DynamicLandingEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass