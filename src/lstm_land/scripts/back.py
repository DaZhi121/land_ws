#!/usr/bin/env python3
import rospy
import numpy as np
import math
from collections import deque
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Vector3, PointStamped

class DynamicLandingEvaluator:
    def __init__(self):
        rospy.init_node('dynamic_landing_evaluator')
        
        # 数据同步参数
        self.buffer_size = 30  # 对应约1.5秒数据（假设20Hz频率）
        
        # 实验状态记录
        self.takeoff_time = None
        self.land_time = None
        self.recording = False
        self.target = None
        # 数据缓存队列（使用deque保证线程安全）
        self.uav_buffer = deque(maxlen=self.buffer_size)
        self.ugv_buffer = deque(maxlen=self.buffer_size)
        
        # 评价指标
        self.metrics = {
            'success': False,
            'avg_position_error': 0.0,
            'max_position_error': 0.0,
            'avg_velocity_error': 0.0,
            'terminal_position_error': 0.0,
            'terminal_velocity_error': 0.0,
            'angular_variation': 0.0,
            'landing_time': 0.0,
            'traj_smoothness': 0.0
        }

        # 初始化订阅器
        rospy.Subscriber('/status/takeoff_complete', Bool, self.takeoff_cb)
        rospy.Subscriber('/status/land_complete', Bool, self.land_cb)
        rospy.Subscriber('/ugv_0/odom', Odometry, self.ugv_odom_cb)
        rospy.Subscriber('/uav0/mavros/local_position/odom', Odometry, self.uav_odom_cb)
        rospy.Subscriber('/output/converted_point', PointStamped, self.target_cb)

    def takeoff_cb(self, msg):
        if msg.data and not self.takeoff_time:
            self.takeoff_time = rospy.Time.now().to_sec()
            self.recording = True
            rospy.loginfo("🚁 Takeoff detected, start recording...")


    def target_cb(self,msg):
            self.target = msg

    def land_cb(self, msg):
        if msg.data and not self.land_time:
            self.land_time = rospy.Time.now().to_sec()
            self.recording = False
            self.calculate_metrics()
            self.log_results()
            rospy.signal_shutdown("Evaluation completed")

    def ugv_odom_cb(self, msg):
        if self.recording:
            self.ugv_buffer.append({
                'time': rospy.Time.now().to_sec(),
                'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
                'velocity': (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
            })

    def uav_odom_cb(self, msg):
        if not self.recording:
            return

        current_time = rospy.Time.now().to_sec()
        self.uav_buffer.append({
            'time': current_time,
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y),
            'velocity': (msg.twist.twist.linear.x, msg.twist.twist.linear.y),
            'yaw': self.calculate_yaw(msg.pose.pose.orientation)
        })

    def calculate_yaw(self, orientation):
        """从四元数计算偏航角"""
        q = orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2 + q.z**2))
        return yaw

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
        try:
            # 基础时间指标
            self.metrics['landing_time'] = self.land_time - self.takeoff_time
            
            # 轨迹分析
            position_errors = []
            velocity_errors = []
            yaw_changes = []
            
            for uav_data in self.uav_buffer:
                ugv_data = self.find_nearest_ugv_data(uav_data['time'])
                if not ugv_data:
                    continue
                
                if self.target==None:
                    continue
                # 位置误差
                dx = uav_data['position'][0]+3.0 - self.target.point.x
                dy = uav_data['position'][1] - self.target.point.y
                position_errors.append(math.hypot(dx, dy))
                
                # 速度误差
                dvx = uav_data['velocity'][0] - ugv_data['velocity'][0]
                dvy = uav_data['velocity'][1] - ugv_data['velocity'][1]
                velocity_errors.append(math.hypot(dvx, dvy))
                
                # 航向变化
                if len(yaw_changes) > 0:
                    delta_yaw = abs(uav_data['yaw'] - yaw_changes[-1])
                    yaw_changes.append(delta_yaw)
                else:
                    yaw_changes.append(0.0)
            
            # 统计指标
            if position_errors:
                self.metrics['avg_position_error'] = np.mean(position_errors)
                self.metrics['max_position_error'] = np.max(position_errors)
                self.metrics['terminal_position_error'] = position_errors[-1]
                
            if velocity_errors:
                self.metrics['avg_velocity_error'] = np.mean(velocity_errors)
                self.metrics['terminal_velocity_error'] = velocity_errors[-1]
                
            if yaw_changes:
                self.metrics['angular_variation'] = np.mean(yaw_changes[1:])  # 忽略第一个零值
                self.metrics['traj_smoothness'] = np.std(yaw_changes)
                
            # 成功条件判定
            success_conditions = [
                self.metrics['terminal_position_error'] < 0.5,
                self.metrics['terminal_velocity_error'] < 0.3,
                self.metrics['landing_time'] < 60.0,
                self.metrics['max_position_error'] < 1.5
            ]
            self.metrics['success'] = all(success_conditions)
            
        except Exception as e:
            rospy.logerr(f"Metric calculation error: {str(e)}")

    def log_results(self):
        result_str = f"""
        ========== Dynamic Landing Evaluation ==========
        🎯 Success Status:         {'✅ SUCCESS' if self.metrics['success'] else '❌ FAILURE'}
        📍 Average Position Error: {self.metrics['avg_position_error']:.3f} m
        📍 Max Position Error:     {self.metrics['max_position_error']:.3f} m
        📍 Terminal Position Error:{self.metrics['terminal_position_error']:.3f} m
        🚀 Average Velocity Error: {self.metrics['avg_velocity_error']:.3f} m/s
        🚀 Terminal Velocity Error:{self.metrics['terminal_velocity_error']:.3f} m/s
        🌀 Angular Variation:      {self.metrics['angular_variation']:.3f} rad/step
        🛤️ Trajectory Smoothness:  {self.metrics['traj_smoothness']:.3f} rad
        ⏱️ Total Landing Time:     {self.metrics['landing_time']:.2f} s
        ================================================
        """
        rospy.loginfo(result_str)

if __name__ == '__main__':
    try:
        evaluator = DynamicLandingEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass