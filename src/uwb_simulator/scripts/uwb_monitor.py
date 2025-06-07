#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped, Vector3
from uwb_simulator.msg import UWBDistanceArray, UWBDistance
from std_msgs.msg import Float32MultiArray

class UWBMoniTor:
    def __init__(self):
        # 初始化命名空间参数
        self.ugv_ns = rospy.get_param("~ugv_namespace", "ugv_0")
        self.uav_ns = rospy.get_param("~uav_namespace", "uav1")
        
        # 初始化偏差参数
        self.uwb_installation_offset = rospy.get_param(
            "~uwb_installation_offset", [0.2, 0.0, -0.1])  # UWB模块安装偏移 (x,y,z)
        self.uav_initial_offset = rospy.get_param(
            "~uav_initial_offset", [8.0, 0.0, 0.0])        # 无人机初始位置偏差
        
        # 打印加载参数
        rospy.loginfo(f"载入参数：安装偏移={self.uwb_installation_offset}，初始偏差={self.uav_initial_offset}")

        # UGV锚点配置
        self.ugv_anchors = [
            f"{self.ugv_ns}/uwb_A0",
            f"{self.ugv_ns}/uwb_A1",
            f"{self.ugv_ns}/uwb_A2",
            f"{self.ugv_ns}/uwb_A3"
        ]
        
        # TF2初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 无人机位姿订阅
        self.current_pose = None
        rospy.Subscriber(
            f"/{self.uav_ns}/mavros/local_position/pose",
            PoseStamped,
            self.pose_callback
        )
        
        # 发布器初始化
        self.dist_pub = rospy.Publisher('/uwb/distances', UWBDistanceArray, queue_size=10)
        self.uwb_pub = rospy.Publisher('/uwb_data', Float32MultiArray, queue_size=10)
        self.delta_pub = rospy.Publisher('/uwb/deltas', Float32MultiArray, queue_size=10)  # 新增的差值发布器
        self.delta_vec_pub = rospy.Publisher('/uwb/delta_vec', Vector3, queue_size=10)    # 新增的Vector3发布器
        self.center_pub = rospy.Publisher('/uwb/center_distance', UWBDistance, queue_size=10)
        self.center_delta_pub = rospy.Publisher('/uwb/center_deltas', Vector3, queue_size=10)
        
        # 噪声参数
        self.noise_sigma = rospy.get_param("~noise_sigma", 0.05)

    def get_anchor_center(self):
        """计算四个anchor的中心点坐标"""
        anchor_positions = []
        for anchor in self.ugv_anchors:
            pos = self.get_ugv_anchor_pos(anchor)
            if pos is not None:
                anchor_positions.append(pos)
        
        if len(anchor_positions) < 4:
            rospy.logwarn("无法获取全部4个anchor的位置")
            return None
        
        # 计算四个anchor的中心点
        center_x = sum(pos[0] for pos in anchor_positions) / 4
        center_y = sum(pos[1] for pos in anchor_positions) / 4
        center_z = sum(pos[2] for pos in anchor_positions) / 4
        
        return (center_x, center_y, center_z)

    def pose_callback(self, msg):
        """无人机位姿更新回调"""
        self.current_pose = msg.pose

    def get_ugv_anchor_pos(self, anchor_name):
        """获取UGV锚点世界坐标"""
        try:
            trans = self.tf_buffer.lookup_transform(
                "world",
                anchor_name,
                rospy.Time(0),
                timeout=rospy.Duration(0.1)
            )
            return (
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            )
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"UGV坐标查询失败: {str(e)}")
            return None

    def calculate_distances(self):
        """计算距离（包含双偏移）"""
        dist_msg = UWBDistanceArray()
        dist_msg.header.stamp = rospy.Time.now()
        float_msg = Float32MultiArray()
        delta_msg = Float32MultiArray()  # 新增的差值消息
        delta_vec_msg = Vector3()        # 新增的Vector3消息
        
        if self.current_pose is None:
            rospy.logwarn_throttle(5, "等待无人机位姿数据...")
            return dist_msg, float_msg, delta_msg, delta_vec_msg
            
        # 应用双重偏移计算最终位置
        uav_pos = (
            self.current_pose.position.x + self.uav_initial_offset[0] + self.uwb_installation_offset[0],
            self.current_pose.position.y + self.uav_initial_offset[1] + self.uwb_installation_offset[1],
            self.current_pose.position.z + self.uav_initial_offset[2] + self.uwb_installation_offset[2]
        )
        
        # 遍历所有UGV锚点
        for anchor in self.ugv_anchors:
            ugv_pos = self.get_ugv_anchor_pos(anchor)
            if ugv_pos is None:
                continue
                
            # 计算三维欧氏距离和差值
            dx = uav_pos[0] - ugv_pos[0]-0.5
            dy = uav_pos[1] - ugv_pos[1]-0.5
            dz = uav_pos[2] - ugv_pos[2]
            true_dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # 添加高斯噪声
            noisy_dist = true_dist + np.random.normal(0, self.noise_sigma)
            
            # 对差值也添加相同大小的噪声
            noisy_dx = dx + np.random.normal(0, self.noise_sigma)
            noisy_dy = dy + np.random.normal(0, self.noise_sigma)
            noisy_dz = dz + np.random.normal(0, self.noise_sigma)
            
            # 构造消息
            dist = UWBDistance()
            dist.id1 = f"{self.uav_ns}/uwb_anchor"
            dist.id2 = anchor
            dist.true_distance = true_dist
            dist.measured_distance = noisy_dist
            dist_msg.distances.append(dist)
            float_msg.data.append(noisy_dist)
            
            # 添加差值到消息
            delta_msg.data.extend([noisy_dx, noisy_dy, noisy_dz])
            delta_vec_msg.x = noisy_dx
            delta_vec_msg.y = noisy_dy
            delta_vec_msg.z = noisy_dz

        # 计算到中心点的距离
        center_pos = self.get_anchor_center()
        if center_pos is not None:
            # 计算无人机到中心点的dx, dy, dz
            center_dx = uav_pos[0] - center_pos[0]
            center_dy = uav_pos[1] - center_pos[1]
            center_dz = uav_pos[2] - center_pos[2]
            center_dist = math.sqrt(center_dx**2 + center_dy**2 + center_dz**2)
            
            # 添加噪声
            noisy_center_dist = center_dist + np.random.normal(0, self.noise_sigma)
            noisy_center_dx = center_dx + np.random.normal(0, self.noise_sigma)
            noisy_center_dy = center_dy + np.random.normal(0, self.noise_sigma)
            noisy_center_dz = center_dz + np.random.normal(0, self.noise_sigma)
            
            # 构造中心点距离消息
            center_dist_msg = UWBDistance()
            center_dist_msg.id1 = f"{self.uav_ns}/uwb_anchor"
            center_dist_msg.id2 = "anchor_center"
            center_dist_msg.true_distance = center_dist
            center_dist_msg.measured_distance = noisy_center_dist
            
            # 构造中心点差值消息
            center_delta_msg = Vector3()
            center_delta_msg.x = noisy_center_dx
            center_delta_msg.y = noisy_center_dy
            center_delta_msg.z = noisy_center_dz
            
            # 发布消息
            self.center_pub.publish(center_dist_msg)
            self.center_delta_pub.publish(center_delta_msg)
        
        return dist_msg, float_msg, delta_msg, delta_vec_msg

    def run(self):
        """主循环"""
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                dist_msg, float_msg, delta_msg, delta_vec_msg = self.calculate_distances()
                if dist_msg.distances:
                    self.dist_pub.publish(dist_msg)
                if float_msg.data:
                    self.uwb_pub.publish(float_msg)
                if delta_msg.data:  # 发布差值消息
                    self.delta_pub.publish(delta_msg)
                    self.delta_vec_pub.publish(delta_vec_msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                break

if __name__ == '__main__':
    rospy.init_node('uwb_monitor')
    UWBMoniTor().run()