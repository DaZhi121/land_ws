#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped, Point
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
        
        # 位置差噪声参数
        self.pos_noise_ratio = rospy.get_param("~pos_noise_ratio", 0.02)  # 位置噪声比例
        
        # 打印加载参数
        rospy.loginfo(f"载入参数：安装偏移={self.uwb_installation_offset}，初始偏差={self.uav_initial_offset}，位置噪声比例={self.pos_noise_ratio}")

        # UGV锚点配置
        self.ugv_anchors = [
            f"{self.ugv_ns}/uwb_A0",
            f"{self.ugv_ns}/uwb_A1",
            f"{self.ugv_ns}/uwb_A2",
            f"{self.ugv_ns}/uwb_A3"
        ]
        
        # 存储锚点位置
        self.anchor_positions = {}
        
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
        self.pos_pub = rospy.Publisher('/uwb/pos', PointStamped, queue_size=10)
        
        # 噪声参数
        self.noise_sigma = rospy.get_param("~noise_sigma", 0.05)

    def pose_callback(self, msg):
        """无人机位姿更新回调"""
        self.current_pose = msg.pose

    def get_ugv_anchor_pos(self, anchor_name):
        """获取UGV锚点世界坐标并缓存结果"""
        try:
            # 如果已经缓存过此锚点位置，直接返回
            if anchor_name in self.anchor_positions:
                return self.anchor_positions[anchor_name]
                
            trans = self.tf_buffer.lookup_transform(
                "world",
                anchor_name,
                rospy.Time(0),
                timeout=rospy.Duration(0.5)  # 增加超时时间
            )
            
            # 计算并存储锚点位置
            pos = (
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            )
            self.anchor_positions[anchor_name] = pos
            return pos
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"UGV坐标查询失败: {str(e)}")
            return None

    def calculate_center(self):
        """计算锚点矩形中心坐标"""
        positions = []
        for anchor in self.ugv_anchors:
            pos = self.get_ugv_anchor_pos(anchor)
            if pos is None:
                return None
            positions.append(pos)
            
        # 计算中心点 (四个锚点的平均值)
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        center_z = sum(p[2] for p in positions) / len(positions)
        
        return (center_x, center_y, center_z)

    def publish_position_difference(self, uav_pos, center_pos):
        """发布添加了噪声的无人机到锚点中心的三轴位置差"""
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        # 计算真实三轴位置差
        real_delta_x = center_pos[0] - uav_pos[0]
        real_delta_y = center_pos[1] - uav_pos[1]
        real_delta_z = center_pos[2] - uav_pos[2]
        
        # 计算无人机到中心的真实距离
        center_dist = math.sqrt(real_delta_x**2 + real_delta_y**2 + real_delta_z**2)
        
        # 添加与距离成正比的三轴噪声
        # 位置噪声大小与距离成正比，并考虑噪声比例参数
        pos_noise = self.noise_sigma
        
        noisy_delta_x = real_delta_x + np.random.normal(0, pos_noise)
        noisy_delta_y = real_delta_y + np.random.normal(0, pos_noise)
        noisy_delta_z = real_delta_z + np.random.normal(0, pos_noise)
        
        # # 添加诊断日志（每秒一次）
        # rospy.loginfo_throttle(
        #     1.0, 
        #     f"位置差: 真实({real_delta_x:.2f},{real_delta_y:.2f},{real_delta_z:.2f}) "
        #     f"带噪声({noisy_delta_x:.2f},{noisy_delta_y:.2f},{noisy_delta_z:.2f}) "
        #     f"噪声大小:{pos_noise:.4f}"
        # )
        
        msg.point = Point(noisy_delta_x, noisy_delta_y, noisy_delta_z)
        self.pos_pub.publish(msg)
        
        return (real_delta_x, real_delta_y, real_delta_z, noisy_delta_x, noisy_delta_y, noisy_delta_z)

    def calculate_distances(self):
        """计算距离（包含双偏移）"""
        dist_msg = UWBDistanceArray()
        dist_msg.header.stamp = rospy.Time.now()
        dist_msg.header.frame_id = "world"
        float_msg = Float32MultiArray()
        
        if self.current_pose is None:
            rospy.logwarn_throttle(5, "等待无人机位姿数据...")
            return dist_msg, float_msg
            
        # 应用双重偏移计算最终位置
        uav_pos = (
            self.current_pose.position.x + self.uav_initial_offset[0] + self.uwb_installation_offset[0],
            self.current_pose.position.y + self.uav_initial_offset[1] + self.uwb_installation_offset[1],
            self.current_pose.position.z + self.uav_initial_offset[2] + self.uwb_installation_offset[2]
        )
        
        # 计算锚点矩形中心
        center_pos = self.calculate_center()
        if center_pos:
            # 发布添加噪声后的位置差异
            self.publish_position_difference(uav_pos, center_pos)
        else:
            rospy.logwarn_throttle(5, "无法计算锚点中心，缺少锚点位置数据")
        
        # 遍历所有UGV锚点
        for anchor in self.ugv_anchors:
            ugv_pos = self.get_ugv_anchor_pos(anchor)
            if ugv_pos is None:
                continue
                
            # 计算三维欧氏距离
            dx = uav_pos[0] - ugv_pos[0]
            dy = uav_pos[1] - ugv_pos[1]
            dz = uav_pos[2] - ugv_pos[2]
            true_dist = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # 添加高斯噪声
            noisy_dist = true_dist + np.random.normal(0, self.noise_sigma)
            
            # 构造消息
            dist = UWBDistance()
            dist.id1 = f"{self.uav_ns}/uwb_anchor"
            dist.id2 = anchor
            dist.true_distance = true_dist
            dist.measured_distance = noisy_dist
            dist_msg.distances.append(dist)
            float_msg.data.append(noisy_dist)
        
        return dist_msg, float_msg

    def run(self):
        """主循环"""
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                dist_msg, float_msg = self.calculate_distances()
                if dist_msg.distances:
                    self.dist_pub.publish(dist_msg)
                if float_msg.data:
                    self.uwb_pub.publish(float_msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                break

if __name__ == '__main__':
    rospy.init_node('uwb_monitor')
    UWBMoniTor().run()