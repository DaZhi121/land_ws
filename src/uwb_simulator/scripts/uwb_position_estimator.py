#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

class UWBPositionEstimator:
    def __init__(self):
        rospy.init_node('uwb_position_estimator')
        
        # 订阅UWB距离信息
        rospy.Subscriber('/uwb_data', Float32MultiArray, self.uwb_callback)
        
        # 发布估计的无人机位置
        self.position_pub = rospy.Publisher('/uwb_estimated_position', PointStamped, queue_size=10)
        
        # 定义锚点位置（以A0为原点，A0A1为x轴，A0A3为y轴）
        # 假设锚点构成1m×1m的矩形
        self.anchor_positions = np.array([
            [0.0, 0.0, 0.0],  # A0
            [1.0, 0.0, 0.0],  # A1
            [1.0, 1.0, 0.0],  # A2
            [0.0, 1.0, 0.0]   # A3
        ])
        
        rospy.loginfo("UWB位置估计器已启动，等待距离数据...")

    def uwb_callback(self, msg):
        if len(msg.data) != 4:
            rospy.logwarn("接收到无效的距离数据，需要4个距离值")
            return
        
        # 提取四个锚点的距离
        d0, d1, d2, d3 = msg.data
        
        # 使用最小二乘法估计位置
        position = self.estimate_position(d0, d1, d2, d3)
        
        if position is not None:
            # 发布估计的位置
            pos_msg = PointStamped()
            pos_msg.header.stamp = rospy.Time.now()
            pos_msg.header.frame_id = "uwb_local_frame"
            pos_msg.point.x = position[0]
            pos_msg.point.y = position[1]
            pos_msg.point.z = position[2]
            self.position_pub.publish(pos_msg)

    def estimate_position(self, d0, d1, d2, d3):
        """使用最小二乘法估计无人机位置"""
        # 构造系数矩阵A和向量b
        A = []
        b = []
        
        # 对于每个锚点（除了A0），添加方程
        for i in range(1, 4):
            xi, yi, zi = self.anchor_positions[i]
            A.append([2*xi, 2*yi, 2*zi])
            b.append(d0**2 - d1**2 + xi**2 + yi**2 + zi**2)
        
        # 将列表转换为NumPy数组
        A = np.array(A)
        b = np.array(b)
        
        try:
            # 使用最小二乘法求解 (A^T A) x = A^T b
            ATA = np.dot(A.T, A)
            ATb = np.dot(A.T, b)
            position = np.linalg.solve(ATA, ATb)
            return position
        except np.linalg.LinAlgError:
            rospy.logwarn("矩阵奇异，无法求解位置")
            return None

if __name__ == '__main__':
    try:
        estimator = UWBPositionEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass