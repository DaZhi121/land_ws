#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Vector3, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from tf2_geometry_msgs import do_transform_pose

class UWBVisualization:
    def __init__(self):
        rospy.init_node('uwb_coordinate_converter')
        
        # 参数配置
        self.ugv_ns = rospy.get_param("~ugv_namespace", "ugv_0")
        self.input_z = rospy.get_param("~default_z", 0.0)
        
        # TF配置
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 坐标转换矩阵
        self.local_to_world_matrix = None
        self.anchor_positions = {}
        
        # 订阅器/发布器
        rospy.Subscriber("/input_point", Point, self.input_callback)
        self.marker_pub = rospy.Publisher('/visualization/converted_point', Marker, queue_size=1)
        self.point_pub = rospy.Publisher('/output/converted_point', PointStamped, queue_size=1)
        
        # 初始化定时器
        rospy.Timer(rospy.Duration(1.0), self.update_coordinate_system)

    def update_coordinate_system(self, event):
        """定时更新坐标系转换矩阵"""
        try:
            # 获取所有锚点坐标
            new_positions = {}
            for anchor in [f"{self.ugv_ns}/uwb_A{i}" for i in range(4)]:
                try:
                    trans = self.tf_buffer.lookup_transform("world", anchor, rospy.Time(0))
                    new_positions[anchor] = np.array([
                        trans.transform.translation.x,
                        trans.transform.translation.y,
                        trans.transform.translation.z
                    ])
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    continue

            # 检查是否有新数据且结构一致
            if not hasattr(self, 'anchor_positions') or \
            new_positions.keys() != self.anchor_positions.keys():
                needs_update = True
            else:
                # 逐个比较数组是否相同（允许1cm误差）
                needs_update = any(not np.allclose(new_positions[key], self.anchor_positions[key], atol=0.01)
                                for key in new_positions)

            if needs_update:
                self.anchor_positions = new_positions
                self.calculate_transform_matrix()
                
        except Exception as e:
            rospy.logwarn_throttle(5, f"坐标系更新失败: {str(e)}")

    def calculate_transform_matrix(self):
        """计算局部坐标系到世界坐标系的转换矩阵"""
        try:
            # 获取关键锚点坐标
            A0 = self.anchor_positions[f"{self.ugv_ns}/uwb_A0"]
            A1 = self.anchor_positions[f"{self.ugv_ns}/uwb_A1"]
            A3 = self.anchor_positions[f"{self.ugv_ns}/uwb_A3"]
            
            # 计算坐标轴向量
            x_axis = (A1 - A0)[:3]
            y_axis = (A3 - A0)[:3]
            
            # 正交化处理
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis  # Gram-Schmidt
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            
            # 构建4x4变换矩阵
            self.local_to_world_matrix = np.identity(4)
            self.local_to_world_matrix[:3, 0] = x_axis
            self.local_to_world_matrix[:3, 1] = y_axis
            self.local_to_world_matrix[:3, 2] = z_axis
            self.local_to_world_matrix[:3, 3] = A0
            
        except KeyError:
            rospy.logwarn("缺少必要的锚点坐标")
        except np.linalg.LinAlgError:
            rospy.logerr("坐标轴向量线性相关，无法构建正交基底")

    def local_to_world(self, local_point):
        """将局部坐标转换为世界坐标"""
        if self.local_to_world_matrix is None:
            return None
            
        # 添加齐次坐标
        point = np.array([local_point.x, local_point.y, self.input_z, 1.0])
        world_coord = np.dot(self.local_to_world_matrix, point)
        
        # 构造PointStamped消息
        result = PointStamped()
        result.header.stamp = rospy.Time.now()
        result.header.frame_id = "world"
        result.point.x = world_coord[0]
        result.point.y = world_coord[1]
        result.point.z = world_coord[2]
        return result

    def input_callback(self, msg):
        """处理输入点坐标"""
        if self.local_to_world_matrix is None:
            rospy.logwarn_throttle(5, "坐标系未初始化!")
            return
            
        # 坐标转换
        world_point = self.local_to_world(msg)
        if world_point is None:
            return
            
        # 发布转换结果
        self.point_pub.publish(world_point)
        
        # 可视化标记
        marker = Marker()
        marker.header = world_point.header
        marker.ns = "converted_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = world_point.point
        marker.scale = Vector3(0.3, 0.3, 0.3)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # 红色
        marker.lifetime = rospy.Duration(1.0)
        self.marker_pub.publish(marker)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        converter = UWBVisualization()
        converter.run()
    except rospy.ROSInterruptException:
        pass