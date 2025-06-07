#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from tf2_geometry_msgs import do_transform_pose

class UWBVisualization:
    def __init__(self):
        # 初始化节点参数
        rospy.init_node('uwb_visualization_node', anonymous=True)
        
        # 配置参数
        self.ugv_ns = rospy.get_param("~ugv_namespace", "ugv")
        self.uav_frame_id = rospy.get_param("~uav_frame_id", "uav/base_link")
        self.traj_length = rospy.get_param("~trajectory_length", 100)
        self.marker_scale = 0.2
        self.line_width = 0.05

        # TF配置
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 无人机数据存储
        self.current_uav_pose = None
        self.trajectory = []

        # 订阅者
        rospy.Subscriber("/uav0/mavros/local_position/odom", Odometry, self.uav_odom_cb)

        # 发布者
        self.anchor_pub = rospy.Publisher('/visualization/anchors', MarkerArray, queue_size=1)
        self.uav_pub = rospy.Publisher('/visualization/uav', MarkerArray, queue_size=1)

        # UGV锚点列表
        self.ugv_anchors = [
            f"{self.ugv_ns}/uwb_A0",
            f"{self.ugv_ns}/uwb_A1",
            f"{self.ugv_ns}/uwb_A2",
            f"{self.ugv_ns}/uwb_A3"
        ]

        # 定时器
        rospy.Timer(rospy.Duration(0.1), self.visualization_callback)

        self.local_to_world_matrix = None  # 4x4转换矩阵
        # self.init_coordinate_system()

    def uav_odom_cb(self, msg):
        try:
            # 创建PoseStamped对象
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = msg.pose.pose

            # 转换到世界坐标系
            transform = self.tf_buffer.lookup_transform(
                "world",
                msg.header.frame_id,
                rospy.Time(0),
                timeout=rospy.Duration(0.1)
            )
            
            # 转换位姿
            pose_transformed = do_transform_pose(pose_stamped, transform)
            
            # 更新当前位姿
            self.current_uav_pose = pose_transformed.pose.position
            
            # 更新轨迹
            self.trajectory.append(pose_transformed.pose.position)
            if len(self.trajectory) > self.traj_length:
                self.trajectory.pop(0)
                
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF转换失败: {str(e)}")

    def get_anchor_positions(self):
        """获取所有锚点的世界坐标"""
        positions = []
        for anchor in self.ugv_anchors:
            try:
                trans = self.tf_buffer.lookup_transform(
                    "world", 
                    anchor,
                    rospy.Time(0),
                    timeout=rospy.Duration(0.1))
                positions.append((
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ))
            except:
                continue
        return positions

    def init_coordinate_system(self):
        """初始化局部坐标系转换矩阵"""
        # 获取关键锚点坐标
        A = self.get_anchor_positions()
        if len(self.get_anchor_positions()) >= 4:
            A0 = A[0]
            A1 = A[1]
            A3 = A[2]

        # 计算坐标系轴向量
        x_axis = np.array(A1) - np.array(A0)
        y_axis = np.array(A3) - np.array(A0)
        
        # 正交化处理
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis  # Gram-Schmidt
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        # 构建4x4变换矩阵
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        translation = np.array(A0)
        
        self.local_to_world_matrix = np.identity(4)
        self.local_to_world_matrix[:3, :3] = rotation
        self.local_to_world_matrix[:3, 3] = translation

    def local_to_world(self, local_point):
        """将局部坐标转换为世界坐标"""
        if self.local_to_world_matrix is None:
            rospy.logwarn("坐标系未初始化!")
            return None
            
        # 添加齐次坐标
        point = np.append(np.array(local_point), 1.0)
        world_coord = np.dot(self.local_to_world_matrix, point)
        return world_coord[:3]  # 返回(x,y,z)


    def create_anchor_markers(self):
        """生成锚点可视化标记"""
        marker_array = MarkerArray()
        
        # 锚点球体
        for i, pos in enumerate(self.get_anchor_positions()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "uwb_anchors"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(*pos)
            marker.scale = Vector3(self.marker_scale, self.marker_scale, self.marker_scale)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # 绿色
            marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(marker)

        # 锚点连线
        if len(self.get_anchor_positions()) >= 4:
            line_marker = Marker()
            line_marker.header.frame_id = "world"
            line_marker.ns = "anchor_lines"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = self.line_width
            line_marker.color = ColorRGBA(1.0, 0.5, 0.0, 0.6)  # 橙色
            line_marker.points = [Point(*pos) for pos in self.get_anchor_positions()]
            line_marker.points.append(line_marker.points[0])  # 闭合形状
            line_marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(line_marker)

        return marker_array

    def create_uav_markers(self):
        """生成无人机相关标记"""
        marker_array = MarkerArray()
        if not self.current_uav_pose:
            return marker_array

        # 无人机当前位置
        pose_marker = Marker()
        pose_marker.header.frame_id = "world"
        pose_marker.ns = "uav_position"
        pose_marker.id = 0
        pose_marker.type = Marker.SPHERE
        pose_marker.pose.position = self.current_uav_pose
        pose_marker.scale = Vector3(0.3, 0.3, 0.3)
        pose_marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)  # 蓝色
        marker_array.markers.append(pose_marker)

        # 飞行轨迹
        if len(self.trajectory) >= 2:
            traj_marker = Marker()
            traj_marker.header.frame_id = "world"
            traj_marker.ns = "uav_trajectory"
            traj_marker.id = 0
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.scale.x = 0.05
            traj_marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.6)  # 半透明蓝色
            traj_marker.points = self.trajectory.copy()
            marker_array.markers.append(traj_marker)

        # 无人机-锚点连线
        anchor_positions = self.get_anchor_positions()
        if anchor_positions and self.current_uav_pose:
            line_marker = Marker()
            line_marker.header.frame_id = "world"
            line_marker.ns = "uav_connections"
            line_marker.id = 0
            line_marker.type = Marker.LINE_LIST
            line_marker.scale.x = self.line_width
            line_marker.color = ColorRGBA(1.0, 0.5, 0.0, 0.4)  # 半透明橙色
            
            for anchor_pos in anchor_positions:
                line_marker.points.append(self.current_uav_pose)
                line_marker.points.append(Point(*anchor_pos))
            
            marker_array.markers.append(line_marker)

        # 设置统一生命周期
        for m in marker_array.markers:
            m.header.stamp = rospy.Time.now()
            m.lifetime = rospy.Duration(0.5)

        return marker_array

    def visualization_callback(self, event):
        """定时可视化回调"""
        try:
            # 发布锚点
            self.anchor_pub.publish(self.create_anchor_markers())
            
            # 发布无人机信息
            self.uav_pub.publish(self.create_uav_markers())
            
        except Exception as e:
            rospy.logerr(f"可视化异常: {str(e)}")

if __name__ == '__main__':
    try:
        visualizer = UWBVisualization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass