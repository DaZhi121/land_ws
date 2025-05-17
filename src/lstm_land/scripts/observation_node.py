#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Header, Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import TwistStamped, Vector3, Quaternion
from lstm_land.msg import Observation

class ObservationNode:
    def __init__(self):
        rospy.init_node('observation_generator', anonymous=True)
        
        # 初始化参数
        self.load_parameters()
        
        # 初始化数据缓存
        self.uwb_dist = np.zeros(4, dtype=np.float32)
        self.ugv_odom = Odometry()
        self.drone_velocity = Vector3()
        self.height = 0.0
        self.height_local = 0.0
        # 创建发布器
        self.obs_pub = rospy.Publisher('/lstm_land/observation', Observation, queue_size=10)
        
        # 订阅者配置
        rospy.Subscriber("/uwb_data", Float32MultiArray, self.uwb_callback)
        rospy.Subscriber("/ugv_0/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/range", Range, self.range_callback)
        rospy.Subscriber("/mavros/local_position/velocity_local",
                        TwistStamped, self.vel_callback)
        
        # 控制参数
        self.publish_rate = 20  # Hz
        self.last_pub_time = rospy.Time.now()

    def load_parameters(self):
        """从参数服务器加载目标位置"""
        try:
            target = rospy.get_param('~target_local')
            # self.height_local = rospy.get_param('height')
            self.target_local = np.array(target, dtype=np.float32)
            
            if len(self.target_local) != 2:
                raise ValueError("Target position must be 2D [x, y]")
        except KeyError:
            rospy.logerr("Missing parameter ~target_local")
            rospy.signal_shutdown("Parameter missing")
            exit(1)

    def uwb_callback(self, msg):
        """处理UWB距离数据"""
        if len(msg.data) == 4:
            self.uwb_dist = np.array(msg.data, dtype=np.float32)
        else:
            rospy.logwarn_throttle(1, "Invalid UWB data dimension")

    def odom_callback(self, msg):
        """处理车辆状态"""
        self.ugv_odom = msg

    def range_callback(self, msg):
        """处理高度计数据"""
        self.height = msg.range

    def vel_callback(self, msg):
        """处理无人机速度"""
        self.drone_velocity = msg.twist.linear

    def compose_observation(self):
        """组合观测数据"""
        obs = Observation()
        
        # Header信息
        obs.header = Header(
            stamp=rospy.Time.now(),
            frame_id="ned"
        )
        
        # UWB距离测量
        obs.uwb_dist = self.uwb_dist.tolist()
        
        # 车辆状态
        obs.car_speed = Vector3(
            x=self.ugv_odom.twist.twist.linear.x,
            y=self.ugv_odom.twist.twist.linear.y,
            z=self.ugv_odom.twist.twist.linear.z
        )
        obs.car_quat = Quaternion(
            x=self.ugv_odom.pose.pose.orientation.x,
            y=self.ugv_odom.pose.pose.orientation.y,
            z=self.ugv_odom.pose.pose.orientation.z,
            w=self.ugv_odom.pose.pose.orientation.w
        )
        
        # 无人机状态
        obs.height = self.height+self.height_local
        obs.last_velocity = self.drone_velocity
        
        # 目标位置
        obs.target_local = self.target_local.tolist()
        
        return obs

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # 控制发布频率
            if (rospy.Time.now() - self.last_pub_time).to_sec() < 1.0/self.publish_rate:
                continue
                
            try:
                obs_msg = self.compose_observation()
                self.obs_pub.publish(obs_msg)
                self.last_pub_time = rospy.Time.now()
            except Exception as e:
                rospy.logerr(f"Error composing observation: {str(e)}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = ObservationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass