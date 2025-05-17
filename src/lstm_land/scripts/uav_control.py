#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool

class HybridController:
    def __init__(self):
        rospy.init_node('hybrid_controller', anonymous=True)
        
        # 加载参数
        self.load_parameters()
        
        # 初始化状态
        self.current_pose = PoseStamped()
        self.current_state = State()
        self.target_pose = PoseStamped()
        self.control_mode = "position"  # position/velocity
        self.last_velocity_cmd = TwistStamped()  # 保存最后一次速度指令
        
        # 设置初始目标位置
        self.configure_initial_target()
        
        # 初始化通信接口
        self.init_communication()
        
    def load_parameters(self):
        """加载运行时参数"""
        param_config = {
            'simulation': rospy.get_param('~simulation_mode', True),
            'target_x': rospy.get_param('~target_position/x', 0.0),
            'target_y': rospy.get_param('~target_position/y', 0.0),
            'target_z': rospy.get_param('~target_position/z', 2.0),
            'publish_rate': rospy.get_param('~publish_rate', 20),
            'tolerance': rospy.get_param('~arrival_tolerance', 0.2)
        }
        
        # 参数验证
        if not isinstance(param_config['simulation'], bool):
            rospy.logerr("Invalid simulation mode type, must be boolean")
            rospy.signal_shutdown("Parameter error")
            
        self.__dict__.update(param_config)
        
    def configure_initial_target(self):
        """配置初始目标位置"""
        self.target_pose.pose.position.x = self.target_x
        self.target_pose.pose.position.y = self.target_y
        self.target_pose.pose.position.z = self.target_z
        
    def init_communication(self):
        """初始化ROS通信接口"""
        # 发布器
        self.pos_pub = rospy.Publisher('/uav0/mavros/setpoint_position/local', 
                                      PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/uav0/mavros/setpoint_velocity/cmd_vel', 
                                      TwistStamped, queue_size=10)  
        self.takeoff_complete_pub = rospy.Publisher('/status/takeoff_complete',
            Bool,
            queue_size=1
        )
        
        # 订阅器
        rospy.Subscriber('/uav0/mavros/state', State, self.state_cb)
        rospy.Subscriber('/uav0/mavros/local_position/pose', 
                        PoseStamped, self.pose_cb)
        rospy.Subscriber('/LSTM/cmd_vel',  # 统一订阅速度指令
                        TwistStamped, self.vel_cmd_cb)
        
        # 服务客户端
        self.arming_client = rospy.ServiceProxy('/uav0/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/uav0/mavros/set_mode', SetMode)
        
    def state_cb(self, msg):
        """飞控状态回调"""
        self.current_state = msg
        
    def pose_cb(self, msg):
        """位置信息回调"""
        self.current_pose = msg
        
    def vel_cmd_cb(self, msg):
        """速度指令回调"""
        self.last_velocity_cmd = msg  # 更新最新速度指令
            
    def distance_to_target(self):
        """计算与目标位置的距离"""
        dx = self.current_pose.pose.position.x - self.target_x
        dy = self.current_pose.pose.position.y - self.target_y
        dz = self.current_pose.pose.position.z - self.target_z
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def arm_and_set_mode(self):
        """自动解锁流程"""
        last_request = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - last_request).to_sec() > 5.0:
                break
                
            if self.current_state.mode != "OFFBOARD":
                self.set_mode_client(custom_mode="OFFBOARD")
            elif not self.current_state.armed:
                self.arming_client(True)
            else:
                break
                
            rospy.sleep(0.2)
            
    def run_position_control(self):
        """位置控制主循环"""
        rate = rospy.Rate(self.publish_rate)
        # print("pos")
        # 仿真模式自动解锁
        if self.simulation:
            self.arm_and_set_mode()
            # print("arm")
            
        while not rospy.is_shutdown() and self.control_mode == "position":
            # 持续发布目标位置
            self.target_pose.header.stamp = rospy.Time.now()
            self.pos_pub.publish(self.target_pose)
            self.takeoff_complete_pub.publish(Bool(False))
            # print("send")
            # 检查是否到达目标
            if self.distance_to_target() < self.tolerance:
                rospy.loginfo("Target reached, switching to velocity control mode")
                self.control_mode = "velocity"
                self.pos_pub.unregister()  # 停止位置发布
                break  # 退出位置控制循环
                
            rate.sleep()
            
    def run_velocity_control(self):
        """速度控制主循环"""
        rate = rospy.Rate(self.publish_rate)
        rospy.loginfo("Entering velocity control mode")
        
        while not rospy.is_shutdown() and self.control_mode == "velocity":
            # 发布最后一次接收到的速度指令
            self.last_velocity_cmd.header.stamp = rospy.Time.now()
            self.vel_pub.publish(self.last_velocity_cmd)
            self.takeoff_complete_pub.publish(Bool(True))
            rate.sleep()
            
    def run(self):
        """主运行逻辑"""
        # 等待飞控连接
        while not rospy.is_shutdown() and not self.current_state.connected:
            rospy.sleep(0.1)
            
        # 运行控制模式
        self.run_position_control()
        
        if self.control_mode == "velocity":
            self.run_velocity_control()

if __name__ == '__main__':
    try:
        controller = HybridController()
        controller.run()
    except rospy.ROSInterruptException:
        pass