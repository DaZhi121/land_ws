#!/usr/bin/env python3
import os
import rospy
import numpy as np
import onnxruntime as ort
from geometry_msgs.msg import TwistStamped,Vector3  # 修改消息类型
from lstm_land.msg import Observation
from scipy.optimize import least_squares
from mavros_msgs.srv import CommandBool, CommandLong, SetMode
import math
from std_msgs.msg import Bool
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from std_msgs.msg import Float32  # 用于发布方差
class DecisionNode:
    def __init__(self):
        rospy.init_node('lstm_decision_maker', anonymous=True)
        
        # 初始化参数
        self.load_parameters()
        
        # 初始化ONNX推理引擎

        self.agent = RecurrentPPO.load(self.model_path)
        self.episode_starts = np.ones((1,), dtype=bool)
        self.terminated = False
        self.lstm_states = None
        # 创建控制指令发布器（修改消息类型）
        self.cmd_pub = rospy.Publisher('/lstm/cmd_vel', 
                                     TwistStamped,  # 使用TwistStamped
                                     queue_size=10)
        
        self.land_complete_pub = rospy.Publisher('/status/land_complete',
            Bool,
            queue_size=1
        )
        self.uwb_variance_pub = rospy.Publisher('/uwb_variance', Float32, queue_size=10)

        self._is_started = 1
        self.lock = 0
        # 订阅观测数据
        rospy.Subscriber('/lstm_land/observation', Observation, self.obs_callback)
        rospy.Subscriber('uwb/center_deltas', Vector3, self.error_callback)
        
        self.set_command_srv = rospy.ServiceProxy('/uav0/mavros/cmd/command', CommandLong)
        self.car_length = rospy.get_param('~car_length', 1.0)
        self.car_width = rospy.get_param('~car_width', 1.0)
        self.safe_altitude = rospy.get_param('~safe_altitude', 0.3)
        self.safe_r = rospy.get_param('~safe_r', 0.2)
       # sta
        # self.height_local = rospy.get_param('~height_local', 1.0)  # 车体坐标系下的目标位置
        # self.target_local_x = rospy.get_param('~target_local_x', 1.2)  # 车体坐标系下的目标位置
        # self.target_local_y = rospy.get_param('~target_local_y', 1.0)  # 车体坐标系下的目标位置
       
       
        # #line circle
        # self.height_local = rospy.get_param('~height_local', 1.0)  # 车体坐标系下的目标位置
        # self.target_local_x = rospy.get_param('~target_local_x', 0.6)  # 车体坐标系下的目标位置
        # self.target_local_y = rospy.get_param('~target_local_y', 0.6)  # 车体坐标系下的目标位置

        self.height_local = rospy.get_param('~height_local', 1.15)  # 车体坐标系下的目标位置
        self.target_local_x = rospy.get_param('~target_local_x', 0.5)  # 车体坐标系下的目标位置
        self.target_local_y = rospy.get_param('~target_local_y', 0.5)  # 车体坐标系下的目标位置


        # 控制参数
        self.control_rate = 20  # Hz
        self.last_obs = None
        self.seq = 0  # 添加消息序列号
        self.yaw = 0
        self.ugv_vx = 0
        self.ugv_vy = 0
        self.latest_uwb = None
        self.range_value = 0
        self.uwb_position = None
        self.uav_vel = np.zeros(3)
        self.h_distance = 0
        self.v_diff = 0
        self.stage = 0
        self.last_error = None


    def error_callback(self, msg):
        self.last_error = np.array([msg.x,msg.y,msg.z-0.2])

    def is_lock(self):

        # 计算UWB布局中心坐标
        center_x = self.car_length / 2
        center_y = self.car_width / 2
        vel_vec = np.linalg.norm(self.uav_vel[0:2])
        vel_ugv = np.linalg.norm(np.array([self.ugv_vx,self.ugv_vy]))
        # 解析无人机坐标
        x= self.uwb_position[0]
        y= self.uwb_position[1]
        z= self.uwb_position[2]
        
        # 计算水平面距离
        
        horizontal_distance = np.linalg.norm(np.array([self.last_error[0],self.last_error[1]]))
        self.h_distance = horizontal_distance
        self.v_diff = abs(vel_ugv-vel_vec)
        
        if(self.h_distance <self.safe_r+0.3 and self.v_diff<0.2 and self.stage == 0):
            self.stage = 1
        if(self.h_distance <self.safe_r+0.1 and self.v_diff<0.1 and self.stage == 1):
            self.stage = 2
        # else:
        #     self.height_local = 1.0
        # if(self.stage == 1):
        #     self.height_local = 0.8
        # if(self.stage == 2):
        #     self.height_local = 0.35

        # 检查高度约束和水平距离
        return (
            horizontal_distance <= self.safe_r and  # 水平距离条件
            0 <= self.last_error[2] <= self.safe_altitude                # 高度范围条件
        )
    
    def force_disarm(self):
        if not self._is_started:
            raise Exception('Not armed')
        
        try:
            self.set_command_srv(
                command = 400,
                confirmation = 0,
                param1 = 0,
                param2 = 21196,
                param3 = 0,
                param4 = 0,
                param5 = 0,
                param6 = 0,
                param7 = 0
            )
            
        except rospy.ServiceException as e:
            rospy.logerr(e)
        

    def uwb_positioning(self):
        L = self.car_length
        W = self.car_width
        H = self.range_value-0.2
        d0 = self.latest_uwb[0]
        d1 = self.latest_uwb[1]
        d2 = self.latest_uwb[2]
        d3 = self.latest_uwb[3]
        # 初始猜测计算（使用A0-A1和A0-A3的测量值）
        x_initial = (d0**2 - d1**2 + L**2) / (2 * L)
        y_initial = (d0**2 - d3**2 + W**2) / (2 * W)
        
        # 约束初始值在合理范围内
        x_initial = np.clip(x_initial, 0, L)
        y_initial = np.clip(y_initial, 0, W)
        
        # 定义残差函数
        def residual(params, L, W, H, d0, d1, d2, d3):
            x, y = params
            # 计算各基站的预测距离
            pred_d0 = np.sqrt(x**2 + y**2 + H**2)
            pred_d1 = np.sqrt((x-L)**2 + y**2 + H**2)
            pred_d2 = np.sqrt((x-L)**2 + (y-W)**2 + H**2)
            pred_d3 = np.sqrt(x**2 + (y-W)**2 + H**2)
            return [
                pred_d0 - d0,
                pred_d1 - d1,
                pred_d2 - d2,
                pred_d3 - d3
            ]
        
        # 设置优化边界（x∈[0,L], y∈[0,W]）
        bounds = ([0, 0], [L, W])
        
        # 执行最小二乘优化
        result = least_squares(
            residual,
            [x_initial, y_initial],
            args=(L, W, H, d0, d1, d2, d3),
            bounds=bounds
        )
        
        x_opt, y_opt = result.x
        # rospy.loginfo(f"UWB_pos: X={x_opt:.2f}, Y={y_opt:.2f}, Z={H:.2f} m")
        self.uwb_position = np.array([x_opt, y_opt, H])



    def load_parameters(self):
        """加载模型路径配置"""
        self.model_path = rospy.get_param('~model_path', 
                                         '/home/leiyifei/land_ws/src/lstm_land/models/lstm_best_model.zip')
        # 验证模型路径
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file not found: {self.model_path}")
            rospy.signal_shutdown("Invalid model path")

    def obs_callback(self, msg):
        """缓存最新观测数据"""
        self.last_obs = msg
        
        self.yaw = math.atan2(2.0 * (msg.car_quat.w * msg.car_quat.z - msg.car_quat.x * msg.car_quat.y), 
                1.0 - 2.0 * (msg.car_quat.y**2 + msg.car_quat.z**2))
        self.ugv_vx = msg.car_speed.x
        self.ugv_vy = msg.car_speed.y
        self.uav_vel[0] =  msg.last_velocity.x
        self.uav_vel[1] =  msg.last_velocity.y
        self.uav_vel[2] =  msg.last_velocity.z
        self.latest_uwb = msg.uwb_dist
        self.uwb_positioning()
        self.range_value = msg.height
        if len(msg.uwb_dist) > 0:  # 确保有数据
            mean = sum(msg.uwb_dist) / len(msg.uwb_dist)
            variance = sum((x - mean) ** 2 for x in msg.uwb_dist) / len(msg.uwb_dist)
            variance_msg = Float32()
            variance_msg.data = variance
            self.uwb_variance_pub.publish(variance_msg)
        if self.is_lock():
            # 确保self.lock属性已初始化
            if not hasattr(self, 'lock'):
                self.lock = 0
            self.lock = 1  # 设置锁定标志

    def convert_observation(self, msg):
        """将ROS消息转换为维度安全的NumPy数组，并集成增强功能
        
        Args:
            msg: 输入的ROS观测消息
            
        Returns:
            dict: 包含所有观测数据的字典，每个值都是 (1, N) 形状的NumPy数组
        """

        # 原始速度向量
        speed_vector = np.array([msg.car_speed.x, msg.car_speed.y, msg.car_speed.z], 
                            dtype=np.float32)
        


        obs_dict = {
            'height': np.array([msg.height - self.height_local], dtype=np.float32).reshape(1, 1),
            'uwb_dist': np.array(msg.uwb_dist, dtype=np.float32).reshape(1, -1),
            'car_quat': np.array([
                msg.car_quat.x,
                msg.car_quat.y,
                msg.car_quat.z,
                msg.car_quat.w
            ], dtype=np.float32).reshape(1, 4),
            'car_speed': speed_vector.reshape(1, 3),
            'last_velocity': np.array([
                msg.last_velocity.x,
                msg.last_velocity.y,
                msg.last_velocity.z
            ], dtype=np.float32).reshape(1, 3),
            'target_local': np.array([
                self.target_local_x,
                self.target_local_y
            ], dtype=np.float32).reshape(1, 2),
        }

        return obs_dict
    
    def publish_action(self, action):
        """将动作转换为ROS控制指令（更新为TwistStamped）"""
        cmd = TwistStamped()
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        # 填充头部信息
        cmd.header.seq = self.seq
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = ""  # 根据实际情况设置坐标系
        action = action * 0.65

        vx_world = action[0] * cos_yaw - (action[1]) * sin_yaw+self.ugv_vx
        vy_world = action[0] * sin_yaw + (action[1]) * cos_yaw+self.ugv_vy
        cmd.twist.linear.x = vx_world  # 前向速度
        cmd.twist.linear.y = vy_world  # 横向速度
        cmd.twist.linear.z = np.clip(action[2]*0.8, -2.5, 2.5)
        # cmd.twist.linear.z = action[2]
        # 如果有角速度需求可添加
        # cmd.twist.angular.z = action[3] 
        rospy.loginfo(f"控制指令: X={cmd.twist.linear.x:.2f}, Y={cmd.twist.linear.y:.2f}, Z={cmd.twist.linear.z:.2f} m/s,{self.h_distance:.2f},{self.v_diff:.2f},{self.height_local:.2f}")
        
        
        self.cmd_pub.publish(cmd)
        self.seq += 1  # 更新序列号

    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.last_obs is None:
                rate.sleep()
                continue
                

            # 转换观测数据
            obs_dict = self.convert_observation(self.last_obs)
            # print(obs_dict)
            # 执行推理
            
            action, self.lstm_states = self.agent.predict(observation=obs_dict, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
            self.episode_starts = self.terminated
            if(self.lock==1):
                self.land_complete_pub.publish(Bool(True))
                self.force_disarm()
            # 发布控制指令

            elif(self.lock==0):
                self.land_complete_pub.publish(Bool(False))  
            self.publish_action(action[0])
                
            # except Exception as e:
            #     rospy.logerr_throttle(1, f"Decision error: {str(e)}")
            
            rate.sleep()




if __name__ == '__main__':
    try:
        node = DecisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass