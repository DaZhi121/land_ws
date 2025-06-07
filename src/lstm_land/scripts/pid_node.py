#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped,Vector3
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandLong
from std_msgs.msg import Bool, Float32MultiArray
import math
from scipy.optimize import least_squares
from lstm_land.msg import Observation
class PIDController:
    def __init__(self, 
                 kp_pos=np.array([2.5, 2.5, 2.0]),
                 ki_pos=np.array([1.5, 1.5, 0.0]),
                 kd_pos=np.array([0.8, 0.8, 0.1]),
                 kp_vel=np.array([0.5, 0.5, 0.3]),
                 max_pos_integral=np.array([5.0, 5.0, 2.0]),
                 max_output=10.0,
                 dt=0.05):
        # 位置PID参数
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        
        # 速度环P参数
        self.kp_vel = kp_vel
        
        # 积分项存储
        self.pos_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        
        # 限幅参数
        self.max_pos_integral = max_pos_integral
        self.max_output = max_output
        self.dt = dt

    def reset(self):
        """重置控制器状态"""
        self.pos_integral = np.zeros(3)
        self.last_pos_error = np.zeros(3)

    def compute(self, 
               current_pos, 
               target_pos, 
               current_vel=np.zeros(3)):
        """
        双环PID控制计算(位置环+速度环)
        输入:
            current_pos: 当前无人机ENU位置 [x,y,z]
            target_pos: 目标ENU位置 [x,y,z] 
            current_vel: 当前速度 [vx, vy, vz] (可选)
        返回:
            速度控制指令 [vx_cmd, vy_cmd, vz_cmd] (m/s)
        """
        pos_error = target_pos - current_pos
        
        # 位置环PID计算
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(self.pos_integral, -self.max_pos_integral, self.max_pos_integral)
        pos_derivative = (pos_error - self.last_pos_error) / self.dt
        
        # 计算期望速度
        vel_desired = (
            self.kp_pos * pos_error +
            self.ki_pos * self.pos_integral + 
            self.kd_pos * pos_derivative
        )
        
        # 速度环P控制(将速度误差转换为加速度指令)
        vel_error = vel_desired - current_vel
        acc_cmd = self.kp_vel * vel_error
        
        # 输出限幅 (转换为速度指令)
        cmd = np.clip(acc_cmd, -self.max_output, self.max_output)
        
        # 更新上一次误差
        self.last_pos_error = pos_error.copy()
        
        return cmd.astype(np.float32)

class DecisionNode:
    def __init__(self):
        rospy.init_node('lstm_decision_maker', anonymous=True)

        
        # 初始化ONNX推理引擎
        self.agent = PIDController()
        self.agent.reset()
        
        # 创建控制指令发布器（修改消息类型）
        self.cmd_pub = rospy.Publisher('/pid/cmd_vel', 
                                     TwistStamped,  # 使用TwistStamped
                                     queue_size=10)
        
        self.land_complete_pub = rospy.Publisher('/status/land_complete',
            Bool,
            queue_size=1
        )


        self._is_started = 1
        self.lock = 0
        # 订阅观测数据

        rospy.Subscriber('uwb/center_deltas', Vector3, self.error_callback)
        rospy.Subscriber('/lstm_land/observation', Observation, self.obs_callback)
        
        self.set_command_srv = rospy.ServiceProxy('/uav0/mavros/cmd/command', CommandLong)
        self.car_length = rospy.get_param('~car_length', 1.0)
        self.car_width = rospy.get_param('~car_width', 1.0)
        self.safe_altitude = rospy.get_param('~safe_altitude', 0.3)
        self.safe_r = rospy.get_param('~safe_r', 0.2)
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
        self.h_distance = 0
    def is_lock(self):

        # 计算UWB布局中心坐标
        center_x = self.car_length / 2
        center_y = self.car_width / 2
        
        # 解析无人机坐标
        x= self.uwb_position[0]
        y= self.uwb_position[1]
        z= self.uwb_position[2]
        
        # 计算水平面距离
        self.h_distance = np.linalg.norm(np.array([self.last_error[0],self.last_error[1]]))
        # 检查高度约束和水平距离
        return (
            self.h_distance <= self.safe_r and  # 水平距离条件
            0 <= self.last_error[2] <= self.safe_altitude                   # 高度范围条件
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
        H = self.range_value
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

        self.uwb_position = np.array([x_opt, y_opt, H])





    def obs_callback(self, msg):
        """缓存最新观测数据"""
        self.last_obs = msg
        
        self.yaw = math.atan2(2.0 * (msg.car_quat.w * msg.car_quat.z - msg.car_quat.x * msg.car_quat.y), 
                1.0 - 2.0 * (msg.car_quat.y**2 + msg.car_quat.z**2))
        self.ugv_vx = msg.car_speed.x
        self.ugv_vy = msg.car_speed.y
        self.latest_uwb = msg.uwb_dist
        self.uwb_positioning()
        self.range_value = msg.height
        if self.is_lock():
            # 确保self.lock属性已初始化
            if not hasattr(self, 'lock'):
                self.lock = 0
            self.lock = 1  # 设置锁定标志

    def error_callback(self, msg):
        self.last_error = np.array([msg.x,msg.y,msg.z-0.2])



    def publish_action(self, action):
        """将动作转换为ROS控制指令（更新为TwistStamped）"""
        cmd = TwistStamped()

        # 填充头部信息
        cmd.header.seq = self.seq
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = ""  # 根据实际情况设置坐标系


        vx_world = action[0]
        vy_world = action[1]
        cmd.twist.linear.x = vx_world  # 前向速度
        cmd.twist.linear.y = vy_world  # 横向速度
        cmd.twist.linear.z = np.clip(action[2], -0.5, 0.5)
        # 如果有角速度需求可添加
        # cmd.twist.angular.z = action[3] 
        rospy.loginfo(f"控制指令: X={cmd.twist.linear.x:.2f}, Y={cmd.twist.linear.y:.2f}, Z={cmd.twist.linear.z:.2f} m/s, {self.h_distance:.2}")
        
        self.cmd_pub.publish(cmd)
        self.seq += 1  # 更新序列号

    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.last_obs is None:
                rate.sleep()
                continue
                
            try:

                
                # 执行推理
                action = self.agent.compute(np.array([0.0,0.0,0.0]),-self.last_error)
                if(self.lock==1):
                  self.land_complete_pub.publish(Bool(True))
                  self.force_disarm()
                # 发布控制指令

                elif(self.lock==0):
                  self.land_complete_pub.publish(Bool(False))  
                self.publish_action(action)
                
            except Exception as e:
                rospy.logerr_throttle(1, f"Decision error: {str(e)}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = DecisionNode()
        controller.run()
    except rospy.ROSInterruptException:
        pass