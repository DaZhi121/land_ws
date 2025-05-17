#!/usr/bin/env python3
import os
import rospy
import numpy as np
import onnxruntime as ort
from geometry_msgs.msg import TwistStamped  # 修改消息类型
from lstm_land.msg import Observation
import math
class DecisionNode:
    def __init__(self):
        rospy.init_node('lstm_decision_maker', anonymous=True)
        
        # 初始化参数
        self.load_parameters()
        
        # 初始化ONNX推理引擎
        self.agent = ONNXAgent(self.model_path)
        
        # 创建控制指令发布器（修改消息类型）
        self.cmd_pub = rospy.Publisher('/ppo/cmd_vel', 
                                     TwistStamped,  # 使用TwistStamped
                                     queue_size=10)
        
        # 订阅观测数据
        rospy.Subscriber('/lstm_land/observation', Observation, self.obs_callback)
        
        # 控制参数
        self.control_rate = 20  # Hz
        self.last_obs = None
        self.seq = 0  # 添加消息序列号
        self.yaw = 0
        self.ugv_vx = 0
        self.ugv_vy = 0
        self.height_local = rospy.get_param('~height_local', 1.0)  # 车体坐标系下的目标位置

    def load_parameters(self):
        """加载模型路径配置"""
        self.model_path = rospy.get_param('~model_path', 
                                         'default_model.onnx')
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

    def convert_observation(self, msg):
        """维度安全转换方法"""
        return {
            'height': np.array([msg.height+self.height_local], dtype=np.float32).reshape(1, 1),
            'uwb_dist': np.array(msg.uwb_dist, dtype=np.float32).reshape(1, -1),
            'car_quat': np.array([
                msg.car_quat.x,
                msg.car_quat.y,
                msg.car_quat.z,
                msg.car_quat.w
            ], dtype=np.float32).reshape(1, 4),
            'car_speed': np.array([
                np.linalg.norm([msg.car_speed.x, msg.car_speed.y, msg.car_speed.z]),
                0,
                0
            ], dtype=np.float32).reshape(1, 3),
            'last_velocity': np.array([
                msg.last_velocity.x,
                msg.last_velocity.y,
                msg.last_velocity.z
            ], dtype=np.float32).reshape(1, 3),
            'target_local': np.array(msg.target_local, dtype=np.float32).reshape(1, 2)
        }

    def publish_action(self, action):
        """将动作转换为ROS控制指令（更新为TwistStamped）"""
        cmd = TwistStamped()
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        # 填充头部信息
        cmd.header.seq = self.seq
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "uav0/base_link"  # 根据实际情况设置坐标系

        vx_world = action[0] * cos_yaw - action[1] * sin_yaw+self.ugv_vx
        vy_world = action[0] * sin_yaw + action[1] * cos_yaw+self.ugv_vy
        # 填充速度指令
        cmd.twist.linear.x = vx_world  # 前向速度
        cmd.twist.linear.y = vy_world  # 横向速度
        cmd.twist.linear.z = np.clip(action[2], -1.0, 1.0)
        # 如果有角速度需求可添加
        # cmd.twist.angular.z = action[3] 
        rospy.loginfo(f"控制指令: X={cmd.twist.linear.x:.2f}, Y={cmd.twist.linear.y:.2f}, Z={cmd.twist.linear.z:.2f} m/s")
        
        self.cmd_pub.publish(cmd)
        self.seq += 1  # 更新序列号

    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.last_obs is None:
                rate.sleep()
                continue
                
            try:
                # 转换观测数据
                obs_dict = self.convert_observation(self.last_obs)
                
                # 执行推理
                action = self.agent.predict(obs_dict)
                
                # 发布控制指令
                self.publish_action(action)
                
            except Exception as e:
                rospy.logerr_throttle(1, f"Decision error: {str(e)}")
            
            rate.sleep()


class ONNXAgent:
    def __init__(self, onnx_path):
        # 硬件加速配置
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo'
            }),
            'CPUExecutionProvider'
        ]
        
        # 会话选项优化
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 初始化推理会话
        self.ort_session = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=self.providers
        )
        self.input_names = [i.name for i in self.ort_session.get_inputs()]
        # 验证输入输出
        self.input_info = {i.name: i.shape for i in self.ort_session.get_inputs()}
        self.output_info = [o.name for o in self.ort_session.get_outputs()]

    def predict(self, observation):
        # 创建精确的输入字典
        model_input = {}
        for name in self.input_names:
            if name not in observation:
                raise KeyError(f"Model requires input '{name}' but not provided")
                
            data = observation[name].astype(np.float32)
            expected_shape = self.input_info[name]
            
            # 自动填充批量维度
            if len(data.shape) < len(expected_shape):
                data = data.reshape([1]*len(expected_shape))
                
            model_input[name] = data
            
        outputs = self.ort_session.run(None, model_input)
        return outputs[0][0]

if __name__ == '__main__':
    try:
        node = DecisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass