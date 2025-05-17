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
        self.agent = LSTMDronePolicyDeployer(self.model_path,1,256)
        self.agent.reset_states()
        
        # 创建控制指令发布器（修改消息类型）
        self.cmd_pub = rospy.Publisher('/LSTM/cmd_vel', 
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
            'height': np.array([msg.height], dtype=np.float32).reshape(1, 1),
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
        cmd.header.frame_id = ""  # 根据实际情况设置坐标系


        vx_world = action[0] * cos_yaw - (action[1]) * sin_yaw+self.ugv_vx
        vy_world = action[0] * sin_yaw + (action[1]) * cos_yaw+self.ugv_vy
        cmd.twist.linear.x = vx_world  # 前向速度
        cmd.twist.linear.y = vy_world  # 横向速度
        cmd.twist.linear.z = np.clip(action[2], -0.3, 0.3)
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


class LSTMDronePolicyDeployer:
    def __init__(self, onnx_path, lstm_num_layers=1, lstm_hidden_size=64):
        """
        初始化部署器
        :param onnx_path: ONNX模型文件路径
        :param lstm_num_layers: LSTM层数 (需与训练时配置一致)
        :param lstm_hidden_size: LSTM隐藏层大小 (需与训练时配置一致)
        """
        self.providers = [
            'CUDAExecutionProvider',  # 优先使用GPU加速
            'CPUExecutionProvider'    # 其次使用CPU
        ]
        # 初始化ONNX推理会话
        self.session = ort.InferenceSession(
            onnx_path,
            providers=self.providers
        )
        
        # 存储LSTM参数
        self.num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size
        
        # 初始化状态存储器
        self.hidden_state = None
        self.cell_state = None
        self.reset_states()
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # 验证模型结构
        self._sanity_check()

    def _sanity_check(self):
        """验证模型输入输出是否符合预期"""
        expected_inputs = [
            'height', 'uwb_dist', 'car_quat',
            'car_speed', 'last_velocity', 'target_local',
            'lstm_hidden_in', 'lstm_cell_in'
        ]
        assert all(name in self.input_names for name in expected_inputs), \
            "模型输入结构不符合预期"
        
        expected_outputs = ['action', 'lstm_hidden_out', 'lstm_cell_out']
        assert all(name in self.output_names for name in expected_outputs), \
            "模型输出结构不符合预期"

    def reset_states(self, batch_size=1):
        """
        重置LSTM状态
        :param batch_size: 当前批次大小 (默认为1)
        """
        self.hidden_state = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), 
            dtype=np.float32
        )
        self.cell_state = np.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=np.float32
        )


    def _validate_inputs(self, observations):
        """验证输入数据格式"""
        required_keys = [
            'height', 'uwb_dist', 'car_quat',
            'car_speed', 'last_velocity', 'target_local'
        ]
        missing_keys = [key for key in required_keys if key not in observations]
        if missing_keys:
            raise ValueError(f"缺失必要观测字段: {missing_keys}")
        
        # 修改为接受(1,)或(1,1)的维度
        if observations['height'].shape not in [(1,), (1, 1)]:
            raise ValueError(f"height维度应为(1,)或(1,1)，实际是{observations['height'].shape}")

    def predict(self, observations, deterministic=True):
        """
        执行策略推理
        :param observations: 观测数据字典，包含：
            - height: 高度 (1,1)
            - uwb_dist: UWB距离测量 (1,4)
            - car_quat: 四元数姿态 (1,4)
            - car_speed: 速度向量 (1,3)
            - last_velocity: 上一时刻速度 (1,3)
            - target_local: 目标位置 (1,2)
        :param deterministic: 是否使用确定性策略 (保持True以与训练一致)
        :return: 动作数组 (1, action_dim)
        """
        """执行策略推理"""
        # 预处理观测数据
        processed_obs = {
            'height': np.asarray(observations['height'], dtype=np.float32).reshape(1, 1),
            'uwb_dist': np.asarray(observations['uwb_dist'], dtype=np.float32).reshape(1, 4),
            'car_quat': np.asarray(observations['car_quat'], dtype=np.float32).reshape(1, 4),
            'car_speed': np.asarray(observations['car_speed'], dtype=np.float32).reshape(1, 3),
            'last_velocity': np.asarray(observations['last_velocity'], dtype=np.float32).reshape(1, 3),
            'target_local': np.asarray(observations['target_local'], dtype=np.float32).reshape(1, 2)
        }
        
        self._validate_inputs(processed_obs)

        
        # 构建输入字典
        ort_inputs = {
            'height': processed_obs['height'],
            'uwb_dist': processed_obs['uwb_dist'],
            'car_quat': processed_obs['car_quat'],
            'car_speed': processed_obs['car_speed'],
            'last_velocity': processed_obs['last_velocity'],
            'target_local': processed_obs['target_local'],
            'lstm_hidden_in': self.hidden_state,
            'lstm_cell_in': self.cell_state
        }
        
        # 执行推理
        action, new_hidden, new_cell = self.session.run(
            self.output_names,
            ort_inputs
        )
        
        # 更新LSTM状态
        self.hidden_state = new_hidden
        self.cell_state = new_cell
        
        return action[0]  # 返回第一个(唯一)批次的动作

    @property
    def state(self):
        """获取当前LSTM状态(用于调试)"""
        return {
            'hidden': self.hidden_state.copy(),
            'cell': self.cell_state.copy()
        }

if __name__ == '__main__':
    try:
        node = DecisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass