#!/usr/bin/env python3
import rospy
import math
import time
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class MotionController:
    def __init__(self):
        rospy.init_node('ugv_motion_controller')
        
        # 初始化参数
        self._load_parameters()
        
        # 运动控制状态变量
        self.takeoff_allowed = False
        self.lock = threading.Lock()
        
        # ROS通信设置
        self.pub = rospy.Publisher('/ugv_0/cmd_vel', Twist, queue_size=10)
        self.takeoff_sub = rospy.Subscriber('/status/takeoff_complete', Bool, self.takeoff_callback)
        
        # 用户输入处理
        self.mode = 0
        self.running = True
        self.input_thread = threading.Thread(target=self.listen_input)
        self.input_thread.start()

    def _load_parameters(self):
        """加载运动控制参数"""
        self.linear_speed = rospy.get_param('~linear_speed', 3.0)
        
        # 圆周运动参数
        self.circular_angular = rospy.get_param('~circular_angular', 0.1)
        
        # 蛇形运动参数
        serpentine_params = rospy.get_param('~serpentine', {
            'angular_amp': 0.2,
            'amplitude': 0.5
        })
        self.serpentine_angular_amp = serpentine_params['angular_amp']
        self.serpentine_amplitude = serpentine_params['amplitude']
        
        # 自动计算蛇形频率
        self.snake_frequency = math.sqrt(
            (self.linear_speed * self.serpentine_angular_amp) / 
            self.serpentine_amplitude
        ) / (2 * math.pi)

    def takeoff_callback(self, msg):
        """处理起飞完成状态回调"""
        with self.lock:
            self.takeoff_allowed = msg.data
            status = "已解锁" if msg.data else "已锁定"
            rospy.loginfo(f"运动状态变更: {status}")

    def listen_input(self):
        """独立线程监听用户输入"""
        while self.running:
            try:
                user_input = input("\n选择运动模式: \n0-静止 \n1-直线 \n2-圆周 \n3-蛇形 \n请输入数字: ")
                new_mode = int(user_input)
                if 0 <= new_mode <= 3:
                    self.mode = new_mode
                    rospy.loginfo(f"切换至模式 {new_mode}")
                else:
                    rospy.logwarn("无效输入! 请输入0-3")
            except ValueError:
                rospy.logwarn("无效输入! 请输入数字")

    def run(self):
        """主控制循环"""
        start_time = time.time()
        rate = rospy.Rate(10)  # 10Hz
        
        while not rospy.is_shutdown() and self.running:
            cmd = Twist()
            
            # 检查运动许可状态
            with self.lock:
                allowed = self.takeoff_allowed
            
            if allowed:
                # 根据模式生成控制指令
                if self.mode == 0:   # 静止
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    
                elif self.mode == 1: # 直线运动
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = 0.0
                    
                elif self.mode == 2: # 圆周运动
                    cmd.linear.x = self.linear_speed
                    cmd.angular.z = self.circular_angular
                    
                elif self.mode == 3: # 蛇形运动
                    cmd.linear.x = self.linear_speed
                    elapsed_time = time.time() - start_time
                    cmd.angular.z = self.serpentine_angular_amp * math.sin(
                        2 * math.pi * self.snake_frequency * elapsed_time
                    )
            else:
                # 未获得运动许可时强制停止
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            
            self.pub.publish(cmd)
            rate.sleep()
        
        # 退出时停止车辆
        self.pub.publish(Twist())
        self.running = False
        self.input_thread.join()

if __name__ == '__main__':
    try:
        controller = MotionController()
        controller.run()
    except rospy.ROSInterruptException:
        pass