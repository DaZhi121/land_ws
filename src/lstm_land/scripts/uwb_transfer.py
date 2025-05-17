#!/usr/bin/env python3

import rospy
from nlink_parser.msg import LinktrackNodeframe2
from std_msgs.msg import Float32MultiArray

class UWBNodeOrderProcessor:
    def __init__(self):
        rospy.init_node('uwb_node_order_processor')
        
        # 初始化默认顺序（可根据实际需求修改）
        self.default_order = [0, 2, 3,4]  # <--- 默认顺序在这里设置
        
        # 加载节点顺序配置
        self.node_order = self._load_node_order()
        
        # 订阅和发布配置
        self.sub = rospy.Subscriber("/UAV7/nlink_linktrack_nodeframe2", 
                                   LinktrackNodeframe2,
                                   self.nodeframe_callback)
        self.pub = rospy.Publisher("/uwb_data", 
                                  Float32MultiArray, 
                                  queue_size=10)

    def _load_node_order(self):
        """加载节点顺序配置，优先使用参数服务器配置"""
        try:
            # 尝试读取参数，如果不存在则使用默认值
            order = rospy.get_param("~node_order", self.default_order)
            
            # 参数类型验证
            if not isinstance(order, list) or not all(isinstance(x, int) for x in order):
                error_msg = f"Invalid node_order format: {order}. Expected list of integers."
                rospy.logfatal(error_msg)
                rospy.signal_shutdown(error_msg)
                return []
                
            # 显示加载来源
            source = "parameter server" if rospy.has_param("~node_order") else "default"
            rospy.loginfo(f"Loaded node order from {source}: {order}")
            
            return order
            
        except rospy.ROSException as e:
            rospy.logfatal("Parameter system error: %s", str(e))
            rospy.signal_shutdown("Parameter error")
            return []

    def nodeframe_callback(self, msg):
        """处理数据并生成有序数组"""
        current_nodes = {node.id: node.dis for node in msg.nodes}
        
        # 按顺序填充距离，找不到的节点用0填充
        ordered_data = [current_nodes.get(id, 0.0) for id in self.node_order]
        
        # 构造并发布消息
        output_msg = Float32MultiArray()
        output_msg.data = ordered_data
        self.pub.publish(output_msg)

if __name__ == '__main__':
    try:
        processor = UWBNodeOrderProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass