#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from detection import *
 
bridge = CvBridge() # 转换为ros2的消息类型(imgmsg)的工具
 
class NodeSubscribe(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("大家好，我是listener:%s" % name)
        self.publisher_ = self.create_publisher(Image, "processed_image_topic", 1)
 
    def callback(self,data):
        global bridge
        # ros2消息类型(imgmsg)转换为np.array
        np_arr = np.frombuffer(data.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #cv_img = bridge.imgmsg_to_cv2(data, "bgr8") 
        #cv_img = operation(cv_image)
        processed_img_msg = bridge.cv2_to_imgmsg(cv_img, "bgr8")
        self.publisher_.publish(processed_img_msg)
 
def main(args=None):
    rclpy.init()
    node = NodeSubscribe("image_node") # 实例化创建一个节点--image_node
    # 创建一个话题(image_data)得与发送端一致，定义其中的消息类型为Image。利用callback函数持续接收
    node.create_subscription(CompressedImage,'/image5/compressed', node.callback, 1)
    
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
  main()
