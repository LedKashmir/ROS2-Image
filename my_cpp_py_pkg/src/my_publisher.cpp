// Copyright 2021, Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/rclcpp.hpp"
#include "stdio.h"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("image_publisher", options);
  image_transport::ImageTransport it(node);
  image_transport::Publisher pub = it.advertise("camera/image", 1);
  
  for (int i=2;i<=51;i++)
  {
  	char file[100];
  	sprintf(file,"/home/z0123/image/bev_Camera1_%d.png",i);
  	cv::Mat image = cv::imread(file, cv::IMREAD_COLOR);
  	std_msgs::msg::Header hdr;
  	sensor_msgs::msg::Image::SharedPtr msg;
  	msg = cv_bridge::CvImage(hdr, "bgr8", image).toImageMsg();
  	rclcpp::WallRate loop_rate(5);
	pub.publish(msg);
	rclcpp::spin_some(node);
	loop_rate.sleep();
  }

}
