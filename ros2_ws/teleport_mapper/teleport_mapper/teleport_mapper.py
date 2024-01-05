#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import subprocess


class TeleportMapper(Node):
    def __init__(self):
        super().__init__("teleport_mapper")
        self.get_logger().info("TeleportMapper node started.")

    def launch_turtlebot3_world(self, x_pose, y_pose):
        command = f"ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py x_pose:={x_pose} y_pose:={y_pose}"
        subprocess.call(command, shell=True)


def main(args=None):
    rclpy.init(args=args)
    node = TeleportMapper()
    rclpy.spin(node)
    rclpy.shutdown()
