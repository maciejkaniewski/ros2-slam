#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")
        self.get_logger().info("ParticleFilter node started.")


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    rclpy.shutdown()
