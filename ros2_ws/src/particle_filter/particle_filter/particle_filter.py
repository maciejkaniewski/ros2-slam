#!/usr/bin/env python3

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils.laser_scan_data import LaserScanData
from utils.pgm_map_loader import PgmMapLoader


class ParticleFilter(Node):
    def __init__(self, ax):
        super().__init__("particle_filter")
        self.get_logger().info("ParticleFilter node started.")

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.ax = ax
        self.loaded_laser_scan_data = None
        self.loaded_laser_scan_data_histograms = np.array([])

        self.plot_timer = self.create_timer(0.1, self.plot_callback)

        self.robot_x_true = 0.0
        self.robot_y_true = 0.0
        self.robot_theta_true_rad = 0.0
        self.robot_theta_true_deg = 0.0

        self.robot_x_estimated = 0.0
        self.robot_y_estimated = 0.0
        self.robot_theta_estimated_rad = 0.0
        self.robot_theta_estimated_deg = 0.0

        self.load_map()
        self.load_laser_scan_data()

    def load_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

        # TODO: Load the map from the specified PGM and YAML files with parameters.

        world_pgm_path = os.path.join(
            get_package_share_directory("scan_collector"),
            "worlds",
            "turtlebot3_dqn_stage4.pgm",
        )


        world_yaml_path = os.path.join(
            get_package_share_directory("scan_collector"),
            "config",
            "turtlebot3_dqn_stage4.yaml",
        )

        self.pgm_map = PgmMapLoader(world_yaml_path, world_pgm_path)

    def load_laser_scan_data(self):
        """
        Load laser scan data from a pickle file.
        """

        # laser_scan_data_path = os.path.join(
        #     get_package_share_directory("histogram_filter"),
        #     "data",
        #     "turtlebot3_dqn_stage4.pkl",
        # )

        laser_scan_data_path = "/home/mkaniews/Desktop/laser_scan_data_test.pkl"

        with open(laser_scan_data_path, "rb") as file:
            self.loaded_laser_scan_data = pickle.load(file)

    def calculate_yaw_from_quaternion(self, quaternion):
        """
        Calculates the yaw angle from a given quaternion.

        Args:
            quaternion (Quaternion): The input quaternion.

        Returns:
            float: The yaw angle in radians.
        """

        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.

        Args:
            msg (Odometry): The incoming odometry message.
        """

        # fmt: off
        self.robot_x_true = msg.pose.pose.position.x
        self.robot_y_true = msg.pose.pose.position.y
        self.robot_theta_true_rad = self.calculate_yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_theta_true_deg = np.degrees(self.robot_theta_true_rad)
        # fmt: on

    def plot_callback(self):
        """
        Update and plot the histogram of current measurements and the robot localization.
        """
        self.ax.cla()

        self.ax.scatter(
            self.robot_x_true,
            self.robot_y_true,
            s=64,
            c="#16FF00",
            edgecolors="#176B87",
            label="True Robot Position",
        )

        # Draw the orientation arrow
        self.ax.arrow(
            self.robot_x_true,
            self.robot_y_true,
            0.25 * np.cos(self.robot_theta_true_rad),
            0.25 * np.sin(self.robot_theta_true_rad),
            head_width=0.05,
            head_length=0.1,
            fc="red",
            ec="red",
            label="True Robot Orientation",
        )


        map_width = self.pgm_map.width * self.pgm_map.resolution
        map_height = self.pgm_map.height * self.pgm_map.resolution

        # Define extent of the image in the plot
        extent = [
            self.pgm_map.origin[0],
            self.pgm_map.origin[0] + map_width,
            self.pgm_map.origin[1],
            self.pgm_map.origin[1] + map_height,
        ]

        self.ax.imshow(self.pgm_map.img, cmap="gray", extent=extent)

        self.ax.grid()
        self.ax.set_title("Robot Localization")
        self.ax.set_xlabel("X [meters]")
        self.ax.set_ylabel("Y [meters]")
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)

        ticks = np.arange(-3, 3.1, 1)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.legend()

        plt.draw()
        plt.pause(0.00001)


def main(args=None):
    plt.figure()
    ax = plt.gca()
    plt.ion()
    plt.show()
    rclpy.init(args=args)
    node = ParticleFilter(ax=ax)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
