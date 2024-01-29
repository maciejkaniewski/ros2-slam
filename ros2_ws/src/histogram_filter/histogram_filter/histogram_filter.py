#!/usr/bin/env python3

import math
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

HISTOGRAM_RANGE = (0.0, 3.5)
BINS = 15


class HistogramFilter(Node):
    def __init__(self, ax):
        super().__init__("histogram_filter")
        self.get_logger().info("HistogramFilter node started.")

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.ax = ax
        self.loaded_laser_scan_data = None
        self.loaded_laser_scan_data_histograms = np.array([])

        self.timer = self.create_timer(0.1, self.plot_callback)
        self.scan_callback_started = False

        self.robot_x_true = 0.0
        self.robot_y_true = 0.0

        self.robot_x_estimated = 0.0
        self.robot_y_estimated = 0.0

    def load_laser_scan_data(self):
        """
        Load laser scan data from a pickle file.
        """

        laser_scan_data_path = os.path.join(
            get_package_share_directory("histogram_filter"),
            "data",
            "turtlebot3_dqn_stage4.pkl",
        )

        with open(laser_scan_data_path, "rb") as file:
            self.loaded_laser_scan_data = pickle.load(file)

        self.get_logger().info("LaserScan data loaded.")

    def localize_robot(self, current_histogram: np.ndarray) -> tuple[float, float]:
        """
        Locates the robot by finding the closest coordinates based on the histogram difference.
        """
        
        min_difference = float('inf')
        estimated_x, estimated_y = None, None

        for laser_scan_data in self.loaded_laser_scan_data_histograms:
            # Compute absolute difference between histograms
            difference = np.abs(laser_scan_data.measurements - current_histogram)
            total_difference = np.sum(difference)

            # Check if this is the smallest difference so far
            if total_difference < min_difference:
                min_difference = total_difference
                estimated_x, estimated_y = laser_scan_data.coords

        return estimated_x, estimated_y

    def convert_laser_scan_data_to_histograms(self):
        """
        Converts the loaded laser scan data to histograms.

        This method iterates over the loaded laser scan data and converts each set of measurements
        into a histogram using the specified range and number of bins. The resulting histograms are
        stored in the `loaded_laser_scan_data_histograms` array.
        """
        
        # fmt: off
        for laser_scan_data in self.loaded_laser_scan_data:
            hist, _ = np.histogram(laser_scan_data.measurements, range=HISTOGRAM_RANGE, bins=BINS)
            new_laser_scan_data = LaserScanData(coords=laser_scan_data.coords, measurements=hist)
            self.loaded_laser_scan_data_histograms = np.append(self.loaded_laser_scan_data_histograms, new_laser_scan_data)

        self.get_logger().info("LaserScan data converted to histograms.")
        # fmt: on

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        # fmt: off
        self.scan_data = LaserScanData(coords=(0.0, 0.0), measurements=msg.ranges)
        self.current_histogram, self.bin_edges = np.histogram(self.scan_data.measurements, range=HISTOGRAM_RANGE, bins=BINS)
        self.robot_x_estimated, self.robot_y_estimated = self.localize_robot(self.current_histogram)
        self.scan_callback_started = True
        clear = lambda: os.system("clear")
        clear()
        self.get_logger().info(f"True robot position: X: {self.robot_x_true:.3f} [m], Y: {self.robot_y_true:.3f} [m]")
        self.get_logger().info(f"Estimated robot position: X: {self.robot_x_estimated:.3f} [m], Y: {self.robot_y_estimated:.3f} [m]")
        # fmt: on

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.

        Args:
            msg (Odometry): The incoming odometry message.
        """

        self.robot_x_true = msg.pose.pose.position.x
        self.robot_y_true = msg.pose.pose.position.y

    def plot_callback(self):
        if self.scan_callback_started:
            self.ax.cla()

            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            bin_width = self.bin_edges[1] - self.bin_edges[0]
            # fmt: off
            self.ax.bar(bin_centers, self.current_histogram, align="center", width=bin_width)
            # fmt: on
            self.ax.grid()
            
            self.ax.set_title("Histogram of Current Measurements")
            self.ax.set_xlabel("Distance [meters]")
            self.ax.set_ylabel("Frequency")
            plt.draw()
            plt.pause(0.00001)


def main(args=None):
    plt.figure()
    ax = plt.gca()
    plt.ion()
    plt.show()

    rclpy.init(args=args)
    node = HistogramFilter(ax=ax)
    node.load_laser_scan_data()
    node.convert_laser_scan_data_to_histograms()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
