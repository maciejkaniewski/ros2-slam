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
from utils.pgm_map_loader import PgmMapLoader

HISTOGRAM_RANGE = (0.0, 3.5)
BINS = 25

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
    
    def load_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

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

        min_difference = float("inf")
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
        self.get_logger().info(f"     True robot position: X: {self.robot_x_true:.3f} [m], Y: {self.robot_y_true:.3f} [m]")
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
        """
        Update and plot the histogram of current measurements and the robot localization.
        """

        if self.scan_callback_started:
            self.ax[0].cla()
            self.ax[1].cla()

            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            bin_width = self.bin_edges[1] - self.bin_edges[0]
            # fmt: off
            self.ax[0].bar(bin_centers, self.current_histogram, align="center", width=bin_width, edgecolor="#176B87", color="#61AFEF")
            self.ax[0].set_xticks(self.bin_edges)
            self.ax[0].grid()
            # fmt: on

            self.ax[0].set_title("Histogram of Current Measurements")
            self.ax[0].set_xlabel("Distance [meters]")
            self.ax[0].set_ylabel("Frequency")

            self.ax[1].scatter(
                self.robot_x_true,
                self.robot_y_true,
                s=64,
                c="#16FF00",
                edgecolors="#176B87",
                label="True Robot Position",
            )
            self.ax[1].scatter(
                self.robot_x_estimated,
                self.robot_y_estimated,
                s=64,
                c="#fcba03",
                edgecolors="#176B87",
                label="Estimated Robot Position",
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

            self.ax[1].imshow(self.pgm_map.img, cmap="gray", extent=extent)

            self.ax[1].grid()
            self.ax[1].set_title("Robot Localization")
            self.ax[1].set_xlabel("X [meters]")
            self.ax[1].set_ylabel("Y [meters]")
            self.ax[1].set_xlim(-3, 3)
            self.ax[1].set_ylim(-3, 3)

            ticks = np.arange(-3, 3.1, 1)
            self.ax[1].set_xticks(ticks)
            self.ax[1].set_yticks(ticks)

            self.ax[1].legend()

            plt.draw()
            plt.pause(0.00001)


def main(args=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Create a figure with 2 subplots)
    plt.ion()
    plt.show()
    rclpy.init(args=args)
    node = HistogramFilter(ax=axs)
    node.load_laser_scan_data()
    node.convert_laser_scan_data_to_histograms()
    node.load_map()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
