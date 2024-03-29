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

HISTOGRAM_RANGE = (0.0, 3.5)
BINS = 10


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

        self.plot_timer = self.create_timer(0.1, self.plot_callback)
        self.scan_callback_started = False

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
        self.convert_laser_scan_data_to_histograms()

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

        laser_scan_data_path = "/home/mkaniews/Desktop/laser_scan_data.pkl"

        with open(laser_scan_data_path, "rb") as file:
            self.loaded_laser_scan_data = pickle.load(file)

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
        # fmt: on

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
                estimated_x, estimated_y, _ = laser_scan_data.coords

        return estimated_x, estimated_y

    def get_measurements_near_coordinate(self, target_x, target_y, tolerance=0.01):
        """
        Returns the measurements near the specified target coordinates.

        Args:
            target_x (float): The x-coordinate of the target position.
            target_y (float): The y-coordinate of the target position.
            tolerance (float, optional): The maximum allowable distance between a data point and the target coordinates. Defaults to 0.01.

        Returns:
            list or None: A list of measurements near the target coordinates, or None if no matching data is found.
        """

        # fmt: off
        # Iterate through all loaded laser scan data
        for data_point in self.loaded_laser_scan_data:
            data_x, data_y, _= data_point.coords
            # Check if the data point is within the specified tolerance of the target coordinates
            if abs(data_x - target_x) <= tolerance and abs(data_y - target_y) <= tolerance:
                return data_point.measurements
        # Return None or an appropriate value if no matching data is found
        return None
        # fmt: on

    def calculate_orientation(self, current_scan_data):

        ref_data = self.get_measurements_near_coordinate(
            self.robot_x_estimated, self.robot_y_estimated
        )

        # Adjust reference data to 0 degrees orientation by shifting the data
        adjusted_ref_data = np.roll(ref_data, -90)

        # Replace 'inf' values with a large numeric value that doesn't affect other calculations.
        # Use the maximum value from the actual readings, or a specified value that is higher than any expected reading.
        max_valid_value = np.nanmax(
            ref_data[~np.isinf(ref_data)]
        )  # Maximum value among non-infinite values.
        adjusted_ref_data[np.isinf(adjusted_ref_data)] = max_valid_value
        current_scan_data_replaced_inf = np.where(
            np.isinf(current_scan_data), max_valid_value, current_scan_data
        )

        min_diff = float("inf")
        best_shift = 0

        # Iterate through all possible shifts from 0 to 359 degrees.
        for shift in range(360):
            # Use vector operations instead of loops for efficiency
            shifted_current_scan_data = np.roll(current_scan_data_replaced_inf, shift)
            diff = adjusted_ref_data - shifted_current_scan_data
            sum_of_squares = np.sum(diff**2)

            # Check if this sum of squares is less than any previously found.
            if sum_of_squares < min_diff:
                min_diff = sum_of_squares
                best_shift = shift

        # Adjust the result.
        return -(180 - best_shift)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        # fmt: off
        self.current_scan_data = LaserScanData(coords=(0.0, 0.0), measurements=msg.ranges)
        self.current_histogram, self.bin_edges = np.histogram(self.current_scan_data.measurements, range=HISTOGRAM_RANGE, bins=BINS)
        self.robot_x_estimated, self.robot_y_estimated = self.localize_robot(self.current_histogram)
        self.robot_theta_estimated_deg = self.calculate_orientation(self.current_scan_data.measurements)
        self.robot_theta_estimated_rad = np.radians(self.robot_theta_estimated_deg)

        self.scan_callback_started = True
        clear = lambda: os.system("clear")
        clear()
        self.get_logger().info(f"     True robot position: X: {self.robot_x_true:.3f} [m], Y: {self.robot_y_true:.3f} [m]")
        self.get_logger().info(f"Estimated robot position: X: {self.robot_x_estimated:.3f} [m], Y: {self.robot_y_estimated:.3f} [m]")
        self.get_logger().info(f"     True robot orientation \u03B8: {self.robot_theta_true_deg:.3f} [\u00b0]")
        self.get_logger().info(f"Estimated robot orientation \u03B8: {self.robot_theta_estimated_deg:.3f} [\u00b0]")
        # fmt: on

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

        if self.scan_callback_started:
            self.ax[0].cla()
            self.ax[1].cla()
            self.ax[2].cla()

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

            data = self.get_measurements_near_coordinate(
                self.robot_x_estimated, self.robot_y_estimated
            )
            self.ax[1].plot(data, label="LaserScan Data from Memory", linewidth=3)
            self.ax[1].plot(
                self.current_scan_data.measurements,
                label="Current LaserScan Data",
                color="#16FF00",
            )
            self.ax[1].set_title(
                "LaserScan Data from Memory and Current LaserScan Data"
            )
            self.ax[1].set_xlabel("\u03B8 [\u00b0]")
            self.ax[1].set_ylabel("Distance [meters]")
            self.ax[1].legend()
            self.ax[1].grid()

            self.ax[2].scatter(
                self.robot_x_true,
                self.robot_y_true,
                s=64,
                c="#16FF00",
                edgecolors="#176B87",
                label="True Robot Position",
            )
            self.ax[2].scatter(
                self.robot_x_estimated,
                self.robot_y_estimated,
                s=64,
                c="#fcba03",
                edgecolors="#176B87",
                label="Estimated Robot Position",
            )

            # Draw the orientation arrow
            self.ax[2].arrow(
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

            self.ax[2].arrow(
                self.robot_x_estimated,
                self.robot_y_estimated,
                0.25 * np.cos(self.robot_theta_estimated_rad),
                0.25 * np.sin(self.robot_theta_estimated_rad),
                head_width=0.05,
                head_length=0.1,
                fc="blue",
                ec="blue",
                label="Estimated Robot Orientation",
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

            self.ax[2].imshow(self.pgm_map.img, cmap="gray", extent=extent)

            self.ax[2].grid()
            self.ax[2].set_title("Robot Localization")
            self.ax[2].set_xlabel("X [meters]")
            self.ax[2].set_ylabel("Y [meters]")
            self.ax[2].set_xlim(-3, 3)
            self.ax[2].set_ylim(-3, 3)

            ticks = np.arange(-3, 3.1, 1)
            self.ax[2].set_xticks(ticks)
            self.ax[2].set_yticks(ticks)
            self.ax[2].legend()

            plt.draw()
            plt.pause(0.00001)


def main(args=None):
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    plt.ion()
    plt.show()
    rclpy.init(args=args)
    node = HistogramFilter(ax=axs)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
