#!/usr/bin/env python3

import os
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from utils.laser_scan_data_new import LaserScanDataNew
from geometry_msgs.msg import Twist

import pickle

DISTANCE_BETWEEN_WHEELS_M = 0.160
WHEEL_RADIUS_M = 0.033

LEFT_WHEEL = 0
RIGHT_WHEEL = 1

LINE_SEQUENCE = [
    [0.0, 0.0, 90.0],
    [0.0, 0.0, -90.0],
    [0.0, -1.0, -90.0],
    [0.0, -1.0, 90.0],
    [0.0, -0.9, 90.0],
    [0.0, -0.8, 90.0],
    [0.0, -0.7, 90.0],
    [0.0, -0.6, 90.0],
    [0.0, -0.5, 90.0],
    [0.0, -0.4, 90.0],
    [0.0, -0.3, 90.0],
    [0.0, -0.2, 90.0],
    [0.0, -0.1, 90.0],
    [0.0, 0.1, 90.0],
    [0.0, 0.2, 90.0],
    [0.0, 0.3, 90.0],
    [0.0, 0.4, 90.0],
    [0.0, 0.5, 90.0],
    [0.0, 0.6, 90.0],
    [0.0, 0.7, 90.0],
    [0.0, 0.8, 90.0],
    [0.0, 0.9, 90.0],
    [0.0, 1.0, 90.0],
]  # [x, y, theta_deg]

# Crete 4x4 grid with 0.1m spacing heading 90 degrees
GRID = [
    [0.0, 0.0, 90.0],  # scan 1
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.0, 90.0],  # scan 2
    [0.5, 0.5, 90.0],  # scan 3
    [0.5, 0.5, -180.0],
    [0.0, 0.5, -180.0],
    [0.0, 0.5, 90.0],  # scan 4
    [0.0, 0.5, -180.0],
    [-0.5, 0.5, -180.0],
    [-0.5, 0.5, 90.0],  # scan 5
    [-0.5, 0.5, -90.0],
    [-0.5, 0.0, -90.0],
    [-0.5, 0.0, 90.0],  # scan 6
    [-0.5, 0.0, -90.0],
    [-0.5, -0.5, -90.0],
    [-0.5, -0.5, 90.0],  # scan 7
    [-0.5, -0.5, 0.0],
    [0.0, -0.5, 0.0],
    [0.0, -0.5, 90.0],  # scan 8
    [0.0, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, -0.5, 90.0],  # scan 9
]


LINE_SEQUENCE = GRID


class ReferenceGrid(Node):
    def __init__(self):
        super().__init__("reference_grid")
        self.get_logger().info("ReferenceGrid node started.")

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

        self.joint_states_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10,
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.laser_scan_data = np.array([])

        self.initial_position_saved = False

        # Velocities
        self.left_wheel_velocity_rad_s = 0.0
        self.right_wheel_velocity_rad_s = 0.0

        self.left_wheel_velocity_m_s = 0.0
        self.right_wheel_velocity_m_s = 0.0

        # Velocities Esitmated Position
        self.robot_x_estimated_v = 0.0
        self.robot_y_estimated_v = 0.0
        self.robot_theta_estimated_rad_v = 0.0
        self.robot_theta_estimated_deg_v = 0.0

        # Time
        self.d_time = 0.0
        self.prev_time_ros = None

        self.robot_x_true = 0.0
        self.robot_y_true = 0.0
        self.robot_theta_true_rad = 0.0
        self.robot_theta_true_deg = 0.0

        self.cmd_timer = self.create_timer(1 / 50, self.cmd_callback)

        self.target_x = LINE_SEQUENCE[0][0]
        self.target_y = LINE_SEQUENCE[0][1]
        self.target_theta_deg = LINE_SEQUENCE[0][2]

        self.prev_error_pos = 0.0
        self.prev_error_theta = 0.0
        self.sequence_index = 0

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

    def calculate_odometry_from_velocities(self, current_time_ros: Time) -> None:
        """
        Calculates the odometry of the robot based on the wheel velocities.

        Args:
            current_time_ros (Time): The current time in ROS.
        """

        if self.prev_time_ros is not None:

            self.d_time = (current_time_ros - self.prev_time_ros).nanoseconds / 1e9

            # fmt: off
            d_theta = ((self.right_wheel_velocity_m_s - self.left_wheel_velocity_m_s) / (DISTANCE_BETWEEN_WHEELS_M) * self.d_time)
            d_x = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * self.d_time * np.cos(self.robot_theta_estimated_rad_v))
            d_y = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * self.d_time * np.sin(self.robot_theta_estimated_rad_v))
            # fmt: ons

            self.robot_x_estimated_v += d_x
            self.robot_y_estimated_v += d_y
            self.robot_theta_estimated_rad_v += d_theta

            if self.robot_theta_estimated_rad_v > np.pi:
                self.robot_theta_estimated_rad_v -= 2 * np.pi
            elif self.robot_theta_estimated_rad_v < -np.pi:
                self.robot_theta_estimated_rad_v += 2 * np.pi

            self.robot_theta_estimated_deg_v = np.degrees(self.robot_theta_estimated_rad_v)

        # Update the previous time with the current time
        self.prev_time_ros = current_time_ros

    def move_robot(self, linear_velocity) -> None:

        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)

    def rotate_robot(self, angular_velocity: float) -> None:

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = angular_velocity
        self.cmd_vel_publisher.publish(msg)

    def stop_robot(self) -> None:

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)

    def angle_difference(self, target, current):
        diff = (target - current + 180) % 360 - 180
        return diff

    def dump_laser_scan_data(self, file_path: str) -> None:
        """
        Dumps the laser scan data to a file using pickle.

        Args:
            file_path (str): The file path to the output file.
        """

        with open(file_path, "wb") as file:
            pickle.dump(self.laser_scan_data, file)

    def cmd_callback(self):

        Kp_pos = 0.5
        Kd_pos = 0.75

        Kp_theta = 0.25
        Kd_theta = 0.50

        error_pos = np.sqrt(
            (self.target_x - self.robot_x_estimated_v) ** 2
            + (self.target_y - self.robot_y_estimated_v) ** 2
        )

        # error_theta = self.target_theta_deg - self.robot_theta_estimated_deg_v
        error_theta = self.angle_difference(
            self.target_theta_deg, self.robot_theta_estimated_deg_v
        )

        d_error_pos = error_pos - self.prev_error_pos
        d_error_theta = error_theta - self.prev_error_theta

        print(f"Error pos: {error_pos:.3f}, Error theta: {error_theta:.3f}")

        # if abs(error_theta) > 0.5:
        #     angular_vel = Kp_theta * error_theta + Kd_theta * d_error_theta
        #     angular_vel = max(min(angular_vel, 0.2), -0.2)
        #     self.rotate_robot(angular_vel)
        # elif abs(error_pos) > 0.005:
        #     linear_vel = Kp_pos * error_pos + Kd_pos * d_error_pos
        #     linear_vel = max(min(linear_vel, 0.20), -0.20)
        #     self.move_robot(linear_vel)
        # else:
        #     self.stop_robot()
        #     time.sleep(1)

        if abs(error_theta) > 0.5:
            angular_vel = Kp_theta * error_theta + Kd_theta * d_error_theta
            angular_vel = max(min(angular_vel, 0.4), -0.4)
            self.rotate_robot(angular_vel)
        elif abs(error_pos) >= 0.01:  # Increased threshold to provide a more robust stopping criterion
            linear_vel = Kp_pos * error_pos + Kd_pos * d_error_pos
            # Reduce velocity as the robot nears the target
            if abs(error_pos) < 0.005:  # Start reducing speed 10 cm before the target
                linear_vel *= (abs(error_pos) / 0.1)
            linear_vel = max(min(linear_vel, 0.20), -0.20)
            self.move_robot(linear_vel)
        else:
            self.stop_robot()
            time.sleep(1)

            if (
                self.sequence_index != 1
                and self.sequence_index != 2
                and self.sequence_index != 5
                and self.sequence_index != 6
                and self.sequence_index != 8
                and self.sequence_index != 9
                and self.sequence_index != 11
                and self.sequence_index != 12
                and self.sequence_index != 14
                and self.sequence_index != 15
                and self.sequence_index != 17
                and self.sequence_index != 18
                and self.sequence_index != 20
                and self.sequence_index != 21
            ):
                scan_data = LaserScanDataNew(
                    coords=(
                        self.robot_x_estimated_v,
                        self.robot_y_estimated_v,
                        self.robot_theta_estimated_deg_v,
                    ),
                    measurements=self.current_scan_data,
                )
                self.laser_scan_data = np.append(self.laser_scan_data, scan_data)
                self.get_logger().info("Laser scan data appended to array.")

            self.sequence_index += 1
            if self.sequence_index == (len(LINE_SEQUENCE)):
                self.dump_laser_scan_data(
                    "/home/mkaniews/Desktop/grid_updated_dqn4_new.pkl"
                )
                self.get_logger().info("Laser scan data dumped to file.")
                self.get_logger().info("Sequence finished.")
                self.stop_robot()
                time.sleep(1)
                self.destroy_node()
                rclpy.shutdown()

            self.target_x = LINE_SEQUENCE[self.sequence_index][0]
            self.target_y = LINE_SEQUENCE[self.sequence_index][1]
            self.target_theta_deg = LINE_SEQUENCE[self.sequence_index][2]

        self.prev_error_pos = error_pos
        self.prev_error_theta = error_theta

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        # fmt: off
        self.current_scan_data = msg.ranges
        # fmt: on

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states message.

        Args:
            msg (JointState): The joint states message.
        """

        clear = lambda: os.system("clear")
        clear()

        # Get the velocities of the wheels in rad/s
        self.left_wheel_velocity_rad_s = msg.velocity[LEFT_WHEEL]
        self.right_wheel_velocity_rad_s = msg.velocity[RIGHT_WHEEL]

        # Convert the velocities to m/s
        self.left_wheel_velocity_m_s = self.left_wheel_velocity_rad_s * WHEEL_RADIUS_M
        self.right_wheel_velocity_m_s = self.right_wheel_velocity_rad_s * WHEEL_RADIUS_M

        # fmt: off
        print(f" Left wheel velocity: {self.left_wheel_velocity_rad_s:.3f} rad/s, {self.left_wheel_velocity_m_s:.3f} m/s")
        print(f"Right wheel velocity: {self.right_wheel_velocity_rad_s:.3f} rad/s, {self.right_wheel_velocity_m_s:.3f} m/s\n")
        current_time_ros = Time.from_msg(msg.header.stamp)
        self.calculate_odometry_from_velocities(current_time_ros)

        print(f"                True robot position: X:{self.robot_x_true:.3f} m, Y:{self.robot_y_true:.3f} m, \u03B8:{self.robot_theta_true_deg:.3f} deg")
        print(f"Velocities estimated robot position: X:{self.robot_x_estimated_v:.3f} m, Y:{self.robot_y_estimated_v:.3f} m, \u03B8:{self.robot_theta_estimated_deg_v:.3f} deg")
        print(f"Target position: X:{self.target_x:.3f} m, Y:{self.target_y:.3f} m, \u03B8:{self.target_theta_deg:.3f} deg\n")
        # fmt: on

        # self.move_robot(0.1, 0.0)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.

        Args:
            msg (Odometry): The incoming odometry message.
        """

        # fmt: off
        if self.initial_position_saved is False:
            self.robot_x_estimated_v = msg.pose.pose.position.x
            self.robot_y_estimated_v = msg.pose.pose.position.y
            self.robot_theta_estimated_rad_v = self.calculate_yaw_from_quaternion(msg.pose.pose.orientation)
            self.initial_position_saved = True

        self.robot_x_true = msg.pose.pose.position.x
        self.robot_y_true = msg.pose.pose.position.y
        self.robot_theta_true_rad = self.calculate_yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_theta_true_deg = np.degrees(self.robot_theta_true_rad)
        # fmt: on


def main(args=None):

    rclpy.init(args=args)
    node = ReferenceGrid()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
