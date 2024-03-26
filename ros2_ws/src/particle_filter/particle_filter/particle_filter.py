#!/usr/bin/env python3

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, JointState
from utils.laser_scan_data import LaserScanData
from utils.pgm_map_loader import PgmMapLoader
from numpy.random import uniform
from numpy.random import randn
from rclpy.time import Time
from geometry_msgs.msg import Twist

DISTANCE_BETWEEN_WHEELS_M = 0.160
WHEEL_RADIUS_M = 0.033

LEFT_WHEEL = 0
RIGHT_WHEEL = 1

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

        self.joint_states_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10,
        )

        self.initial_position_saved = False


        self.ax = ax
        self.loaded_laser_scan_data = None
        self.loaded_laser_scan_data_histograms = np.array([])

        self.plot_timer = self.create_timer(0.1, self.plot_callback)

        self.robot_x_true = 0.0
        self.robot_y_true = 0.0
        self.robot_theta_true_rad = 0.0
        self.robot_theta_true_deg = 0.0

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
        
        self.d_time_particles = 0.0
        self.prev_time_ros_particles = None



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
        
        self.reference_points = np.array([data.coords for data in self.loaded_laser_scan_data])
        self.particles = self.create_uniform_particles((-2.25, 2.25), (-2.25, 2.25), (0,2*np.pi), 50)

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
    
    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi
        return particles

    def predict(self, particles, current_time_ros):

        if self.prev_time_ros_particles is not None:
            self.d_time_particles = (current_time_ros - self.prev_time_ros_particles).nanoseconds / 1e9

            # Calculate average rotation and translation from wheel velocities
            d_theta = ((self.right_wheel_velocity_m_s - self.left_wheel_velocity_m_s) / DISTANCE_BETWEEN_WHEELS_M) * self.d_time
            
            # For each particle, calculate its individual movement based on its own orientation
            for i in range(len(particles)):
                # Individual particle angle for this step
                particle_angle = particles[i, 2]

                # Apply movement based on individual particle orientation
                # Note: d_x and d_y are computed individually for each particle
                d_x = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * self.d_time_particles * np.cos(particle_angle))
                d_y = ((self.left_wheel_velocity_m_s + self.right_wheel_velocity_m_s) / 2 * self.d_time_particles * np.sin(particle_angle))
                
                # Update particle position and orientation
                particles[i, 0] += d_x
                particles[i, 1] += d_y
                particles[i, 2] += d_theta  # This updates the particle's orientation based on robot's overall rotation

                # Normalize the particle's orientation to remain within [-pi, pi]
                if particles[i, 2] > np.pi:
                    particles[i, 2] -= 2 * np.pi
                elif particles[i, 2] < -np.pi:
                    particles[i, 2] += 2 * np.pi

        # Update the previous time with the current time
        self.prev_time_ros_particles = current_time_ros

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
        self.predict(self.particles, current_time_ros)

        print(f"                True robot position: X:{self.robot_x_true:.3f} m, Y:{self.robot_y_true:.3f} m, \u03B8:{self.robot_theta_true_deg:.3f} deg")
        print(f"Velocities estimated robot position: X:{self.robot_x_estimated_v:.3f} m, Y:{self.robot_y_estimated_v:.3f} m, \u03B8:{self.robot_theta_estimated_deg_v:.3f} deg")
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

        laser_scans = plt.scatter(
            self.reference_points[:, 0],
            self.reference_points[:, 1],
            color="#98C379",
            s=12,
            label="Reference Points",
        )

        for particle in self.particles:
            self.ax.scatter(particle[0], particle[1], c="blue", s=4)
            self.ax.arrow(
            particle[0],
            particle[1],
            0.25/4 * np.cos(particle[2]),
            0.25/4 * np.sin(particle[2]),
            head_width=0.05,
            head_length=0.05,
            fc="blue",
            ec="blue",
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
