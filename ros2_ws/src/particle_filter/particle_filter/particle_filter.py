#!/usr/bin/env python3

import os
import pickle
import scipy.stats
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
from numpy.random import random

DISTANCE_BETWEEN_WHEELS_M = 0.160
WHEEL_RADIUS_M = 0.033

LEFT_WHEEL = 0
RIGHT_WHEEL = 1
# fmt: off
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

        # Perticle Filter Estimated Position
        self.robot_x_estimated_pf = 0.0
        self.robot_y_estimated_pf = 0.0
        self.robot_theta_estimated_rad_pf = 0.0

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

        self.particle_number = 4000
        
        self.reference_points = np.array([data.coords for data in self.loaded_laser_scan_data])
        self.reference_points = np.array(self.reference_points)[:, :2] 
        self.particles = self.create_uniform_particles((-2.25, 2.25), (-2.25, 2.25), (0,6.28), self.particle_number)
        self.weights = np.ones(self.particle_number) / self.particle_number
        print(self.weights)

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
            # fmt: on

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

    def predict(self, particles, current_time_ros, std=(.2, .05)):

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
                particles[i, 0] += d_x + (randn() * std[1])
                particles[i, 1] += d_y + (randn() * std[1])
                particles[i, 2] += d_theta + (randn() * std[0]) # This updates the particle's orientation based on robot's overall rotation

                # Normalize the particle's orientation to remain within [-pi, pi]
                if particles[i, 2] > np.pi:
                    particles[i, 2] -= 2 * np.pi
                elif particles[i, 2] < -np.pi:
                    particles[i, 2] += 2 * np.pi

        # Update the previous time with the current time
        self.prev_time_ros_particles = current_time_ros
    
    def update(self, particles, weights, z, R, landmarks, estimated_orientation):
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            weights *= scipy.stats.norm(distance, R).pdf(z[i])

        # Incorporate orientation: reduce weight for orientation deviation
        orientation_diff = np.abs(particles[:, 2] - estimated_orientation)
        # Update weights based on orientation difference
        weights *= scipy.stats.norm(0, 0.01).pdf(orientation_diff)

        # Avoid round-off to zero, normalize
        weights += 1.0e-300
        weights /= np.sum(weights)

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))
    
    def resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights.resize(len(particles))
        weights.fill (1.0 / len(weights))
        
    def systematic_resample(self, weights):
        """ Performs the systemic resampling algorithm used by particle filters.

        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.

        Parameters
        ----------
        weights : list-like of float
            list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""

        pos = particles[:, 0:2]
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)

        orientations = particles[:, 2]
        weighted_sin = np.sum(np.sin(orientations) * weights)
        weighted_cos = np.sum(np.cos(orientations) * weights)

        mean_orientation = np.arctan2(weighted_sin, weighted_cos)
        if mean_orientation > np.pi:
            mean_orientation -= 2 * np.pi
        elif mean_orientation < -np.pi:
            mean_orientation += 2 * np.pi

        return mean, var, mean_orientation

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

        print(f"                True robot position: X:{self.robot_x_true:.3f} m, Y:{self.robot_y_true:.3f} m, \u03B8:{self.robot_theta_true_deg:.3f} deg")
        print(f"Velocities estimated robot position: X:{self.robot_x_estimated_v:.3f} m, Y:{self.robot_y_estimated_v:.3f} m, \u03B8:{self.robot_theta_estimated_deg_v:.3f} deg")
        print(f" Particles estimated robot position: X:{self.robot_x_estimated_pf:.3f} m, Y:{self.robot_y_estimated_pf:.3f} m, \u03B8:{np.degrees(self.robot_theta_estimated_rad_pf):.3f} deg")
        print(f"Number of particles: {len(self.particles)}")
    
        #distance from self.robot_x_estimated_v  and self.robot_y_estimated_v o each reference point
        # Your robot's current estimated position
        robot_pos = np.array([self.robot_x_estimated_v, self.robot_y_estimated_v])

        # Calculate the distance from the robot's position to each landmark
        zs = np.linalg.norm(self.reference_points - robot_pos, axis=1)

        self.predict(self.particles, current_time_ros)

        self.update(self.particles, self.weights, zs, 0.1, self.reference_points, self.robot_theta_estimated_rad_v)

        if self.neff(self.weights) < self.particle_number/2:
            indexes = self.systematic_resample(self.weights)
            self.resample_from_index(self.particles, self.weights, indexes)
            assert np.allclose(self.weights, 1/self.particle_number)

        mu , var, ori = self.estimate(self.particles, self.weights)
        self.robot_x_estimated_pf = mu[0]
        self.robot_y_estimated_pf = mu[1]
        self.robot_theta_estimated_rad_pf = ori


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

        self.ax.scatter(
            self.robot_x_estimated_pf,
            self.robot_y_estimated_pf,
            s=64,
            c="#fcba03",
            edgecolors="#176B87",
            label="Estimated Robot Position (PF)",
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

        # Draw the orientation arrow
        self.ax.arrow(
            self.robot_x_estimated_pf,
            self.robot_y_estimated_pf,
            0.25 * np.cos(self.robot_theta_estimated_rad_pf),
            0.25 * np.sin(self.robot_theta_estimated_rad_pf),
            head_width=0.05,
            head_length=0.1,
            fc="blue",
            ec="blue",
            label="Estimated Robot Orientation (PF)",
        )

        laser_scans = plt.scatter(
            self.reference_points[:, 0],
            self.reference_points[:, 1],
            color="#98C379",
            s=12,
            label="Reference Points",
        )

        # for particle in self.particles:
        #     self.ax.scatter(particle[0], particle[1], c="blue", s=4)
        #     self.ax.arrow(
        #     particle[0],
        #     particle[1],
        #     0.25/4 * np.cos(particle[2]),
        #     0.25/4 * np.sin(particle[2]),
        #     head_width=0.05,
        #     head_length=0.05,
        #     fc="blue",
        #     ec="blue",
        # )


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
