import os
import pickle

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header
from utils.laser_scan_data import LaserScanData
from utils.pgm_map_loader import PgmMapLoader

PARTICLES_NUM = 1000


class Particle:
    def __init__(self, x, y, theta, weight):
        self._x = x
        self._y = y
        self._theta = theta
        self._weight = weight

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def theta(self):
        return self._theta

    @property
    def weight(self):
        return self._weight

    def __repr__(self) -> str:
        return f"(Particle x:{self.x}, y:{self.y}, \u03B8:{self.theta}, {self.weight} weight)"


class RobotConstants:
    DISTANCE_BETWEEN_WHEELS_M = 0.160
    WHEEL_RADIUS_M = 0.033
    LEFT_WHEEL = 0
    RIGHT_WHEEL = 1


class Robot:
    def __init__(self):

        self._raw_scan = None
        self._initial_position_saved = False

        # True position
        self._robot_x_true = 0.0
        self._robot_y_true = 0.0
        self._robot_theta_true_rad = 0.0
        self._robot_theta_true_deg = 0.0

        # Wheel velocities
        self._left_wheel_velocity_rad_s = 0.0
        self._right_wheel_velocity_rad_s = 0.0

        self._left_wheel_velocity_m_s = 0.0
        self._right_wheel_velocity_m_s = 0.0

        # Position estimated with odometry
        self._robot_x_estimated_v = 0.0
        self._robot_y_estimated_v = 0.0
        self._robot_theta_estimated_rad_v = 0.0
        self._robot_theta_estimated_deg_v = 0.0

        # Perticle Filter Estimated Position
        self._robot_x_estimated_pf = 0.0
        self._robot_y_estimated_pf = 0.0
        self._robot_theta_estimated_rad_pf = 0.0

        # Time
        self.d_time = 0.0
        self.prev_time_ros = None

    @property
    def raw_scan(self):
        return self._raw_scan

    @raw_scan.setter
    def raw_scan(self, value):
        self._raw_scan = value

    @property
    def initial_position_saved(self):
        return self._initial_position_saved

    @initial_position_saved.setter
    def initial_position_saved(self, value):
        self._initial_position_saved = value

    @property
    def left_wheel_velocity_rad_s(self):
        return self._left_wheel_velocity_rad_s

    @left_wheel_velocity_rad_s.setter
    def left_wheel_velocity_rad_s(self, value):
        self._left_wheel_velocity_rad_s = value

    @property
    def right_wheel_velocity_rad_s(self):
        return self._right_wheel_velocity_rad_s

    @right_wheel_velocity_rad_s.setter
    def right_wheel_velocity_rad_s(self, value):
        self._right_wheel_velocity_rad_s = value

    @property
    def left_wheel_velocity_m_s(self):
        return self._left_wheel_velocity_m_s

    @left_wheel_velocity_m_s.setter
    def left_wheel_velocity_m_s(self, value):
        self._left_wheel_velocity_m_s = value

    @property
    def right_wheel_velocity_m_s(self):
        return self._right_wheel_velocity_m_s

    @right_wheel_velocity_m_s.setter
    def right_wheel_velocity_m_s(self, value):
        self._right_wheel_velocity_m_s = value

    def true_position(self) -> str:
        """
        Returns the true position of the robot.

        Returns:
            A string representing the true position of the robot in the format:
            "(X:<x-coordinate> [m], Y:<y-coordinate> [m], θ: <theta-coordinate> [deg])"
        """
        return f"(X:{self._robot_x_true:.3f} [m], Y:{self._robot_y_true:.3f} [m], θ: {self._robot_theta_true_deg:.3f} [deg])"

    def odometry_position(self) -> str:
        """
        Returns the estimated position of the robot based on odometry readings.

        Returns:
            A string representing the estimated position in the format "(X: [x-coordinate] [m], Y: [y-coordinate] [m], θ: [theta] [deg])".
        """
        return f"(X:{self._robot_x_estimated_v:.3f} [m], Y:{self._robot_y_estimated_v:.3f} [m], θ: {self._robot_theta_estimated_deg_v:.3f} [deg])"

    def wheel_velocities_m_s(self) -> str:
        """
        Returns the wheel velocities of the robot in m/s.

        Returns:
            A string representing the wheel velocities in the format "(Left: [left_wheel_velocity] m/s, Right: [right_wheel_velocity] m/s)".
        """
        return f"(Left: {self._left_wheel_velocity_m_s:.3f} m/s, Right: {self._right_wheel_velocity_m_s:.3f} m/s)"

    def set_estimated_position_v(self, x, y, theta):
        """
        Set the estimated position of the robot.

        Args:
            x (float): The estimated x-coordinate of the robot.
            y (float): The estimated y-coordinate of the robot.
            theta (float): The estimated orientation angle of the robot in radians.
        """

        self._robot_x_estimated_v = x
        self._robot_y_estimated_v = y
        self._robot_theta_estimated_rad_v = theta

        if self._robot_theta_estimated_rad_v > np.pi:
            self._robot_theta_estimated_rad_v -= 2 * np.pi
        elif self._robot_theta_estimated_rad_v < -np.pi:
            self._robot_theta_estimated_rad_v += 2 * np.pi

        self._robot_theta_estimated_deg_v = np.degrees(theta)

    def set_true_position(self, x, y, theta):
        """
        Set the true position of the robot.

        Args:
            x (float): The x-coordinate of the true position.
            y (float): The y-coordinate of the true position.
            theta (float): The orientation angle in radians of the true position.
        """
        self._robot_x_true = x
        self._robot_y_true = y
        self._robot_theta_true_rad = theta

        if self._robot_theta_true_rad > np.pi:
            self._robot_theta_true_rad -= 2 * np.pi
        elif self._robot_theta_true_rad < -np.pi:
            self._robot_theta_true_rad += 2 * np.pi

        self._robot_theta_true_deg = np.degrees(theta)

    def calculate_position_v(self, current_time_ros: Time) -> None:
        """
        Calculates the position of the robot based on the wheel velocities.

        Args:
            current_time_ros (Time): The current time in ROS.
        """

        if self.prev_time_ros is not None:

            self.d_time = (current_time_ros - self.prev_time_ros).nanoseconds / 1e9

            # fmt: off
            d_theta = ((self._right_wheel_velocity_m_s - self._left_wheel_velocity_m_s) / (RobotConstants.DISTANCE_BETWEEN_WHEELS_M) * self.d_time)
            d_x = ((self._left_wheel_velocity_m_s + self._right_wheel_velocity_m_s) / 2 * self.d_time * np.cos(self._robot_theta_estimated_rad_v))
            d_y = ((self._left_wheel_velocity_m_s + self._right_wheel_velocity_m_s) / 2 * self.d_time * np.sin(self._robot_theta_estimated_rad_v))
            # fmt: on

            self.set_estimated_position_v(
                self._robot_x_estimated_v + d_x,
                self._robot_y_estimated_v + d_y,
                self._robot_theta_estimated_rad_v + d_theta,
            )

        # Update the previous time with the current time
        self.prev_time_ros = current_time_ros


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("pf")
        self.get_logger().info("ParticleFilter node started.")

        # fmt: off

        # Create publishers
        self.landmarks_publisher = self.create_publisher(PoseArray, "/landmarks", 10)
        self.particles_publisher = self.create_publisher(PoseArray, "/particles", 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, "/custom_occupancy_grid_map", 10)

        # Create subscribers
        self.joint_states_subscription = self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Create timers for pubishers/loggers
        self.timer_particles = self.create_timer(0.1, self.particles_callback)
        self.timer_logger = self.create_timer(0.1, self.logger_callback)

        # Publish the map
        self.load_map()
        self.publish_loaded_map()

        # Load laser scan data and publish landmarks
        self.loaded_laser_scan_data = None  
        self.load_laser_scan_data()
        self.landmarks = self.convert_laser_scan_data_to_pose_array(self.loaded_laser_scan_data)
        self.landmarks_publisher.publish(self.landmarks)

        # Initialize particles
        self.particles = self.create_uniform_particles((-2.25, 2.25), (-2.25, 2.25), (0, 6.28), PARTICLES_NUM)
        self.particles_poses = self.convert_particles_to_pose_array(self.particles)
        self.particles_publisher.publish(self.particles_poses)
        # fmt: on

        self.robot = Robot()

    def load_map(self) -> None:
        """
        Loads the map from the specified PGM and YAML files.
        """

        world_pgm_path = os.path.join(
            get_package_share_directory("scan_collector"),
            "worlds",
            "turtlebot3_dqn_stage4_updated.pgm",
        )

        world_yaml_path = os.path.join(
            get_package_share_directory("scan_collector"),
            "config",
            "turtlebot3_dqn_stage4.yaml",
        )

        self.pgm_map = PgmMapLoader(world_yaml_path, world_pgm_path)

    def publish_loaded_map(self) -> None:
        """
        Publishes the loaded map as an OccupancyGrid message.

        This method converts the loaded map into an OccupancyGrid message and publishes it using the map_publisher.
        """

        occupancy_grid_msg = OccupancyGrid()
        inverted_img = 255 - self.pgm_map.img
        scaled_img = np.flip((inverted_img * (100.0 / 255.0)).astype(np.int8), 0)
        self.map_data = scaled_img.ravel()

        occupancy_grid_msg.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id="odom"
        )
        occupancy_grid_msg.info.width = self.pgm_map.width
        occupancy_grid_msg.info.height = self.pgm_map.height
        occupancy_grid_msg.info.resolution = self.pgm_map.resolution
        occupancy_grid_msg.info.origin = Pose()
        occupancy_grid_msg.info.origin.position.x = self.pgm_map.origin[0]
        occupancy_grid_msg.info.origin.position.y = self.pgm_map.origin[1]
        occupancy_grid_msg.info.origin.position.z = 0.0
        occupancy_grid_msg.info.origin.orientation.w = 1.0
        occupancy_grid_msg.data = self.map_data.tolist()

        self.map_publisher.publish(occupancy_grid_msg)

    def load_laser_scan_data(self):
        """
        Load laser scan data from a pickle file.
        """

        laser_scan_data_path = os.path.join(
            get_package_share_directory("histogram_filter"),
            "data",
            "reference_line_2_updated_dqn4.pkl",
        )

        with open(laser_scan_data_path, "rb") as file:
            self.loaded_laser_scan_data = pickle.load(file)

    def convert_laser_scan_data_to_pose_array(self, loaded_laser_scan_data):
        
        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"
        
        reference_points = np.array([data.coords for data in loaded_laser_scan_data])

        for reference_point in reference_points:
            pose = Pose()
            pose.position.x = reference_point[0]
            pose.position.y = reference_point[1]
            q = self.get_quaternion_from_euler(0, 0, np.deg2rad(reference_point[2]))
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

        return pose_array

    def create_uniform_particles(self, x_range, y_range, theta_range, N) -> np.array:
        """
        Create an array of uniformly distributed particles.

        Args:
            x_range (tuple): Range of x values for the particles.
            y_range (tuple): Range of y values for the particles.
            theta_range (tuple): Range of theta values (in radians) for the particles.
            N (int): Number of particles to create.

        Returns:
            np.array: Array of Particle objects representing the particles.
        """
        particles = np.array([])
        for _ in range(N):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            theta = np.random.uniform(theta_range[0], theta_range[1])
            theta %= 2 * np.pi
            particles = np.append(particles, Particle(x, y, theta, 1.0 / N))
        return particles

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

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """

        # fmt: off
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        # fmt: on
        return [qx, qy, qz, qw]

    def convert_particles_to_pose_array(self, particles):
        """
        Converts a list of particles to a PoseArray message.

        Args:
            particles (list): List of particles.

        Returns:
            PoseArray: A PoseArray message representing the converted particles.
        """
        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"

        for particle in particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            q = self.get_quaternion_from_euler(0, 0, particle.theta)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

        return pose_array

    def particles_callback(self) -> None:
        """
        Publishes the particle poses.
        """

        self.particles_publisher.publish(self.particles_poses)

    def logger_callback(self) -> None:

        # fmt: off
        clear = lambda: os.system("clear")
        clear()
        self.get_logger().info("              True Position:" + self.robot.true_position())
        self.get_logger().info("Estimated Position Odometry:" + self.robot.odometry_position())
        self.get_logger().info("Wheels Velocities:" + self.robot.wheel_velocities_m_s())
        # fmt: on

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function for handling odometry messages.

        Args:
            msg (Odometry): The incoming odometry message.
        """

        if self.robot.initial_position_saved is False:
            self.robot.set_estimated_position_v(
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                self.calculate_yaw_from_quaternion(msg.pose.pose.orientation),
            )
            self.robot.initial_position_saved = True

        self.robot.set_true_position(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self.calculate_yaw_from_quaternion(msg.pose.pose.orientation),
        )

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states message.

        Args:
            msg (JointState): The joint states message.
        """

        # Get the velocities of the wheels in rad/s
        self.robot.left_wheel_velocity_rad_s = msg.velocity[RobotConstants.LEFT_WHEEL]
        self.robot.right_wheel_velocity_rad_s = msg.velocity[RobotConstants.RIGHT_WHEEL]

        # Convert the velocities to m/s
        self.robot.left_wheel_velocity_m_s = (
            self.robot.left_wheel_velocity_rad_s * RobotConstants.WHEEL_RADIUS_M
        )
        self.robot.right_wheel_velocity_m_s = (
            self.robot.right_wheel_velocity_rad_s * RobotConstants.WHEEL_RADIUS_M
        )

        current_time_ros = Time.from_msg(msg.header.stamp)
        self.robot.calculate_position_v(current_time_ros)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        self.scan_callback_started = True
        self.robot.raw_scan = msg.ranges


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
