import os
import pickle

import numpy as np
import rclpy
import scipy.stats
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from numpy.random import randn, random
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header
from utils.laser_scan_data import LaserScanData
from utils.pgm_map_loader import PgmMapLoader

PARTICLES_NUM = 1000

HISTOGRAM_RANGE = (0.0, 3.5)
BINS = 20


class Particle:
    def __init__(self, x, y, theta, weight):
        self._x = x
        self._y = y
        self._theta = theta
        self._weight = weight

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

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

    @property
    def robot_x_estimated_v(self):
        return self._robot_x_estimated_v

    @property
    def robot_y_estimated_v(self):
        return self._robot_y_estimated_v

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

    def particle_filter_position(self) -> str:
        """
        Returns the estimated position of the robot based on Particle Filter readings.

        Returns:
            A string representing the estimated position in the format "(X: [x-coordinate] [m], Y: [y-coordinate] [m], θ: [theta] [deg])".
        """
        return f"(X:{self._robot_x_estimated_pf:.3f} [m], Y:{self._robot_y_estimated_pf:.3f} [m], θ: {np.degrees(self._robot_theta_estimated_rad_pf):.3f} [deg])"

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

    def set_estimated_position_pf(self, x, y, theta):
        """
        Set the estimated position of the robot.

        Args:
            x (float): The estimated x-coordinate of the robot.
            y (float): The estimated y-coordinate of the robot.
            theta (float): The estimated orientation angle of the robot in radians.
        """

        self._robot_x_estimated_pf = x
        self._robot_y_estimated_pf = y
        self._robot_theta_estimated_rad_pf = theta

        if self._robot_theta_estimated_rad_pf > np.pi:
            self._robot_theta_estimated_rad_pf -= 2 * np.pi
        elif self._robot_theta_estimated_rad_pf < -np.pi:
            self._robot_theta_estimated_rad_pf += 2 * np.pi

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

    def is_robot_moving(self) -> bool:
        """
        Checks if the robot is moving.

        Returns:
            bool: True if the robot is moving, False otherwise.
        """
        if np.isclose(self._left_wheel_velocity_m_s, 0.0, atol=1e-3) and np.isclose(
            self._right_wheel_velocity_m_s, 0.0, atol=1e-3
        ):
            return False
        else:
            return True


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("pf")
        self.get_logger().info("ParticleFilter node started.")

        # fmt: off

        # Create publishers
        self.landmarks_publisher = self.create_publisher(PoseArray, "/landmarks", 10)
        self.particles_publisher = self.create_publisher(PoseArray, "/particles", 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, "/custom_occupancy_grid_map", 10)
        self.pf_estimated_position_publisher = self.create_publisher(PoseStamped, "/pf_pose", 10)
        self.closest_point_publisher = self.create_publisher(PoseStamped, "/closest_point", 10)

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

        self.current_histogram = np.array([])
        self.loaded_laser_scan_data_histograms = np.array([])
        self.hist_diff_chi = np.array([])
        self.hist_diff_my = np.array([])


        # Load laser scan data and publish landmarks
        self.loaded_laser_scan_data = None  
        self.load_laser_scan_data()
        self.convert_laser_scan_data_to_histograms()
        self.landmarks = self.convert_laser_scan_data_to_pose_array(self.loaded_laser_scan_data)
        self.landmarks_publisher.publish(self.landmarks)

        # Initialize particles
        self.particles = self.create_uniform_particles((-2.25, 2.25), (-2.25, 2.25), (0, 6.28), PARTICLES_NUM)
        self.particle_number = len(self.particles)
        self.particles_poses = self.convert_particles_to_pose_array(self.particles)
        self.particles_publisher.publish(self.particles_poses)
        # fmt: on

        self.robot = Robot()
        self.d_time_particles = 0.0
        self.prev_time_ros_particles = None
        self.pf_pose = PoseStamped()

        self.closest_point = np.array([0.0, 0.0, 0.0])
        self.closest_point_pose = PoseStamped()
        self.orient_diff = 0.0
        self.aligned_scand_data_to_ref_point = np.array([])
        self.diff_test = None
        self.distX = 0
        self.distY = 0
        self.true_dist_from_ref_points = np.array([])
        self.euclidains = np.array([0,0,0,0,0,0,0,0,0])

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
            "grid_updated_dqn4_new_0.25.pkl",
        )

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

    def histograms_differences_chi(self, current_histogram: np.ndarray) -> np.ndarray:
        result = []

        for laser_scan_data in self.loaded_laser_scan_data_histograms:
            # Compute chi-squared distance between histograms
            chi_squared = np.sum(((laser_scan_data.measurements - current_histogram) ** 2) /
                                    (laser_scan_data.measurements + current_histogram + 1e-10))  # Avoid division by zero
            result.append(chi_squared)

        return np.array(result)

    def histograms_differences_my(self, current_histogram: np.ndarray) -> np.ndarray:

        result = []

        for laser_scan_data in self.loaded_laser_scan_data_histograms:
            # Compute absolute difference between histograms
            difference = np.abs(laser_scan_data.measurements - current_histogram)
            total_difference = np.sum(difference)
            result = np.append(result, total_difference)

        return np.array(result)
    
    def get_closest_reference_point(self, difference: np.ndarray) -> np.ndarray:
        # Find the index of the smallest distance
        min_index = np.argmin(difference)
        # Retrieve the corresponding reference point
        closest_point = self.reference_points[min_index]
        return closest_point, min_index

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
        self.reference_points = np.array(reference_points)
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
        particles = []
        for _ in range(N):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            theta = np.random.uniform(theta_range[0], theta_range[1])
            theta %= 2 * np.pi
            particles.append(Particle(x, y, theta, 1.0 / N))
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

    def convert_to_pose(self, estimation, orienation):
        # Create a new Pose message
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "odom"

        # Set the estimated x and y positions
        pose_msg.pose.position.x = estimation[0]
        pose_msg.pose.position.y = estimation[1]
        pose_msg.pose.position.z = 0.0  # Assuming the robot is moving in a 2D plane

        # Set the orientation to no rotation
        q = self.get_quaternion_from_euler(0, 0, orienation)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]  # No rotation

        return pose_msg
    
    def calculate_orientation(self, current_scan_data, min_indx):

        # Get the reference data from the closest reference point
        ref_data = self.loaded_laser_scan_data[min_indx].measurements

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

    
    def calac_distX(self, data):


        indices = list(range(354, 360)) + list(range(0, 5))
        distances = [
            np.abs(data[i] - self.aligned_scand_data_to_ref_point[i]) for i in indices
        ]
        min_distance1 = min(distances)

        ############################################################
        distances = [
            np.abs(data[i] - self.aligned_scand_data_to_ref_point[i]) for i in range(174, 186)
        ]
        min_distance2 = min(distances)

        avg_dist = (min_distance1 + min_distance2) / 2
        return avg_dist
    
    def calac_distY(self, data):

        distances = [
            np.abs(data[i] - self.aligned_scand_data_to_ref_point[i]) for i in range(84, 96)
        ]
        min_distance1 = min(distances)


        ############################################################
        distances = [
            np.abs(data[i] - self.aligned_scand_data_to_ref_point[i]) for i in range(264, 276)
        ]
        min_distance2 = min(distances)

        avg_dist = (min_distance1 + min_distance2) / 2
        return avg_dist
    
    # PARTICLE FILTER ALGORITHM METHODS
    def predict(self, particles, current_time_ros, std=(0.0, 0.0)):

        pos_std = std[0]
        angle_std = std[1]

        # fmt: off
        if self.prev_time_ros_particles is not None:
            self.d_time_particles = (current_time_ros - self.prev_time_ros_particles).nanoseconds / 1e9

            # Calculate average rotation and translation from wheel velocities
            d_theta = ((self.robot.right_wheel_velocity_m_s - self.robot.left_wheel_velocity_m_s) / RobotConstants.DISTANCE_BETWEEN_WHEELS_M) * self.d_time_particles

            # For each particle, calculate its individual movement based on its own orientation
            for i in range(len(particles)):
                # Individual particle angle for this step
                particle_angle = particles[i].theta

                # Apply movement based on individual particle orientation
                # Note: d_x and d_y are computed individually for each particle
                d_x = ((self.robot.left_wheel_velocity_m_s + self.robot.right_wheel_velocity_m_s) / 2 * self.d_time_particles * np.cos(particle_angle))
                d_y = ((self.robot.left_wheel_velocity_m_s + self.robot.right_wheel_velocity_m_s) / 2 * self.d_time_particles * np.sin(particle_angle))

                # Update particle position and orientation
                particles[i].x += d_x + (randn() * pos_std)
                particles[i].y += d_y + (randn() * pos_std)
                particles[i].theta += d_theta + (randn() * angle_std)

                # Normalize the particle's orientation to remain within [-pi, pi]
                if particles[i].theta > np.pi:
                    particles[i].theta  -= 2 * np.pi
                elif particles[i].theta  < -np.pi:
                    particles[i].theta += 2 * np.pi

        # Update the previous time with the current time
        self.prev_time_ros_particles = current_time_ros
        # fmt: on

    def update(self, particles, z, R, landmarks):
        """
        Update particles' weights based on measurement z, noise R, and landmarks.

        Args:
            particles (list of Particle): The particles to update.
            z (np.array): Array of measurements.
            R (float): Measurement noise.
            landmarks (list of tuples): Positions of landmarks.
        """
        weights = np.array([particle.weight for particle in particles])
        for i, landmark in enumerate(landmarks):
            distances = np.array(
                [
                    np.linalg.norm([particle.x - landmark[0], particle.y - landmark[1]])
                    for particle in particles
                ]
            )
            weights *= scipy.stats.norm(distances, R).pdf(z[i])

        weights += 1.0e-300  # avoid round-off to zero
        weights /= np.sum(weights)  # normalize

        # Assign updated weights back to particles
        for i, particle in enumerate(particles):
            particle.weight = weights[i]

    def neff(self, particles):
        """
        Calculate the effective number of particles (N_eff), based on their weights.

        Args:
            particles (list of Particle): The particles whose effective number is to be calculated.

        Returns:
            float: The effective number of particles.
        """
        weights = np.array([particle.weight for particle in particles])
        return 1.0 / np.sum(np.square(weights))

    def systematic_resample(self, particles):
        """
        Performs systematic resampling on a list of Particle objects, returning only the indexes.

        Args:
            particles (list of Particle): The particles to resample.

        Returns:
            ndarray of ints: Array of indexes into the particles defining the resample.
        """
        N = len(particles)
        weights = np.array([particle.weight for particle in particles])

        positions = (random() + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    def resample_from_index(self, particles, indexes):
        """
        Resample the list of particles according to the provided indexes, adjusting weights.

        Args:
            particles (list of Particle): The particles to resample.
            indexes (np.ndarray): Array of indexes defining which particles to sample.

        Returns:
            list of Particle: A new list of particles resampled according to the indexes.
        """
        new_particles = [particles[index] for index in indexes]
        resampled_particles = [
            Particle(p.x, p.y, p.theta, 1.0 / len(indexes)) for p in new_particles
        ]

        return resampled_particles

    def estimate(self, particles):
        """
        Estimates the mean and variance of the particle positions.

        Args:
            particles (list): A list of particle objects.

        Returns:
            tuple: A tuple containing the mean and variance of the particle positions.
        """
        pos = np.array([[p._x, p._y] for p in particles])
        weights = np.array([p._weight for p in particles])
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

    def particles_callback(self) -> None:
        """
        Publishes the particle poses.
        """

        self.particles_publisher.publish(self.particles_poses)
        self.pf_estimated_position_publisher.publish(self.pf_pose)
        self.closest_point_publisher.publish(self.closest_point_pose)

    def logger_callback(self) -> None:

        # fmt: off
        clear = lambda: os.system("clear")
        clear()
        self.get_logger().info("                     True Position:" + self.robot.true_position())
        self.get_logger().info("       Estimated Position Odometry:" + self.robot.odometry_position())
        self.get_logger().info("Estimated Position Particle Filter:" + self.robot.particle_filter_position())
        self.get_logger().info("Wheels Velocities:" + self.robot.wheel_velocities_m_s())
        # print(f"Chi-Squared Distances: {self.hist_diff_chi}")
        # print(f"Histogram Differences: {self.hist_diff_my}")
        self.get_logger().info(f"Closest Reference Point:(X:{self.closest_point[0]:.3f} [m], Y:{self.closest_point[1]:.3f} [m], θ: {self.closest_point[2]:.3f} [deg])")
        self.get_logger().info(f"Orientation Difference: {self.orient_diff:.3f} [deg]")
        #diff
        self.get_logger().info(f"Distance X: {self.distY:.3f} [m]")
        self.get_logger().info(f"Distance Y: {self.distX:.3f} [m]")
        self.get_logger().info(f"Euclidean Distance: {np.linalg.norm([self.distX, self.distY]):.3f} [m]")

        for i, dist in enumerate(self.true_dist_from_ref_points):
            self.get_logger().info(f"True Distance from Reference Point {i+1}: {dist:.3f} [m], Laser Distance: {self.euclidains[i]:.3f} [m]")

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

        # Particle Filter Algorithm
        robot_pos = np.array(
            [self.robot._robot_x_true, self.robot._robot_y_true]
        )

        # Calculate the distance from the robot's position to each landmark
        zs = np.linalg.norm(self.reference_points[:,:2] - robot_pos, axis=1)
        self.true_dist_from_ref_points = zs

        # if self.robot.is_robot_moving():
        self.predict(self.particles, current_time_ros, std=(0.025, 0.025))
        self.particles_poses = self.convert_particles_to_pose_array(self.particles)
        self.update(self.particles, self.euclidains, 0.05, self.reference_points)

        if self.neff(self.particles) < len(self.particles) / 2:
            indexes = self.systematic_resample(self.particles)
            self.particles = self.resample_from_index(self.particles, indexes)
            self.particles_poses = self.convert_particles_to_pose_array(self.particles)
            assert np.allclose(
                np.array([p.weight for p in self.particles]),
                1 / len(self.particles),
            )

        mean, var = self.estimate(self.particles)
        self.robot.set_estimated_position_pf(mean[0], mean[1], 0)
        self.pf_pose = self.convert_to_pose(mean, 0)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        self.scan_callback_started = True
        self.robot.raw_scan = msg.ranges
        self.current_histogram, self.bin_edges = np.histogram(self.robot.raw_scan, range=HISTOGRAM_RANGE, bins=BINS)
        self.hist_diff_chi = self.histograms_differences_chi(self.current_histogram)
        self.hist_diff_my = self.histograms_differences_my(self.current_histogram)

        self.closest_point, min_indx = self.get_closest_reference_point(self.hist_diff_chi)
        self.closest_point_pose = self.convert_to_pose(self.closest_point, 3.14/2)
        self.orient_diff = self.calculate_orientation(self.robot.raw_scan, min_indx)
        self.aligned_scand_data_to_ref_point = np.roll(self.robot.raw_scan, int(-(self.closest_point[2]-self.orient_diff)))        
        self.distX = self.calac_distX(self.loaded_laser_scan_data[min_indx].measurements)
        self.distY = self.calac_distY(self.loaded_laser_scan_data[min_indx].measurements)

        #Cacluate distX and disy for every loaded laser scan data
        self.distsX = np.array([self.calac_distX(data.measurements) for data in self.loaded_laser_scan_data])
        self.distsY = np.array([self.calac_distY(data.measurements) for data in self.loaded_laser_scan_data])
        # Now calculate the euclidean distance
        self.euclidains = np.linalg.norm(np.array([self.distsX, self.distsY]), axis=0)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    print(node.load_laser_scan_data)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
