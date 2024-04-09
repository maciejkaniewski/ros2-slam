import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from ament_index_python.packages import get_package_share_directory
from utils.pgm_map_loader import PgmMapLoader
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import numpy as np
from numpy.random import uniform
from numpy.random import randn
from rclpy.time import Time
from nav_msgs.msg import Odometry
import pickle
import scipy.stats
from numpy.random import random
import time

from sensor_msgs.msg import LaserScan
from utils.laser_scan_data import LaserScanData

DISTANCE_BETWEEN_WHEELS_M = 0.160
WHEEL_RADIUS_M = 0.033

LEFT_WHEEL = 0
RIGHT_WHEEL = 1

PARTICLES_NUM = 2000

HISTOGRAM_RANGE = (0.0, 3.5)
BINS = 15


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter_new")
        self.get_logger().info("ParticleFilter node started.")

        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/custom_occupancy_grid_map", 10
        )

        # publish the current particle cloud
        self.particles_publisher = self.create_publisher(PoseArray, "/particles", 10)

        self.landmarks_publisher = self.create_publisher(PoseArray, "/landmarks", 10)
        self.pf_pose_publisher = self.create_publisher(PoseStamped, "/pf_pose", 10)

        self.joint_states_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )
        self.current_histogram = np.array([])
        self.loaded_laser_scan_data_histograms = np.array([])

        self.scan_callback_started = False
        self.particle_histograms = None

        self.load_map()
        self.load_laser_scan_data()
        self.convert_laser_scan_data_to_histograms()

        # Create OccupancyGrid message
        self.occupancy_grid_msg = OccupancyGrid()
        self.set_occupancy_grid_info()

        self.particles = self.create_uniform_particles(
            (-2.25, 2.25), (-2.25, 2.25), (0, 6.28), PARTICLES_NUM
        )

        self.pose_array = self.convert_particles_to_pose_array(self.particles)
        self.pf_pose = self.convert_to_pose_msg(self.particles.mean(axis=0))
        self.pf_pose_publisher.publish(self.pf_pose)
        self.particles_publisher.publish(self.pose_array)

        self.timer_particles = self.create_timer(0.1, self.callback_particles)

        self.timer_particles_ref = self.create_timer(0.1, self.callback_particles_ref)

        # Time
        self.d_time = 0.0
        self.prev_time_ros = None

        self.d_time_particles = 0.0
        self.prev_time_ros_particles = None

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

        self.initial_position_saved = False

        self.robot_x_true = 0.0
        self.robot_y_true = 0.0
        self.robot_theta_true_rad = 0.0
        self.robot_theta_true_deg = 0.0

        # Perticle Filter Estimated Position
        self.robot_x_estimated_pf = 0.0
        self.robot_y_estimated_pf = 0.0
        self.robot_theta_estimated_rad_pf = 0.0

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        # fmt: off
        self.scan_callback_started = True
        self.current_scan_data = LaserScanData(coords=(self.robot_x_estimated_v, self.robot_y_estimated_v), measurements=msg.ranges)
        self.current_histogram, self.bin_edges = np.histogram(self.current_scan_data.measurements, range=HISTOGRAM_RANGE, bins=BINS)

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

    def histograms_differences(self, current_histogram: np.ndarray) -> np.ndarray:

        result = np.array([])

        for laser_scan_data in self.loaded_laser_scan_data_histograms:
            # Compute absolute difference between histograms
            difference = np.abs(laser_scan_data.measurements - current_histogram)
            total_difference = np.sum(difference)
            result = np.append(result, total_difference)

        return result

    def convert_particles_to_pose_array(self, particles):
        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"

        for particle in particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            # Convert heading to quaternion
            q = self.get_quaternion_from_euler(0, 0, particle[2])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

        return pose_array

    def convert_to_pose_msg(self, mu):
        # Create a new Pose message
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "odom"

        # Set the estimated x and y positions
        pose_msg.pose.position.x = mu[0]
        pose_msg.pose.position.y = mu[1]
        pose_msg.pose.position.z = 0.0  # Assuming the robot is moving in a 2D plane

        # Set the orientation to no rotation
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0  # No rotation

        return pose_msg

    def callback_particles(self):
        self.particles_publisher.publish(self.pose_array)
        self.pf_pose_publisher.publish(self.pf_pose)

    def callback_particles_ref(self):

        if np.isclose(self.left_wheel_velocity_m_s, 0, atol=1e-3) and np.isclose(
            self.right_wheel_velocity_m_s, 0, atol=1e-3
        ):
            self.move_particles_to_closest_reference()
            self.particles_publisher.publish(self.pose_array)

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

        laser_scan_data_path = "/home/mkaniews/Desktop/laser_scan_data_test.pkl"

        with open(laser_scan_data_path, "rb") as file:
            self.loaded_laser_scan_data = pickle.load(file)

        self.reference_points = np.array(
            [data.coords for data in self.loaded_laser_scan_data]
        )
        print(self.reference_points)
        # self.reference_points = np.array(self.reference_points)[:, :2]
        self.weights = np.ones(PARTICLES_NUM) / PARTICLES_NUM

        pose_array = PoseArray()
        pose_array.header.frame_id = "odom"

        for reference_point in self.reference_points:
            pose = Pose()
            pose.position.x = reference_point[0]
            pose.position.y = reference_point[1]
            # Convert heading to quaternion
            q = self.get_quaternion_from_euler(0, 0, np.deg2rad(reference_point[2]))
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

        self.landmarks_publisher.publish(pose_array)
        self.reference_points = np.array(self.reference_points)[:, :2]

    def set_occupancy_grid_info(self):

        inverted_img = 255 - self.pgm_map.img
        scaled_img = np.flip((inverted_img * (100.0 / 255.0)).astype(np.int8), 0)
        self.map_data = scaled_img.ravel()

        map_width = self.pgm_map.width * self.pgm_map.resolution
        map_height = self.pgm_map.height * self.pgm_map.resolution
        self.occupancy_grid_msg.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id="odom"
        )
        self.occupancy_grid_msg.info.width = self.pgm_map.width
        self.occupancy_grid_msg.info.height = self.pgm_map.height
        self.occupancy_grid_msg.info.resolution = self.pgm_map.resolution
        self.occupancy_grid_msg.info.origin = Pose()
        self.occupancy_grid_msg.info.origin.position.x = self.pgm_map.origin[0]
        self.occupancy_grid_msg.info.origin.position.y = self.pgm_map.origin[1]
        self.occupancy_grid_msg.info.origin.position.z = 0.0
        self.occupancy_grid_msg.info.origin.orientation.w = 1.0
        self.occupancy_grid_msg.data = self.map_data.tolist()

        self.map_publisher.publish(self.occupancy_grid_msg)

    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi
        return particles

    def predict(self, particles, current_time_ros, std=(0.1, 0.025/1.25)):

        # fmt: off
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
                particles[i, 2] += d_theta + (randn() * std[0])

                # Normalize the particle's orientation to remain within [-pi, pi]
                if particles[i, 2] > np.pi:
                    particles[i, 2] -= 2 * np.pi
                elif particles[i, 2] < -np.pi:
                    particles[i, 2] += 2 * np.pi

        # Update the previous time with the current time
        self.prev_time_ros_particles = current_time_ros
        # fmt: on

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

            self.robot_theta_estimated_deg_v = np.degrees(
                self.robot_theta_estimated_rad_v
            )

        # Update the previous time with the current time
        self.prev_time_ros = current_time_ros

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

    def update(self, particles, weights, z, R, landmarks):
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.0e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize

    def update_hist(self, particles, weights, z, R):
        # for each particle check the difference between its histogram and the current histogram
        for i in range(len(particles)):
            # Compute absolute difference between histograms
            difference = np.abs(self.particle_histograms[i] - self.current_histogram)
            total_difference = np.sum(difference)
            # the smaller the difference, the higher the weight
            weights[i] = np.exp(-total_difference)
        
        weights += 1.0e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize

    def neff(self, weights):
        return 1.0 / np.sum(np.square(weights))

    def resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights.resize(len(particles))
        weights.fill(1.0 / len(weights))

    def systematic_resample(self, weights):
        """Performs the systemic resampling algorithm used by particle filters.

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

        indexes = np.zeros(N, "i")
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
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var

    def move_particles_to_closest_reference(self):
        if self.particles is None or len(self.reference_points) == 0:
            return  # No operation if particles or reference points are not set

        # Calculate squared distances from each particle to each reference point
        distances = np.sqrt(
            (
                (
                    self.particles[:, np.newaxis, :2]
                    - self.reference_points[np.newaxis, :, :]
                )
                ** 2
            ).sum(axis=2)
        )

        # Find the index of the closest reference point for each particle
        closest_indices = np.argmin(distances, axis=1)

        self.particle_histograms = []  # Reset or initialize the list of histograms
        for i, idx in enumerate(closest_indices):
            self.particles[i, :2] = self.reference_points[idx]
            self.particle_histograms.append(
                self.loaded_laser_scan_data_histograms[idx].measurements
            )

        # Optionally, convert particle_histograms to a numpy array for efficient processing
        self.particle_histograms = np.array(self.particle_histograms)

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
        # Your robot's current estimated position
        # robot_pos = np.array([self.robot_x_estimated_v, self.robot_y_estimated_v])
        # #print(self.particle_histograms)
        # # # Calculate the distance from the robot's position to each landmark
        # zs = np.linalg.norm(self.reference_points  - robot_pos, axis=1)
        # print(zs)

        if self.scan_callback_started:
            zs = self.histograms_differences(self.current_histogram)
            print(zs)

        print(f"Particle Hist {self.particle_histograms}")
        # #print(f"Loaded Hist {self.loaded_laser_scan_data_histograms}")
        # # print(f"Particles {self.particles}")
        # # print(f"Particles Sliced {self.particles[:, 0:2]}")
        print(f"Current Hist {self.current_histogram}")

        self.predict(self.particles, current_time_ros)
        self.pose_array = self.convert_particles_to_pose_array(self.particles)

        # #self.update(self.particles, self.weights, zs, 0.5, (np.array(self.reference_points)[:, :2]))
        if self.scan_callback_started:
            self.update_hist(self.particles, self.weights, zs, 0.5)
        self.pose_array = self.convert_particles_to_pose_array(self.particles)
        print(self.weights)
        if self.neff(self.weights) < PARTICLES_NUM/2:
            indexes = self.systematic_resample(self.weights)
            self.resample_from_index(self.particles, self.weights, indexes)
            assert np.allclose(self.weights, 1/PARTICLES_NUM)
        self.pose_array = self.convert_particles_to_pose_array(self.particles)

        mu , var = self.estimate(self.particles, self.weights)
        self.robot_x_estimated_pf = mu[0]
        self.robot_y_estimated_pf = mu[1]
        self.pf_pose = self.convert_to_pose_msg(mu)
        self.pf_pose_publisher.publish(self.pf_pose)
        # # fmt: on


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
