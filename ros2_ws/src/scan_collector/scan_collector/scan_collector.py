#!/usr/bin/env python3

import pickle
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from utils.entity_service_handler import EntityServiceHandler
from utils.laser_scan_data import LaserScanData
from utils.map_loader import MapLoader
from utils.physics_service_handler import PhysicsServiceHandler

TURTLEBOT3_BURGER_WIDTH_M = 0.178
TURTLEBOT3_BURGER_LENGTH_M = 0.138
TURTLEBOT3_BURGER_CENTER_OFFSET_M = 0.032


class ScanCollector(Node):
    def __init__(self):
        """
        Initializes an instance of the ScanCollector class.
        """

        super().__init__("scan_collector")
        self.get_logger().info("ScanCollector node started.")
        self.__entity_service_handler = EntityServiceHandler(self)
        self.__physics_service_handler = PhysicsServiceHandler(self)

        self.robot_x = 0.0
        self.robot_y = 0.0

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.laser_scan_data = np.array([])

        map_loader = MapLoader("/home/maciej/dqn_stage4.yaml", "/home/maciej/test.pgm")
        self.obstacle_coordinates = map_loader.get_obstacle_coordinates()
        self.obstacle_radius = map_loader.get_obstacle_radius()

    def spawn_entity(self, x_set: float, y_set: float) -> None:
        """
        Spawns an entity at the specified coordinates.

        Args:
            x_set (float): The x-coordinate of the entity.
            y_set (float): The y-coordinate of the entity.
        """

        self.robot_x = x_set
        self.robot_y = y_set
        self.__entity_service_handler.spawn_entity(x_set, y_set)

    def delete_entity(self) -> None:
        """
        Deletes the entity using the entity service handler.
        """

        self.__entity_service_handler.delete_entity()

    def pause_physics(self) -> None:
        """
        Pauses the physics using the physics service handler.
        """

        self.__physics_service_handler.pause_physics()

    def unpause_physics(self) -> None:
        """
        Unpauses the physics using the physics service handler.
        """

        self.__physics_service_handler.unpause_physics()

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback function for handling laser scan messages.

        Args:
            msg (LaserScan): The incoming laser scan message.
        """

        scan_data = LaserScanData((self.robot_x, self.robot_y), msg.ranges)
        self.laser_scan_data = np.append(self.laser_scan_data, scan_data)

    def dump_laser_scan_data(self, file_path: str) -> None:
        """
        Dumps the laser scan data to a file using pickle.

        Args:
            file_path (str): The file path to the output file.
        """

        with open(file_path, "wb") as file:
            pickle.dump(self.laser_scan_data, file)

    def spawn_robot_across_map(
        self, step: float, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """
        Spawns the robot across the map with a specified step, avoiding obstacles.

        Args:
            step (float): The step size in meters for iterating across the map.
            x_min (float): The minimum x-coordinate for spawning the robot.
            x_max (float): The maximum x-coordinate for spawning the robot.
            y_min (float): The minimum y-coordinate for spawning the robot.
            y_max (float): The maximum y-coordinate for spawning the robot.
        """

        for y in np.arange(y_min, y_max + step, step):
            for x in np.arange(x_min, x_max + step, step):
                if not self.is_position_near_obstacle(x, y):
                    self.pause_physics()
                    self.spawn_entity(x, y)
                    self.unpause_physics()
                    rclpy.spin_once(self)
                    self.pause_physics()
                    #time.sleep(1)
                    self.delete_entity()

    def is_position_near_obstacle(self, x: float, y: float) -> bool:
        """
        Checks if the given position is near any obstacle, considering the robot's size.

        Args:
            x (float): The x-coordinate of the robot's center.
            y (float): The y-coordinate of the robot's center.

        Returns:
            bool: True if the position is near an obstacle, False otherwise.
        """

        robot_half_width = TURTLEBOT3_BURGER_WIDTH_M / 2
        robot_top_length = (
            TURTLEBOT3_BURGER_LENGTH_M / 2 - TURTLEBOT3_BURGER_CENTER_OFFSET_M
        )
        robot_bottom_length = (
            TURTLEBOT3_BURGER_LENGTH_M / 2 + TURTLEBOT3_BURGER_CENTER_OFFSET_M
        )

        for obstacle_x, obstacle_y in self.obstacle_coordinates:
            y_distance = abs(y - obstacle_y)
            threshold = (
                robot_top_length + self.obstacle_radius
                if y < obstacle_y
                else robot_bottom_length - self.obstacle_radius
            )

            if (
                y_distance < threshold
                and abs(x - obstacle_x) < robot_half_width + self.obstacle_radius
            ):
                return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = ScanCollector()
    node.spawn_robot_across_map(step=0.05, x_min=-3, x_max=-1.95, y_min=-3, y_max=3)
    node.get_logger().info(
        f"Dumping laser scan data to file... for [{node.robot_x}, {node.robot_y}]"
    )
    node.dump_laser_scan_data("/home/maciej/data/test_6.pkl")
    node.destroy_node()


if __name__ == "__main__":
    main()
