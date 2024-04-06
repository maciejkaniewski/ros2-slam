import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from ament_index_python.packages import get_package_share_directory
from utils.pgm_map_loader import PgmMapLoader
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

import numpy as np


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter_new")
        self.get_logger().info("ParticleFilter node started.")

        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/custom_occupancy_grid_map", 10
        )

        self.load_map()

        # Create OccupancyGrid message
        self.occupancy_grid_msg = OccupancyGrid()
        self.set_occupancy_grid_info()

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

    def set_occupancy_grid_info(self):

        inverted_img = 255 - self.pgm_map.img
        scaled_img =  np.flip((inverted_img * (100.0 / 255.0)).astype(np.int8), 0)
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


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
