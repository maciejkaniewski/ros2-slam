#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from .utils.entity_service_handler import EntityServiceHandler


class ScanCollector(Node):
    def __init__(self):
        """
        Initializes an instance of the ScanCollector class.
        """

        super().__init__("scan_collector")
        self.get_logger().info("ScanCollector node started.")
        self.__entity_service_handler = EntityServiceHandler(self)

    def spawn_entity(self, x_set: float, y_set: float) -> None:
        """
        Spawns an entity at the specified coordinates.

        Args:
            x_set (float): The x-coordinate of the entity.
            y_set (float): The y-coordinate of the entity.
        """

        self.__entity_service_handler.spawn_entity(x_set, y_set)

    def delete_entity(self) -> None:
        """
        Deletes the entity using the entity service handler.
        """

        self.__entity_service_handler.delete_entity()


def main(args=None):
    rclpy.init(args=args)
    node = ScanCollector()
    for i in range(100):
        print(f"Spawning entity {i}")
        node.spawn_entity(0.0, 0.0)
        node.delete_entity()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
