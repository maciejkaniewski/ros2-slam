#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from .utils.entity_service_handler import EntityServiceHandler
from .utils.physics_service_handler import PhysicsServiceHandler


class ScanCollector(Node):
    def __init__(self):
        """
        Initializes an instance of the ScanCollector class.
        """

        super().__init__("scan_collector")
        self.get_logger().info("ScanCollector node started.")
        self.__entity_service_handler = EntityServiceHandler(self)
        self.__physics_service_handler = PhysicsServiceHandler(self)

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


def main(args=None):
    rclpy.init(args=args)
    node = ScanCollector()
    for i in range(1000):
        print(f"Iteraion {i}")
        node.pause_physics()
        node.spawn_entity(0.0, 0.0)
        node.unpause_physics()
        # Collect laser data
        node.pause_physics()
        node.delete_entity()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
