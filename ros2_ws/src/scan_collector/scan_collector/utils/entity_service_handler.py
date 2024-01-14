#!/usr/bin/env python3

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node
from rclpy.task import Future


class EntityServiceHandler:
    def __init__(self, node: Node):
        """
        Initializes an instance of the EntityServiceHandler class.

        Args:
            node (Node): The ROS2 node object.
        """
        
        self.node = node
        self.spawn_entity_client = node.create_client(SpawnEntity, "/spawn_entity")
        self.delete_entity_client = node.create_client(DeleteEntity, "/delete_entity")
        self.entity_name = os.environ["TURTLEBOT3_MODEL"]
        self.model = None

        self._read_model_file()

    def _read_model_file(self) -> None:
        """
        Reads the contents of a model file.
        """
        
        model_folder = "turtlebot3_" + self.entity_name
        urdf_path = os.path.join(
            get_package_share_directory("turtlebot3_gazebo"),
            "models",
            model_folder,
            "model.sdf",
        )

        with open(urdf_path, "r") as file:
            self.model = file.read()

    def _wait_for_service(self, service_client, service_name):
        """
        Waits for a service to become available.

        Args:
            service_client: The service client object.
            service_name: The name of the service.
        """

        while not service_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                f"Service {service_name} not available, waiting again..."
            )

    def _handle_service_result(self, future: Future, service_name: str) -> None:
        """
        Handles the result of a service call.

        Args:
            future (Future): The future object representing the result of the service call.
            service_name (str): The name of the service.
        """

        if future.result() is not None:
            self.node.get_logger().info(f"{service_name} completed successfully")
        else:
            self.node.get_logger().error(
                f"Exception while calling service: {future.exception()}"
            )

    def spawn_entity(self, x_set: float, y_set: float) -> None:
        """
        Spawns an entity at the specified coordinates.

        Args:
            x_set (float): The x-coordinate of the entity's position.
            y_set (float): The y-coordinate of the entity's position.
        """

        self._wait_for_service(self.spawn_entity_client, "/spawn_entity")
        request = SpawnEntity.Request()
        request.name = self.entity_name
        request.xml = self.model
        request.robot_namespace = ""
        request.initial_pose = Pose(
            position=Point(x=x_set, y=y_set, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        request.reference_frame = "world"
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        self._handle_service_result(future, "/spawn_entity")

    def delete_entity(self):
        """
        Deletes the entity.
        """

        self._wait_for_service(self.delete_entity_client, "/delete_entity")
        request = DeleteEntity.Request()
        request.name = self.entity_name
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        self._handle_service_result(future, "/delete_entity")
