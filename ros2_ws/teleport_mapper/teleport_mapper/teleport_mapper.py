#!/usr/bin/env python3

import os

import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node

# TODO: Handle this in launch file
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    current_dir,
    "../../../../turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf",
)
MODEL_PATH = os.path.normpath(MODEL_PATH)


class TeleportMapper(Node):
    def __init__(self):
        super().__init__("teleport_mapper")
        self.get_logger().info("TeleportMapper node started.")

        self.declare_parameter("model_file_path", MODEL_PATH)
        self.model_file_path = self.get_parameter("model_file_path").value
        self._read_model_file(self.model_file_path)

        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.entity_name = "burger"

    def _wait_for_service(self, service_client, service_name):
        """
        Waits for a service to become available.

        Args:
            service_client: The service client object.
            service_name: The name of the service.
        """

        while not service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f"Service {service_name} not available, waiting again..."
            )

    def _handle_service_result(self, future, service_name: str) -> None:
        """
        Handles the result of a service call.

        Args:
            future (Future): The future object representing the result of the service call.
            service_name (str): The name of the service.
        """

        if future.result() is not None:
            self.get_logger().info(f"{service_name} completed successfully")
        else:
            self.get_logger().error(
                f"Exception while calling service: {future.exception()}"
            )

    def _read_model_file(self, model_file_path: str) -> None:
        """
        Reads the contents of a model file and assigns it to the `model` attribute.

        Args:
            model_file_path (str): The path to the model file.
        """

        with open(model_file_path, "r") as file:
            self.model = file.read()

    def spawn_enity(self, x_set: float, y_set: float) -> None:
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
        rclpy.spin_until_future_complete(self, future)
        self._handle_service_result(future, "/spawn_entity")

    def delete_entity(self):
        """
        Deletes the entity with the specified name.
        """

        self._wait_for_service(self.delete_entity_client, "/delete_entity")
        request = DeleteEntity.Request()
        request.name = self.entity_name
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self._handle_service_result(future, "/delete_entity")


def main(args=None):
    rclpy.init(args=args)
    node = TeleportMapper()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
