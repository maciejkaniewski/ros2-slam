import matplotlib.image as mpimg
import numpy as np
import yaml


class PgmMapLoader:
    """
    A class for loading map data from YAML and PGM files.

    Args:
        yaml_file_path (str): The file path to the YAML metadata file.
        pgm_file_path (str): The file path to the PGM image file.

    Attributes:
        yaml_file_path (str): The file path to the YAML metadata file.
        pgm_file_path (str): The file path to the PGM image file.
        metadata (dict): The metadata loaded from the YAML file.
        img (numpy.ndarray): The image loaded from the PGM file.
        obstacle_coords (list): The coordinates of the obstacles in the map.

    """

    def __init__(self, yaml_file_path, pgm_file_path):
        self.yaml_file_path = yaml_file_path
        self.pgm_file_path = pgm_file_path
        self.metadata = self.load_yaml_metadata()
        self.img, self.obstacle_coords, self.obstacle_radius = self.load_map()

    def load_yaml_metadata(self):
        """
        Load the metadata from the YAML file.

        Returns:
            dict: The metadata loaded from the YAML file.

        """
        with open(self.yaml_file_path, "r") as file:
            metadata = yaml.safe_load(file)
        return metadata

    def load_map(self) -> None:
        """
        Load the map image and obstacle coordinates.

        Returns:
            tuple: A tuple containing the map image (numpy.ndarray) and the obstacle coordinates (list).

        """
        img = mpimg.imread(self.pgm_file_path)
        resolution = self.metadata["resolution"]
        origin = self.metadata["origin"]
        height, width = img.shape
        y_coords, x_coords = np.where(img < 250)
        map_x_coords = x_coords * resolution + origin[0]
        map_y_coords = (height - y_coords) * resolution + origin[1]
        map_x_coords = map_x_coords + resolution / 2
        map_y_coords = map_y_coords - resolution / 2
        obstacle_coords = list(zip(map_x_coords, map_y_coords))
        obstacle_radius = resolution / 2
        return img, obstacle_coords, obstacle_radius

    def get_obstacle_coordinates(self):
        """
        Get the coordinates of the obstacles in the map.

        Returns:
            list: The coordinates of the obstacles in the map.

        """
        return self.obstacle_coords

    def get_obstacle_radius(self):
        """
        Get the radius of the obstacles in the map.

        Returns:
            float: The radius of the obstacles in the map.

        """
        return self.obstacle_radius
