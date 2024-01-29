#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_laser_scan_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Combine laser scan data from multiple files.")
    parser.add_argument("file_path", help="Path pattern of the files to combine", type=str)
    parser.add_argument("num_files", help="Number of files to combine", type=int)
    parser.add_argument("destination", help="Path of the destination file", type=str)
    args = parser.parse_args()

    # Generating file paths
    files = [args.file_path.format(i) for i in range(1, args.num_files + 1)]
    all_data = []

    # Load data from all specified files
    for file in files:
        laserscan_data = load_laser_scan_data(file)
        all_data.extend(laserscan_data)

    # Save the combined list of LaserScanData objects to a new pickle file
    with open(args.destination, 'wb') as file:
        pickle.dump(all_data, file)

    # Extracting only coordinates for plotting
    all_coords = np.array([data.coords for data in all_data])

    # Plotting the coordinates
    fig, ax = plt.subplots()
    ax.scatter(all_coords[:, 0], all_coords[:, 1], color="#61AFEF", s=12)
    ax.set_xlabel('X [meters]')
    ax.set_ylabel('Y [meters]')
    ax.set_title('Coordinates from Laser Scan Data')

    # Set the window title
    canvas = FigureCanvasTkAgg(fig)
    canvas.get_tk_widget().master.title("Laser Scan Data Visualization")

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
