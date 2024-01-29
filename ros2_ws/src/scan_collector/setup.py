import os
from glob import glob

from setuptools import find_packages, setup

package_name = "scan_collector"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.pgm")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maciej",
    maintainer_email="maciejkaniewski1999@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scan_collector = scan_collector.scan_collector:main",
        ],
    },
)
