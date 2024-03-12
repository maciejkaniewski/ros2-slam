import os
from glob import glob

from setuptools import find_packages, setup

package_name = "histogram_filter"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "data"), glob("data/*.pkl")),
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
            "histogram_filter = histogram_filter.histogram_filter:main",
            "reference_grid = histogram_filter.reference_grid:main",
        ],
    },
)
