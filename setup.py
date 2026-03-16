from setuptools import setup, find_packages

package_name = "particle_filter_loc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/pf_config.yaml"]),
        ("share/" + package_name + "/launch", ["launch/pf_geo_loc.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "pf_geo_loc_node = particle_filter_loc.ros2_pf_node:main",
        ],
    },
)
