from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config_path",
        default_value="",
        description="Path to pf_config.yaml",
    )

    node = Node(
        package="particle_filter_loc",
        executable="pf_geo_loc_node",
        name="pf_geo_loc_node",
        parameters=[{"config_path": LaunchConfiguration("config_path")}],
        output="screen",
    )

    return LaunchDescription([config_arg, node])
