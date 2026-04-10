"""ROS2 launch file for the SRL inference node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config_path", default_value="", description="Path to YAML model config"),
        DeclareLaunchArgument("checkpoint_path", default_value="", description="Path to checkpoint file"),
        DeclareLaunchArgument("device", default_value="cpu", description="PyTorch device"),
        DeclareLaunchArgument("hz", default_value="20.0", description="Inference frequency in Hz"),
        Node(
            package="srl",
            executable="rl_inference_node",
            name="rl_inference_node",
            output="screen",
            parameters=[{
                "config_path": LaunchConfiguration("config_path"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "device": LaunchConfiguration("device"),
                "hz": LaunchConfiguration("hz"),
            }],
        ),
    ])
