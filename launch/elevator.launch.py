from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='courier_arm',
            executable='arm_elevator',
            name='arm_elevator',
            output='screen',
        ),
        Node(
            package='courier_arm',
            executable='contact_detector',
            name='contact_detector',
            output='screen',
        ),
    ])
