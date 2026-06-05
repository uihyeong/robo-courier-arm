from setuptools import setup
import os
from glob import glob

package_name = 'courier_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'arm_elevator     = courier_arm.arm_elevator:main',
            'arm_delivery     = courier_arm.arm_delivery:main',
            'contact_detector = courier_arm.contact_detector:main',
            'detect_room_sign = courier_arm.detect_room_sign:main',
            'scout            = courier_arm.scout:main',
        ],
    },
)
