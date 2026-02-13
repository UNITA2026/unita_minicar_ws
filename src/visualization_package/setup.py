from setuptools import find_packages, setup

package_name = 'visualization_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='young',
    maintainer_email='young@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ultrasonic_range_publisher = visualization_package.ultrasonic_range_publisher:main',
            'ultrasonic_visualizer = visualization_package.ultrasonic_visualizer:main'
        ],
    },
)
