from setuptools import setup, find_packages

setup(
    name='tello_obstacle_detection',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'djitellopy',
        'opencv-python',
        'numpy',
    ],
)
