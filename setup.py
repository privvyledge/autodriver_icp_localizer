from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autodriver_icp_localizer'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz'))
]

# to recursively add all launch files and keep the subdirectory structure
for root, dirs, files in os.walk('launch'):
    # Get the relative path for each subdirectory
    install_dir = os.path.join('share', package_name, root)
    # Get the list of all launch files in the current subdirectory
    launch_files = [os.path.join(root, f) for f in files if f.endswith(
            ('.launch.py', '.launch.xml', '.launch.yml', '.launch.yaml'))]
    if launch_files:
        # Add each subdirectory and its files to data_files
        data_files.append((install_dir, launch_files))


# to include all directories and subdirectories in a folder
def package_files(data_files, directory_list):
    paths_dict = {}

    for directory in directory_list:

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files


setup(
        name=package_name,
        version='0.0.0',
        packages=find_packages(exclude=['test']),
        data_files=package_files(data_files, ['data/']),
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='Boluwatife Olabiran',
        maintainer_email='bso19a@fsu.edu',
        description='TODO: Package description',
        license='TODO: License declaration',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'icp_localizer = autodriver_icp_localizer.icp_localization_node:main',
            ],
        },
)
