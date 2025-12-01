from setuptools import find_packages, setup

package_name = 'web_pub'

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
    maintainer='jejun',
    maintainer_email='klan1782plan@gmailc.om',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'web_pub=web_pub.web_pub:main',
            'web_sub=web_pub.web_sub:main'
        ],
    },
)
