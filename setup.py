from setuptools import setup, find_packages
setup(
    name="nnvis",
    version='0.0.4',
    packages=['nnvis', 'nnvis.hooks'],
    package_dir={'nnvis': 'src/nnvis'},
)