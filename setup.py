from setuptools import find_packages, setup

setup(
    name='pysotopelib',
    packages=find_packages(),
    version='0.1.0',
    description='Python library for nuclear physics',
    author='James Keatings',
    install_requires=['numpy','matplotlib','scipy']
)
