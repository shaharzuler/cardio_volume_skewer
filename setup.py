from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]


setup(
    name='cardio_volume_skewer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Shahar Zuler',
    author_email='shahar.zuler@gmail.com',
    description='A package that creates a synthetic 4D sequence of a cardiac CT from a single 3D image.',
    url='https://github.com/shaharzuler/cardio_volume_skewer',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)