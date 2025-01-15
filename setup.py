
from setuptools import find_packages, setup

setup(
    name='phenigraph',
    packages=find_packages(include=['phenigraph']),
    version='0.2.0',
    description='A library to plot, store and replot matplotlib plot easly and reproducibly',
    author='Paul Huet',
    readme="README.md",
    license="LICENSE",
    setup_requires=['pytest-runner'],
)
