from importlib.metadata import files

from setuptools import find_packages, setup
from setuptools.command.install import install

setup(
    name='phenigraph',
    packages=find_packages(include=['phenigraph']),
    version='0.2.0',
    install_requires=["numpy","matplotlib", "PyQt5", "scipy"],
    description='A library to plot, store and replot matplotlib plot easly and reproducibly',
    author='Paul Huet',
    readme="README.md",
    license_files="LICENSE",
    setup_requires=['pytest-runner'],
)
