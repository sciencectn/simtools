
from setuptools import setup, find_packages

"""
simutils: numpy and timing tools to help with creating simulations
"""

setup(
    name="particle-simtools",
    author="Chris Taylor",
    author_email="sciencectn@gmail.com",
    url="https://github.com/sciencectn/simtools",
    description="A collection of functions to help with simulating particle models",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["pyquaternion"]
)

