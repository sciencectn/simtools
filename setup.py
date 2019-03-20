
from setuptools import setup, find_packages

"""
simutils: numpy and timing tools to help with creating simulations
"""

setup(
    name="simtools",
    version="0.0",
    packages=find_packages(),
    install_requires=["pyquaternion"]
)


