"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    # mjlab 1.2.0 only declares lower bounds; newer releases of these break it.
    "mujoco==3.5.0",
    "warp-lang==1.12.1",
    "wandb<0.25",
    # Imported by mjlab.terrains but not declared by mjlab.
    "scipy",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
