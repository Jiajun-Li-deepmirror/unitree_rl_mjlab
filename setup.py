"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco==3.5.0",
    "mujoco-warp==3.5.0",
    # mjlab 1.2.0 only declares lower bounds; pinned to the env the boxes train in
    # (warp-lang 1.12.0, mujoco 3.5.0, 2026-11-10)
    "warp-lang==1.12.0",
    # Imported by mjlab.terrains but not declared by mjlab.
    "scipy>=1.15.0",  # for supporting Python >= 3.10
    "pillow>=10.1.0",  # readable comparison legend in policy videos
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
