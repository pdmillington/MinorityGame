#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup configuration for MinorityGame package
"""

from setuptools import setup, find_packages

setup(
    name="MinorityGame",
    version="0.1.0",
    description="Agent-Based Model for Market Structure Research",
    author="Peter Millington",
    packages=find_packages(where="MG"),
    package_dir={"": "MG"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ]
    },
)