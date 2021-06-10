#!/usr/bin/env python3

from setuptools import setup, find_packages

version = "1.0.0"


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

with open("LICENSE", encoding="UTF-8") as f:
    license = f.read()

setup(
    name="EVE",
    version=version,
    description="Package for EVE model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    entry_points={"console_scripts": ["eve=EVE.__main__:main"]},
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.7.0",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "rich",
        "tqdm",
        "seaborn",
    ],
)
