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
    author="Kevin Menden",
    author_email="kevin.menden@t-online.de",
    url="https://github.com/KevinMenden/scaden",
    license="MIT License",
    entry_points={"console_scripts": ["scaden=scaden.__main__:main"]},
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.7.0",
    install_requires=[
        "pytorch=1.7",
        "cudatoolkit=11.0",
        "scikit-learn=0.24.1",
        "numpy=1.20.1",
        "pandas=1.2.4",
        "scipy=1.6.2",
        "matplotlib",
        "rich",
        "tqdm",
        "seaborn",
    ],
)
