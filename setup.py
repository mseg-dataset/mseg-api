#!/usr/bin/env python
"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from os import environ  # check shell under windows (allow gitbash)
from codecs import open  # To use a consistent encoding
from os import path, walk
import glob

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


def files_in_subdirs(start_dir, pattern):
    files = []
    for dir, _, _ in walk(start_dir):
        files.extend(glob.glob(path.join(dir, pattern)))
    return files


# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

add_install_requirement = []
if platform.system() == "Windows":
    if "bash" in environ.get("SHELL", ""):
        print("MSeg support for Windows is experimental; don't use this in production environments.")
        add_install_requirement += ["wget"]  # pip will install wget for Windows if necessary
    else:
        print("MSeg currently does not support Windows, please use Linux/Mac OS; experimental support for gitbash")
        sys.exit(1)

setup(
    name="mseg",
    version="1.0.0",
    description="",
    long_description=long_description,
    url="https://github.com/mseg-dataset",
    author="Intel Labs",
    author_email="johnlambert@gatech.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision robotics dataset-tools",
    packages=find_packages(exclude=["tests"]),
    package_data={"mseg": ["dataset_lists/**/*"]},
    scripts=files_in_subdirs(path.join(here, "download_scripts"), "mseg_download_*.sh"),
    include_package_data=True,
    python_requires=">= 3.7",
    install_requires=[
        "imageio",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",  # rather than "pillow"
        "opencv-python>=4.1.0.25",
        "scipy",
        "torch",
        "tqdm",
    ]
    + add_install_requirement,
)
