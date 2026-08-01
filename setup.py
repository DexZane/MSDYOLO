from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="msdyolo",
    version="1.0.0",
    author="MSDYOLO Team",
    description="Multi-Scale Deformable YOLO for Oriented Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DexZane/MSDYOLO",
    packages=find_packages(include=["msdyolo", "msdyolo.*"]),
    include_package_data=True,
    package_data={
        "msdyolo": [
            "data/*.yaml",
            "data/hyps/obb/*.yaml",
            "data/examples/images/*",
            "data/examples/labelTxt/*.txt",
            "data/examples/*.txt",
            "utils/nms_rotated/setup.py",
            "utils/nms_rotated/src/*.cpp",
            "utils/nms_rotated/src/*.cu",
            "utils/nms_rotated/src/*.h",
        ],
    },
    license="GPL-3.0-or-later",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "msdyolo-train=msdyolo.train:main",
            "msdyolo-val=msdyolo.val:cli",
            "msdyolo-detect=msdyolo.detect:cli",
            "msdyolo-export=msdyolo.export:cli",
        ],
    },
)
