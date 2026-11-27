import os
from glob import glob
from setuptools import find_packages, setup

package_name = "robosignformer"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]

model_files = [
    f for f in glob("models/*")
    if os.path.isfile(f) and not os.path.basename(f).startswith("robosignformer_seed")
]
if model_files:
    data_files.append(("share/" + package_name + "/models", model_files))

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Abdullah Feyyaz Ekici",
    maintainer_email="ekici.abdullahfeyyaz@gmail.com",
    description="RoboSignFormer: Lightweight 4-Stream Skeletal Conv-Transformer for Turkish Sign Language Recognition",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sign_node = robosignformer.sign_node:main",
        ],
    },
)
