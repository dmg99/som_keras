import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SOMkeras",
    version="0.0.1",
    author="Daniel Molinuevo",
    author_email="daniel.molinuevo@starlab.es",
    description="Self-Organizing map keras model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://git.starlab.es/andres.rojas/py_stareeglab.git",
    packages=setuptools.find_packages(),
    classifiers=("Programming Language :: Python :: 3.6",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",)
)