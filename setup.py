from distutils.core import setup
from pyexpat import model

import setuptools

setup(
    name="InteractiveOrNotebook",
    version="0.1.0",
    author="Vladimir Fux",
    author_email="nonvisual.wrk@gmail.com",
    packages=setuptools.find_packages(exclude=("tests", "docs")),
    license="LICENSE.txt",
    description="Dependencies for interactive OR notebook tutorial",
    long_description=open("README.md").read(),
    python_requires=">=3.6",
)
