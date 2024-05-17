from setuptools import setup, find_packages

setup(
    name="DAMM",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "filterpy",
    ],
    author="Gaurav Kaul",
    author_email="kaulg@umich.com",
)
