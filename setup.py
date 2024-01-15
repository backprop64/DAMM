from setuptools import setup, find_packages

with open('requirements-colab.txt') as f:
    required = f.read().splitlines()

setup(
    name='DAMM',
    version='0.1',
    packages=find_packages(),
    install_requires=required,
    author='Gaurav Kaul',
    author_email='kaulg@umich.com',
)
