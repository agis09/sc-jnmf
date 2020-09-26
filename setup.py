import os
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='sc-jnmf',
    version='1.0.0',
    description='Joint-NMF for single cell analysis',
    long_description=readme,
    author='Mikio Shiga',
    url='https://github.com/agis09/sc-jnmf',
    install_requires=read_requirements(),
    license=license,
    # packages=find_packages(exclude=('tests', 'docs'))
    packages=['sc_jnmf'],
)
