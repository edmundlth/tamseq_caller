#!/usr/bin/env python

from distutils.core import setup

LONG_DESCRIPTION = """Genotype and somatic mutation caller for TAmSeq data."""


setup(
    name='',
    version='0.1.0.0',
    author='Edmund Lau',
    author_email='edmund.lau@unimelb.edu.au',
    packages=[''],
    package_dir={'base_counter': 'base_counter'},
    entry_points={
        'console_scripts': ['base_counter = base_counter.base_counter:main',
        'variant_detection = base_counter.variant_detection:main']
    },
    url='https://github.com/bjpop/base_counter',
    license='LICENSE',
    description='',
    long_description=LONG_DESCRIPTION,
    install_requires=["numpy", "scipy", "pandas"],
)
