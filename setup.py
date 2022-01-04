#!/usr/bin/python3

from setuptools import setup, find_packages
import os


datadir = os.path.join('data')
datafiles = [(d, [os.path.join(d, f) for f in files])
             for d, folders, files in os.walk(datadir)]

setup(name='holoprot',
      version='0.1',
      description='Multi-Scale Representation Learning on Proteins',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'torch'],
      zip_safe=False,
      data_files=datafiles,
)
