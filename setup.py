# -*- coding: utf-8 -*-
"""
Setup file for ssms
"""
import os
from setuptools import setup, find_packages
import pkg_resources

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='ssms/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'


try:
    from pypandoc import convert_file

    def read_md(f): return convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")

    def read_md(f): return open(f, 'r').read()


setup(name='ssms',
      version=version,
      description='Support structure mass surrogate for offshore wind turbines',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate',
      project_urls={
          'Documentation': 'https://topfarm.pages.windenergy.dtu.dk/support-structure-mass-surrogate/',
          'Source': 'https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate',
          'Tracker': 'https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate/-/issues',
      },
      author='DTU Wind Energy',
      author_email='mikf@dtu.dk',
      license='MIT',
      packages=find_packages(),
      package_data={
          'ssms': [
            'data/tower_mass_results.dat',
            'data/tower_mass_results_extended_depth_results.dat',
            'models/QLS/*.pickle',
            ],},
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scikit-learn',
          'openturns',
          ],
      extras_require={
        'test': [
            'pytest',],
        'docs': [
            'pypandoc',
            'sphinx',
            'nbsphinx',
            'nbconvert',
            'sphinx_rtd_theme',],},
      include_package_data=True,
      zip_safe=True)
