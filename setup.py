# -*- coding: utf-8 -*-
# oiffile/setup.py

"""Oiffile package setuptools script."""

import sys
import re

from setuptools import setup

with open('oiffile/oiffile.py') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]
description = re.search(r'"""(.*)\.[\r\n?|\n]', code).groups()[0]
readme = re.search(r'[\r\n?|\n]{2}"""(.*)"""[\r\n?|\n]{2}from', code,
                   re.MULTILINE | re.DOTALL).groups()[0]
license = re.search(r'(# Copyright.*?[\r\n?|\n])[\r\n?|\n]+""', code,
                    re.MULTILINE | re.DOTALL).groups()[0]

readme = '\n'.join([description, '=' * len(description)]
                   + readme.splitlines()[1:])
license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)

setup(
    name='oiffile',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    url='https://www.lfd.uci.edu/~gohlke/',
    license='BSD',
    packages=['oiffile'],
    python_requires='>=2.7',
    install_requires=['numpy>=1.11.3', 'tifffile>=2019.1.1'],
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
)
