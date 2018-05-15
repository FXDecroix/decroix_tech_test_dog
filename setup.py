#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from distutils.core import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='decroix_tech_test_dog',
      version='0.1',
      description='Multi resolution dog classification using Resnet50 in Keras',
      long_description=long_description,
      author='F.X. Decroix',
      author_email='francoisxavier.decroix@gmail.com',
      url='https://github.com/FXDecroix/decroix_tech_test_dog',
      py_modules=['dog_classif'],
      )

