#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from os import path
from keras.models import model_from_yaml
from ..set_dataset import load_dataset
from ..set_model import setup_model

class TestMethods(unittest.TestCase):

    def test_load_dataset(self):
        data, labels = load_dataset(path.join(path.abspath(path.dirname(__file__)),'resources/sample_dataset'),256)
        self.assertEqual(data.shape, (4,256,256,3))
        self.assertEqual(len(labels), 4)

    def test_setup_model(self):        
        model = setup_model(256)
        with open(path.join(path.abspath(path.dirname(__file__)),'resources/model_ref.yaml'),'r') as f:
            model_ref = model_from_yaml(f)
        self.assertEqual(model.get_config(),model_ref.get_config())

if __name__ == '__main__':
    unittest.main()

