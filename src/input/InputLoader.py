# ************
# File: InputLoader.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Class for loading the network inputs.
# ************

from src.input.DIRReader import DIRReader
from src.input.CSVReader import CSVReader
import os

class InputLoader:

    def __init__(self, path, normalise=True, min_value=0, max_value=255, shape=None):
        """
        Arguments:

            path: str of path of the CSV file

            normalise: bool of whether to normalise the inputs
            
            min_value: minimum possible value of the input data
            
            max_value: maximum possible value of the input data
        """
        self.path = path
        self.min_value = min_value
        self.max_value = max_value
        self.normalise = normalise
        self.shape = shape

    def load(self):
        """
        Loads the input.

        Returns:
            A dictionary of the inputs where keys are the names of the inputs
            and the values are the inputs.
        """
        if os.path.isdir(self.path):
            dir_reader = DIRReader(self.path)
            inputs = dir_reader.read_inputs()
        else:
            csv_reader = CSVReader(self.path)
            inputs = csv_reader.read_inputs() 

        if self.normalise == True: inputs = self.normalise_inputs(inputs)
        if not self.shape is None: self.reshape_inputs(inputs)

        return inputs

    def normalise_inputs(self, inputs):
        """
        Normalises the inputs to [min_value, max_value]

        Arguments:
            
            inputs: dictionary of the inputs

        Returns:
                
            dictionary of normalised inputs
        """

        for inp in inputs:
            inputs[inp] = (inputs[inp] - self.min_value) / (self.max_value - self.min_value)

        return inputs


    def reshape_inputs(self, inputs):
        """
        Reshapes the inputs to the given shape

        Arguments:
            
            inputs: dictionary of the inputs

        Returns:
                
            dictionary of reshaped inputs
        """

        for inp in inputs:
            inputs[inp] = inputs[inp].reshape(self.shape) 

        return inputs
