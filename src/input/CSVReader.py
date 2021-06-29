# ************
# File: CSVReader.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Class for loading network inputs from a CSV file.
# ************

import os
import csv
import numpy as np

class CSVReader:

    def __init__(self, path):
        """
        Arguments:

            path: str of path of the CSV file
        """
        self.path = path
    
    def read_inputs(self):
        """
        Loads the input from the specified CSV file.

        Returns:

            A dictionary of the inputs where keys are the names of the inputs
            and the values are the inputs.
        """
        #inputs 
        inputs = {}
        # read inputs
        c = 1
        with open(self.path) as f:
            lines = f.readlines()
        for line in lines:
            data = line.rstrip(os.linesep).rstrip(',').split(',')
            input = np.array([np.float64(i) for i in data])
            inputs['image'+str(c)] =  input
            c += 1



        return inputs







