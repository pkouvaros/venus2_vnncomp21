# ************
# File: DIRReader.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Class for loading network inputs from directory, where each
# input is a file within the directory. The files ought to have the same prefix
# followed by the number of inputs.
# ************

import os 
import glob
import re
import numpy as np

class DIRReader:

    def __init__(self, path):
        """
        Arguments:
            path: str of path to input directory
        """
        self.path = path
    
    def read_inputs(self):
        """
        Loads the input from the specified directory.

        Returns:

            A dictionary of the inputs where keys are the names of the input
            files and the values are the inputs.
        """  
        #inputs 
        inputs = {}
        # inputs' filenames
        files = os.listdir(self.path)
        # sort filenames by their numeric content
        files = sorted(files,key=lambda i: int(re.sub('[^0-9]','',i)))
        # read inputs
        for fl in files:
            with open(os.path.join(self.path,fl)) as f:
                data = f.readline().rstrip(os.linesep).rstrip(',').split(',')
                input = np.array([np.float64(i) for i in data])
                inputs[fl] =  input

        return inputs




