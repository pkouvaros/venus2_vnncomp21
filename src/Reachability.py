# ************
# File: Reachability.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for reachability specification.
# ************


from src.Specification import Specification
from src.Formula import VarConstConstraint, NAryConjFormula

class Reachability(Specification):

    def __init__(self, input_bounds, output):
        """
        Arguments:

            input_bounds: pair of arrays  of input lower and upper bounds.

            output: output for which to determine reachability.
        """

        self.input_bounds = input_bounds
        self.output = output
        self.output_formula = self.get_output_formula()

    def get_output_fomrula(self):
        """
        Constructs a logic formula encoding reachability.
        """
        output_dim = len(self.output)
        atoms = [VarConstConstraint(StateCoordinate(i), Formula.EQ, self.output[i] 
                                    for i in range(output_dim)]

        return NAryConjFormula(atoms)


    def to_string(self):
        """
        Returns:

            str describing the specification 
        """
        string = 'Property: Reachability. Input: ' + \
            self.input_layer.name + \
            '. Reachable output: ' + \
            str(self.output) +  \
            '.'
        
        return string
