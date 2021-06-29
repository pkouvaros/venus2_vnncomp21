# ************
# File: AdversarialRobustness.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for local adversarial robustness specification.
# ************

from src.Layers import Input
from src.Specification import Specification
from src.Formula import Formula, StateCoordinate, VarVarConstraint, NAryConjFormula

class AdversarialRobustness(Specification):

    def __init__(self, 
                 input, 
                 label, 
                 num_classes, 
                 radius,
                 mean=0, 
                 std=1, 
                 min_value=0, 
                 max_value=1,
                 input_name='input'):
        """
        Arguments:

            input: network input
            
            label: label of the input
            
            output_dim: output dimension of the network
            
            radius: radius of the perturbation infinity-norm ball
        """
        self.label = label 
        self.num_classes = num_classes
        self.radius = radius
        input_layer = Input(input-radius, input+radius, input_name)
        output_formula = self.get_output_formula()
        super().__init__(input_layer, output_formula)
        self.clip(min_value, max_value)
        self.normalise(mean, std)

    def get_output_formula(self, labels=None):
        """
        Constructs a logic formula encoding local adversarial robustness.

        Arguments:
            
            labels: list of output dimensions to include in the formula.

        Returns:

            NAryConjFormula encoding local adversarial robustness.
        """
        if labels is None: labels = set(range(self.num_classes)) - set([self.label])
        coordinates = [StateCoordinate(i) for i in labels]
        label_coord = StateCoordinate(self.label)
        atoms = [VarVarConstraint(i, Formula.Sense.LT, label_coord) for i in coordinates]

        return NAryConjFormula(atoms)


    def to_string(self):
        """
        Returns:

            str describing the specification 
        """
        string = 'adversarial local robustness for input ' + \
            self.input_layer.name + \
            ' and radius ' + \
            str(self.radius) +  \
            '.'
        
        return string

    def update_adv_labels(self, labels):
        """
        Updates the output formula to include only a given set of labels.

        Arguments:

            labels: list of int.

        Returns:

            None
        """

        self.output_formula = self.get_output_formula(labels)
