# ************
# File: Specification.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for generic specifications.
# ************

from gurobipy import *
from src.Formula import *


class Specification:

    def __init__(self, input_layer, output_formula, name=None):
        """
        Arguments:

            input_layer: input_layer

            output_formula: formula encoding output constraints.
        """
        self.input_layer = input_layer
        self.output_formula = output_formula
        self.name = name

    def get_output_constrs(self, gmodel, output_vars):
        """
        Encodes the output constraints of the spec into MILP

        Arguments:

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network

        Returns:

            list of gurobi constraints encoding the output formula
        """
        if self.output_formula is None:
            return []

        negated_output_formula = NegationFormula(self.output_formula).to_NNF()  
        return self.get_constrs(negated_output_formula, gmodel, output_vars)

    def get_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a given formula into MILP

        Arguments:

            formula: formula to encode

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, Formula), f'Got {type(formula)} instead of Formula'

        if isinstance(formula, Constraint):
            return [self.get_atomic_constr(formula, output_vars)]
        elif isinstance(formula, ConjFormula):
            return self.get_conj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, NAryConjFormula):
            return self.get_nary_conj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, DisjFormula):
            return self.get_disj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, NAryDisjFormula):
            return self.get_nary_disj_formula_constrs(formula, gmodel, output_vars)

    def get_atomic_constr(self, constraint, output_vars):
        """
        Encodes an atomic constraint into MILP

        Arguments:

            constraint: constraint to encode

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given constraint
        """

        assert isinstance(constraint, Constraint), f'Got {type(constraint)} instead of Constraint'

        sense = constraint.sense
        if isinstance(constraint, VarVarConstraint):
            op1 = output_vars[constraint.op1.i]
            op2 = output_vars[constraint.op2.i]
        elif isinstance(constraint, VarConstConstraint):
            op1 = output_vars[constraint.op1.i]
            op2 = constraint.op2
        elif isinstance(constraint, LinExprConstraint):
            op1 = 0
            for i, c in constraint.op1.coord_coeff_map.items():
                op1 += c * vars[i]
            op2 = constraint.op2
        else:
            raise Exception("Unexpected type of atomic constraint", constraint)
        
        if sense == Formula.Sense.GE:
            return op1 >= op2
        elif sense == Formula.Sense.LE:
            return op1 <= op2
        elif sense == Formula.Sense.EQ:
            return op1 == op2
        else:
            raise Exception("Unexpected type of sense", sense)


    def get_conj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a conjunctive formula into MILP

        Arguments:

            formula: conjunctive formula to encode

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, ConjFormula), f'Got {type(formula)} instead of ConjFormula'

        return self.get_constrs(formula.left, gmodel, output_vars) + self.get_constrs(formula.right, gmodel, output_vars)

    def get_nary_conj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes an nary conjunctive formula into MILP

        Arguments:

            formula: nary conjunctive formula to encode

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, NAryConjFormula), f'Got {type(formula)} instead of NAryConjFormula'

        constrs = []
        for subformula in formula.clauses:
            constrs += self.get_constrs(subformula, gmodel, output_vars)

        return constrs

    def get_disj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a disjunctive formula into MILP

        Arguments:

            formula: disjunctive formula to encode

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, DisjFormula), f'Got {type(formula)} instead of DisjFormula'

        split_var = gmodel.addVar(vtype=GRB.BINARY)
        clause_vars = [gmodel.addVars(len(output_vars), lb=-GRB.INFINITY), 
                       gmodel.addVars(len(output_vars), lb=-GRB.INFINITY)]
        constr_sets = [self.get_constrs(formula.left, gmodel, clause_vars[0]), 
                       self.get_constrs(formula.right, gmodel, clause_vars[1])]
        constrs = []
        for i in [0, 1]:
            for j in range(len(output_vars)):
                constrs.append((split_var == i) >> (output_vars[j] == clause_vars[i][j]))
            for disj_constr in constr_sets[i]:
                constrs.append((split_var == i) >> disj_constr)

        return constrs

    def get_nary_disj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes an nary disjunctive formula into MILP

        Arguments:

            formula: nary disjunctive formula to encode

            gmodel: gurobi model

            output_vars: list of gurobi variables of the output of the network
        
        Returns:

            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, NAryDisjFormula), f'Got {type(formula)} instead of NAryDisjFormula'

        clauses = formula.clauses
        split_vars = gmodel.addVars(len(clauses), vtype=GRB.BINARY)
        clause_vars = [gmodel.addVars(len(output_vars), lb=-GRB.INFINITY) 
                       for _ in range(len(clauses))]
        constr_sets = []
        constrs = []
        for i in range(len(clauses)):
            constr_sets.append(self.get_constrs(clauses[i], gmodel, clause_vars[i]))
            for j in range(len(output_vars)):
                constrs.append((split_vars[i] == 1) >> (output_vars[j] == clause_vars[i][j]))
            for disj_constr in constr_sets[i]:
                constrs.append((split_vars[i] == 1) >> disj_constr)
        # exactly one variable must be true
        constrs.append(quicksum(split_vars) == 1)
            
        return constrs

    def copy(self, input_layer=None):
        """
        Returns a copy of the specificaton

        Arguments:

            input_layer: input layer to optionally update in the copy

        Returns:

            Specification
        """
        il = input_layer if not input_layer is None else self.input_layer.copy()

        return Specification(il, self.output_formula, self.name)

    def normalise(self, mean, std):
        """
        Normalises the input bounds

        Arguments:

            mean: normalisation mean

            std: normalisation standard deviation

        Returns:

            None
        """
        self.input_layer.post_bounds.normalise(mean, std)

    def clip(self, min_value, max_value):
        """
        Clips the input bounds

        Arguments:

            min_value: valid lower bound

            max_value: valid upper bound

        Returns:
            
            None
        """
        self.input_layer.post_bounds.clip(min_value, max_value)

    def is_satisfied(self, lower_bounds, upper_bounds):
        return self._is_satisfied(self.output_formula, lower_bounds, upper_bounds)

    def _is_satisfied(self, formula, lower_bounds, upper_bounds):
        if formula is None:
            return True
        elif isinstance(formula, Constraint):
            sense = formula.sense
            if sense == Formula.Sense.LT:
                if isinstance(formula, VarVarConstraint):
                    return upper_bounds[formula.op1.i] < lower_bounds[formula.op2.i]
                if isinstance(formula, VarConstConstraint):
                    return upper_bounds[formula.op1.i] < formula.op2
            elif sense == Formula.Sense.GT:
                if isinstance(formula, VarVarConstraint):
                    return lower_bounds[formula.op1.i] > upper_bounds[formula.op2.i]
                if isinstance(formula, VarConstConstraint):
                    return lower_bounds[formula.op1.i] > formula.op2
            # else:
                # raise Exception('Unexpected sense', formula.sense)
        elif isinstance(formula, ConjFormula):
            return self._is_satisfied(formula.left, lower_bounds, upper_bounds) and \
                   self._is_satisfied(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryConjFormula):
            for clause in formula.clauses:
                if not self._is_satisfied(clause, lower_bounds, upper_bounds):
                    return False
            return True
        elif isinstance(formula, DisjFormula):
            return self._is_satisfied(formula.left, lower_bounds, upper_bounds) or \
                   self._is_satisfied(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryDisjFormula):
            for clause in formula.clauses:
                if self._is_satisfied(clause, lower_bounds, upper_bounds):
                    return True
            return False
        else:
            raise Exception("Unexpected type of formula", type(formula))

    def to_string(self):
        """
        Returns:

            str describing the specification 
        """
        return self.name
