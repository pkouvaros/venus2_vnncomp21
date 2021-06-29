import numpy as np
from src.Formula import *
from src.utils.SplitStrategy import SplitStrategy
from src.utils.OSIPMode import OSIPMode
from src.MILPEncoder import MILPEncoder
from src.AdversarialRobustness import AdversarialRobustness
from src.SIP import SIP
from src.Parameters import SIP as SIP_PARAMS
from gurobipy import *

class VerificationProblem(object):

    prob_count = 0

    def __init__(self, nn, spec, depth, logfile):
        VerificationProblem.prob_count += 1
        self.id = VerificationProblem.prob_count
        self.nn = nn
        self.spec = spec
        self.depth = depth
        self.logfile = logfile
        self.id 
        self._sip_bounds_computed = False
        self._osip_bounds_computed = False


    def bound_analysis(self, sip_params, delta_vals=None):
        sip = self.set_bounds(sip_params, delta_vals)
        if not sip is None:
            self.stability_ratio = self.nn.get_stability_ratio()
            self.output_range = self.nn.get_output_range()
            if isinstance(self.spec, AdversarialRobustness):
                adv_labels = sip.get_adv_labels(self.spec.label)
                self.spec.update_adv_labels(adv_labels)
            else:
                self.spec.output_formula = sip.simplify_formula(self.spec.output_formula)
            return True
        else:
            return False

    def set_bounds(self, sip_params, delta_vals=None):
        # check if bounds are already computed
        if delta_vals is None:
            if sip_params.is_osip_enabled():
               if self._osip_bounds_computed: return None
            else:
                if self._sip_bounds_computed: 
                    return None
        # compute bounds
        sip = SIP([self.spec.input_layer] + self.nn.layers, sip_params, self.logfile)
        sip.set_bounds(delta_vals)
        # flag the computation
        if delta_vals is None:
            if sip_params.is_osip_enabled():
                self._osip_bounds_computed = True
            else:
                self._sip_bounds_computed = True

        return sip

    def encode_into_milp(self, intra=True, inter=True):
        gmodel = Model()
        # encode into MILP
        encode_into_milp(self.nn,
                         self.spec,
                         gmodel,
                         intra=params.INTRA_DEP_CONSTRS, 
                         inter=params.INTER_DEP_CONSTRS)

        return gmodel


    def score(self, initial_fixed_ratio, depth_power):
        return (self.stability_ratio - initial_fixed_ratio) 
        return (self.stability_ratio - initial_fixed_ratio) / ((self.depth+1) ** (1/depth_power))


    def worth_split(self, subprobs, initial_fixed_ratio, depth_power):
        pscore0 = self.score(initial_fixed_ratio, depth_power)
        pscore1 = subprobs[0].score(initial_fixed_ratio, depth_power)
        pscore2 = subprobs[1].score(initial_fixed_ratio, depth_power)

        _max = max(pscore1, pscore2)
        _min = min(pscore1, pscore2) 

        if pscore0 >= _max:
            return False
        elif _min > pscore0:
            return True
        elif  (pscore1 + pscore2)/2 > pscore0:
            return True
        else:
            return False
 
    def check_bound_tightness(self, subprobs):
        out0 = self.nn.layers[-1]
        for sp in subprobs:
            if sp.output_range > self.output_range:
                return False
        return True

        for sp in subprobs:
            out1 = sp.nn.layers[-1]
            for i in out0.outputs():
                b0 = out0.post_bounds.lower[i]
                b1 = out1.post_bounds.lower[i]
                if b1 < b0:
                    return False
                b0 = out0.post_bounds.upper[i]
                b1 = out1.post_bounds.upper[i]
                if b1 > b0:
                    return False
        return True

    def lp_analysis(self, sip_params):
        ver_report = VerificationReport(self.logfile)

        if self.spec.output_formula is None:
            return True
        self.bound_analysis(sip_params)
        lower_bounds = self.nn.layers[-1].post_bounds.lower
        upper_bounds = self.nn.layers[-1].post_bounds.upper
        if self.satisfies_spec(self.spec.output_formula, lower_bounds, upper_bounds):  
            ver_report.result = SolveResult.SATISFIED 



    def satisfies_spec(self):
        if self.spec.output_formula is None:
            return True
        if not self._sip_bounds_computed and not self.osip_bounds_computed:
            raise Exception('Bounds not computed')
        lower_bounds = self.nn.layers[-1].post_bounds.lower
        upper_bounds = self.nn.layers[-1].post_bounds.upper
        return self.spec.is_satisfied(lower_bounds, upper_bounds)
        # return self._satisfies_spec(self.spec.output_formula, lower_bounds, upper_bounds)

    def _satisfies_spec(self, formula, lower_bounds, upper_bounds):
        if  formula is None:
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
        elif isinstance(formula, ConjFormula):
            return self._satisfies_spec(formula.left, lower_bounds, upper_bounds) and \
                   self._satisfies_spec(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryConjFormula):
            for clause in formula.clauses:
                if not self._satisfies_spec(clause, lower_bounds, upper_bounds):
                    return False
            return True
        elif isinstance(formula, DisjFormula):
            return self._satisfies_spec(formula.left, lower_bounds, upper_bounds) or \
                   self._satisfies_spec(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryDisjFormula):
            for clause in formula.clauses:
                if self._satisfies_spec(clause, lower_bounds, upper_bounds):
                    return True
            return False
        else:
            raise Exception("Unexpected type of formula", type(formula))



    def get_var_indices(self, layer, var_type):
        """
        Returns the indices of the MILP variables associated with a given
        layer.

        Arguments:
                
            layer: int of the index of the layer  for which to retrieve the
            indices of the MILP variables

            var_type: str: either 'out' for the output variables or 'delta' for
            the binary variables.

        Returns:
        
            pair of ints indicating the start and end positions of the indices
        """
        layers = [self.spec.input_layer] + self.nn.layers
        start = 0
        end = 0
        for i in range(layer):
            start += layers[i].out_vars.size + layers[i].delta_vars.size
        if var_type == 'out':
            end = start + layers[layer].out_vars.size
        elif var_type == 'delta':
            start += layers[layer].out_vars.size
            end = start + layers[layer].delta_vars.size

        return start, end


    def to_string(self):
        return self.nn.model_path  + ' against ' + self.spec.to_string()

