# ************
# File: SIP.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Symbolic Interval Propagaton.
# ************


from src.Layers import Input, FullyConnected, Conv2D, MaxPooling, GlobalAveragePooling
from src.Equations import Equations
from src.MemEquations import MemEquations
from src.utils.ReluApproximation import ReluApproximation
from src.utils.ReluState import ReluState
from src.utils.Activations import Activations
# from src.utils.Logger import get_logger
from src.OSIP import OSIP
from src.utils.OSIPMode import OSIPMode
from src.Formula import Formula, TrueFormula, VarVarConstraint, DisjFormula, NAryDisjFormula, ConjFormula, NAryConjFormula
from timeit import default_timer as timer
import numpy as np
import math

class SIP():

    # logger = None

    def __init__(self, 
                 layers, 
                 params,
                 logfile):
        """
        Arguments:

            layers: a list [input,layer_1,....,layer_k] of the networks layers
            including an input layer defining the input.

            mem_efficient: If True the computations are carried out with less
            memory requirements.

            logfile: str of path to logfile.
        """
        self.layers = layers
        self.eqs = []
        self.mem_eqs = []
        self.bounds = []
        self.logfile = logfile
        self.params = params
        # if SIP.logger is None:
            # SIP.logger = get_logger(__name__, logfile)

    def set_bounds(self, relu_states=None):
        """
        Sets pre-activation and activation bounds.

        Arguments:

            relu_states: a list of floats, one for each node. 0: inactive,
            1:active, anything else: unstable; use this for runtime computation of
            bounds.

        Returns:
            
                None
        """
        start = timer()
        for l in self.layers[1:]:
            pre_eq, pre_mem_eq, pre_b = self.pre_bounds(l)
            l.pre_bounds.lower = pre_b[0]
            l.pre_bounds.upper = pre_b[1]
            post_eq, post_mem_eq, post_b = self.post_bounds(l, pre_eq, pre_mem_eq, relu_states)
            self.bounds.append(post_b)
            if l.activation == Activations.relu:
                l.post_bounds.lower = np.clip(post_b[0].reshape(l.output_shape), 0, math.inf)
                l.post_bounds.upper = np.clip(post_b[1].reshape(l.output_shape), 0, math.inf)
            else:
                l.post_bounds.lower = post_b[0]
                l.post_bounds.upper = post_b[1]

            self.eqs.append(post_eq)
            self.mem_eqs.append(post_mem_eq)

        # print(self.layers[-1].post_bounds.lower)
        # import sys
        # sys.exit()
        # SIP.logger.info('Bounds computed, time: {:.3f}, '.format(timer()-start))

    def pre_bounds(self, layer):
        ib = (self.layers[0].pre_bounds.lower.flatten(),
              self.layers[0].pre_bounds.upper.flatten())
        if isinstance(layer, Conv2D):
            eq, mem_eq, b = self.conv_layer(layer, self.eqs, self.mem_eqs, ib)
        elif isinstance(layer, FullyConnected):
            eq, mem_eq, b = self.fully_connected_layer(layer, self.eqs, ib)
        elif isinstance(layer, MaxPooling):
            eq, mem_eq, b = self.max_pooling(l, self.eqs, self.bounds[-1], ib)
        elif isinstance(l,GlobalAveragePooling):
            lb = self.layers[layer.depth-1].post_bounds.lower
            ub = self.layers[layer.depth-1].post_bounds.upper
            eq, mem_eq, b = self.global_average_pool(l, self.eqs, (lb,ub))

        if layer.activation == Activations.relu:
            b = self.branching_bounds(layer.state, b[0], b[1])

        return eq, mem_eq, b

    def post_bounds(self, layer, eq, mem_eq, relu_states):
        lb = layer.pre_bounds.lower
        ub = layer.pre_bounds.upper
        if layer.activation == Activations.linear:
            post_eq = [eq, eq]
            mem_post_eq = [mem_eq, mem_eq]
            post_b = [lb, ub]
        elif layer.activation == Activations.relu:
            if self.osip_eligibility(layer) == True:
                approx = ReluApproximation.IDENTITY
                mem_post_eq = [mem_eq.lower_relu_relax(lb.flatten(), 
                                                       ub.flatten(), 
                                                       self.params.RELU_APPROXIMATION),
                               mem_eq.upper_relu_relax(lb.flatten(), 
                                                       ub.flatten())]
            else:
                approx = self.params.RELU_APPROXIMATION
                mem_post_eq = [None, None]
            if relu_states is None:
                post_eq, post_b = self.relu(eq,
                                            lb.flatten(), 
                                            ub.flatten(),
                                            approx,
                                            self.params.RELU_APPROXIMATION)
            else:
                post_eq_r, post_b_r = self.relu(eq, 
                                                lb.flatten(), 
                                                ub.flatten(),
                                                approx,
                                                self.params.RELU_APPROXIMATION)
                post_eq, post_b = self.runtime_bounds(post_eq_r[0],
                                                      post_eq_r[1],
                                                      eq,
                                                      post_b_r[0],
                                                      post_b_r[1],
                                                      lb,
                                                      ub,
                                                      relu_states[layer.depth-1])

        return post_eq, mem_post_eq, post_b 

    def osip_eligibility(self, layer):
        if layer.depth == len(self.layers):
            return False
        if self.params.OSIP_CONV != OSIPMode.ON:
            if isinstance(self.layers[layer.depth+1], Conv2D) \
            or isinstance(layer, Conv2D):
                return False
        if self.params.OSIP_FC != OSIPMode.ON:
            if isinstance(self.layers[layer.depth+1], FullyConnected) \
            or isinstance(layer, FullyConnected):
                return False
        
        return True

         
    def conv_layer(self, layer, eqs, mem_eqs, input_bounds): 
        """
        Computes the pre-activation bound equations of a convolutional layer.

        Arguments:

            layer: a convolutional layer.
            
            eqs: list of bound equations for the outputs of the preceding
            layers.

        Returns:

            a triple of: (i) the linear equation for the bounds of the
            pre-activation of the layer; (ii) the concrete lower bounds of the
            equation; (iii) the concrete upper bounds of the equation.
        """
        M,N,O,K = layer.kernels.shape
        size_per_kernel = int(layer.output_size / K)
        conv = np.arange(layer.input_size).reshape(layer.input_shape)
        conv = Conv2D.pad(conv, layer.padding, values=(layer.input_size, layer.input_size))
        conv = Conv2D.im2col(conv, (M, N), layer.strides)
        conv = np.repeat(conv, K, axis=1)
        weights = np.array([layer.kernels[:,:,:,i].flatten()
                            for j in range(size_per_kernel)
                            for i in range(K)],
                           dtype='float64')
        weights = weights.T
        coeffs = np.zeros((layer.output_size, layer.input_size+1), dtype='float64')
        coeffs[range(layer.output_size), conv] = weights
        coeffs = coeffs[:,range(layer.input_size)]
        const = np.array([layer.bias for i in range(size_per_kernel)], dtype='float64').flatten()
        eq = Equations(coeffs, const, self.logfile)
    
        if self.params.OSIP_CONV == OSIPMode.ON:
            d =[{conv[i,eq]: weights[i,eq] for i in range(M*N*O)
                 if not conv[i, eq] == layer.input_size}
                for eq in range(layer.output_size)]
            mem_eq = MemEquations(d, const, self.logfile)
            osip = OSIP(layer,
                        mem_eq, 
                        self.layers[0:layer.depth], 
                        eqs,
                        self.params.OSIP_CONV_NODES,
                        self.params.OSIP_TIMELIMIT,
                        self.params.RELU_APPROXIMATION,
                        self.logfile,
                        mem_eqs)
            osip.optimise()
        else:
            mem_eq = None

        lbounds = self.back_substitution(eq, eqs, 'lower', input_bounds).reshape(layer.output_shape)
        ubounds = self.back_substitution(eq, eqs, 'upper', input_bounds).reshape(layer.output_shape)

        return eq, mem_eq, [lbounds, ubounds]


    def fully_connected_layer(self, layer, eqs, input_bounds):
        """
        Computes the pre-activation bound equations of a fully-connected layer.

        Arguments:

            layer: a fully-connected layer.
            
            eqs: list of bound equations for the outputs of the preceding
            layers.

        Returns:

            a triple of: (i) the linear equation for the bounds of the
            pre-activation of the layer; (ii) the concrete lower bounds of the
            equation; (iii) the concrete upper bounds of the equation.
        """
        
        eq = Equations(layer.weights, layer.bias, self.logfile)
        if self.params.OSIP_FC == OSIPMode.ON:
            osip = OSIP(layer,
                        eq, 
                        self.layers[0:layer.depth], 
                        eqs,
                        self.params.OSIP_FC_NODES,
                        self.params.OSIP_TIMELIMIT,
                        self.params.RELU_APPROXIMATION,
                        self.logfile)
            osip.optimise()

        lbounds = self.back_substitution(eq, eqs, 'lower', input_bounds)
        ubounds = self.back_substitution(eq, eqs, 'upper', input_bounds) 

        return eq, eq, [lbounds, ubounds]


    def back_substitution(self, eq, peqs, bound, input_bounds):
        """
        Computes the concrete lower and upper bounds of a given linear
        equation.

        Arguments:
            
            eq: linear equation whose bounds are to be computed.
            
            peqs: list of linear equations of the outputs of the preceding
            layers.
            
            bound: bound to be computed: either 'lower' or 'upper'.
            
            input_bounds: the lower and upper bounds of the input layer.

        Returns:

            concrete lower and upper bounds of eq.
        """

        assert bound in ['lower', 'upper']
        q = eq.copy()
        for i in range(len(peqs)-1, -1, -1):
            q = q.dot(bound, peqs[i][0], peqs[i][1])
        if bound=='lower':
            return q.min_values(input_bounds[0], input_bounds[1])
        else:
            return q.max_values(input_bounds[0], input_bounds[1])



    def relu(self, eq, lbounds, ubounds, eq_approx, bound_approx):

        """
        Applies the relu function to the given pre-activation bound equation
        and bounds.

        Arguments:
            
            eq: pre-activation bound equation.
            
            lbounds: the concrete lower bounds of eq.
            
            ubounds: the concrete upper bounds of eq.

        Returns:

            a four-tuple of the lower and upper bound equations and lower and
            upper concrete bounds resulting from applying relu to eq.
        """
        eqlow = eq.lower_relu_relax(lower=lbounds, upper=ubounds, approx=eq_approx)
        equp = eq.upper_relu_relax(lower=lbounds, upper=ubounds)
        l = lbounds.copy()
        u = ubounds.copy()
        for i in range(eqlow.size):
            if l[i] >= 0:
                pass
            elif u[i] > 0: 
                if bound_approx == ReluApproximation.VENUS_HEURISTIC:
                    l[i] = (l[i]*u[i])/(u[i]-l[i]) if abs(l[i]) < u[i] else 0
                elif bound_approx == ReluApproximation.MIN_AREA:
                    if abs(l[i]) > u[i] : l[i] =0 
                elif bound_approx == ReluApproximation.ZERO:
                    l[i] = 0
                elif bound_approx == ReluApproximation.PARALLEL:
                    l[i] = (l[i]*u[i])/(u[i]-l[i])
                elif bound_approx == ReluApproximation.IDENTITY:
                    pass
                else:
                    pass
            else:
                l[i]=0
                u[i]=0
    
        return [eqlow, equp], [l, u]


    def max_pooling(self, layer, eqs, layer_bounds, input_bounds):
        if layer.pool_size == (2,2):
            return self._max_pooling_2x2(layer,
                                         eqs, 
                                         layer_bounds, 
                                         input_bounds) 
        else:
            return self._max_pooling_general(layer,
                                            eqs, 
                                            layer_bounds, 
                                            input_bounds) 


    def _max_pooling_general(self, layer, eqs, layer_bounds, input_bounds):
        so, sho = layer.output_size, layer.output_shape
        si, shi = layer.input_size, layer.input_shape
        #get maxpool indices
        coeffs = np.identity(si,dtype='float64')
        const = np.zeros(si, dtype='float64')
        indices = Equations(coeffs,const).maxpool(shi,sho,layer.pool_size)
        # set low equation to the input equation with the highest lower bound
        coeffs_low = np.zeros(shape=(so,si),dtype='float64')
        coeffs_up = coeffs_low.copy()
        const_low = np.zeros(so, dtype='float64')
        const_up = const_low.copy()
        m_coeffs_low = []
        m_coeffs_up = []
        m_const_low = np.zeros(so,dtype='float32')
        m_const_up = np.zeros(so,dtype='float32')
        lb = np.zeros(so, dtype='float64')
        ub = np.zeros(so, dtype='float64')
        for i in range(so):
            lbounds = [np.take(layer_bounds[0],x[i]) for x in indices]
            lb[i] =  max(lbounds)
            index = lbounds.index(lb[i])
            coeffs_low[i,:] = coeffs[indices[index][i],:]
            m_coeffs_low.append({indices[index][i] : 1})
            ubounds = [np.take(layer_bounds[1],x[i]) for x in indices]
            ub[i] = max(ubounds)
            del ubounds[index]
            if (lb[i] > np.array(ubounds)).all():
                coeffs_up[i,:] = coeffs[indices[index][i],:]
                m_coeffs_up.append({indices[index][i] : 1})
            else:
                const_up[i] = ub[i]
                m_const_up[i] = ub[i]
                m_coeffs_up.append({indices[index][i] : 0})
        q_low = Equations(coeffs_low, const_low)
        q_up = Equations(coeffs_up, const_up)
        m_q_low = Equations(m_coeffs_low,m_const_low)
        m_q_up = Equations(m_coeffs_up,m_const_up)

        return [q_low, q_up], [m_q_low, m_q_up], [lb, ub]


    def global_average_pool(self, layer, eqs, input_bounds):
        """
        set pre-activation and activation bounds of a GlobalAveragePooling layer.
        [it is assumed that the previous layer is a convolutional layer]

        layer: a GlobalAveragePooling layer
        eqs: list of linear equations of the outputs of the preceding layers in terms of their input variables
        input_bounds: the lower and upper bounds of the input layer 

        returns: linear equations of the outputs of the layer in terms of its input variables
        """ 
        kernels = layer.input_shape[-1]
        size = reduce(lambda i,j : i*j, layer.input_shape[:-1])
        weight = np.float64(1)/size
        const = np.zeros(kernels,dtype='float64')
        coeffs = np.zeros(shape=(kernels,layer.input_size),dtype='float64')
        for k in range(kernels):
            indices = list(range(k,k+size*kernels,kernels))
            coeffs[k,:][indices] = weight
        eq = Equations(coeffs,const)
        l,u = np.empty(kernels,dtype='float64'), np.empty(kernels,dtype='float64')
        ib_l, ib_u = (input_bounds[i].reshape(layer.input_shape) for i in [0,1])
        for k in range(kernels):
            l[k] = np.average(ib_l[:,:,k])
            u[k] = np.average(ib_u[:,:,k]) 

        return eq, l, u


    def runtime_bounds(self, eqlow_r, equp_r, eq, lb_r, ub_r, lb, ub, relu_states):
        """
        Modifies the given bound equations and bounds so that they are in line
        with the MILP relu states of the nodes.
    
        Arguments:

            eqlow: lower bound equations of a layer's output.

            equp: upper bound equations of a layer's output.

            lbounds: the concrete lower bounds of eqlow.

            ubounds: the concrete upper bounds of equp.
        
            relu_states: a list of floats, one for each node. 0: inactive,
            1:active, anything else: unstable.
        
        Returns:

            a four-typle of the lower and upper bound equations and the
            concreete lower and upper bounds resulting from stablising nodes as
            per relu_states.
        """
        if relu_states is None:
            return [eqlow_r, equp_r], [lb_r, ub_r]
        eqlow_r.coeffs[(relu_states==0),:] = 0
        eqlow_r.const[(relu_states==0)] = 0
        equp_r.coeffs[(relu_states==0),:] = 0
        equp_r.const[(relu_states==0)] = 0
        lb_r[(relu_states == 0)] = 0
        ub_r[(relu_states == 0)] = 0
        eqlow_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        eqlow_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        equp_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        equp_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        lb_r[(relu_states == 0)] = lb[(relu_states == 0)]
        ub_r[(relu_states == 0)] = ub[(relu_states == 0)]

        return [eqlow_r, equp_r], [lb_r, ub_r]


    def branching_bounds(self, relu_states, lb, ub):
        """
        Modifies the given  bounds so that they are in line with the relu
        states of the nodes as per the branching procedure.

        Arguments:
            
            relu_states: relu states of a layer
            
            lb: concrete lower bounds of the layer
            
            ub: concrete upper bounds of the layer

        Returns:
            
            a pair of the lower and upper bounds resulting from stablising
            nodes as per the relu states.
        """
        shape = relu_states.shape
        relu_states = relu_states.reshape(-1)
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        _max = np.clip(lb,0,math.inf)
        _min = np.clip(ub,-math.inf,0)
        lb[(relu_states==ReluState.ACTIVE)] = _max[(relu_states==ReluState.ACTIVE)]
        ub[(relu_states==ReluState.INACTIVE)] = _min[(relu_states==ReluState.INACTIVE)]

        return lb.reshape(shape), ub.reshape(shape)

    def get_adv_labels(self, ground_label):
        """
        Identifies the classes that could potentially cause the violation of
        the adversarial robustness specification.

        Arguments:

            ground_label: the label the input.

        Returns:

            list of ints of the classes.
        """
        labels=[]
        ib = (self.layers[0].post_bounds.lower.flatten(),
              self.layers[0].post_bounds.upper.flatten())
        size = self.layers[-1].output_size

        for i in set(range(size)) - set([ground_label]):
            coeffs = np.zeros(shape=(1,size))
            coeffs[0,ground_label] = 1
            coeffs[0,i] = -1
            const = np.array([0])
            eq = Equations(coeffs, const, self.logfile)
            diff = self.back_substitution(eq, self.eqs, 'lower', ib)
            if diff <= 0 : labels.append(i)

        return labels


    # def simplify_formula(self, formula):
        # print(formula.__str__())
        # if isinstance(formula, VarVarConstraint):
            # if formula.sense == Formula.Sense.GT:
                # op1 = formula.op1.i
                # op2 = formula.op2.i
            # elif formula.sense == Formula.Sense.LT:
                # op1 = formula.op2.i
                # op2 = formula.op1.i
            # else:
                # raise Exception(f"Unexpected sense {formula.sense} in formula")
            # size = self.layers[-1].output_size
            # ib = (self.layers[0].post_bounds.lower.flatten(),
              # self.layers[0].post_bounds.upper.flatten())
            # coeffs = np.zeros(shape=(1,size))
            # coeffs[0,op2] = 1
            # coeffs[0,op1] = -1
            # const = np.array([0])
            # eq = Equations(coeffs, const, self.logfile)
            # diff = self.back_substitution(eq, self.eqs, 'lower', ib)
            # return formula if diff <= 0 else None
        # elif isinstance(formula, DisjFormula):
            # fleft = self.simplify_formula(formula.left)
            # fright = self.simplify_formula(formula.right)
            # return DisjFormula(fleft, fright)
        # elif isinstance(formula, NAryDisjFormula):
            # clauses = [self.simplify_formula(f) for f in formula.clauses]
            # return NAryDisjFormula(clauses)
        # else:
            # return formula



    def simplify_formula(self, formula):
        if isinstance(formula, VarVarConstraint):
            if formula.sense == Formula.Sense.GT:
                op1 = formula.op1.i
                op2 = formula.op2.i
            elif formula.sense == Formula.Sense.LT:
                op1 = formula.op2.i
                op2 = formula.op1.i
            else:
                raise Exception("Unexpected sense {formula.sense} in formula")
            size = self.layers[-1].output_size
            ib = (self.layers[0].post_bounds.lower.flatten(),
              self.layers[0].post_bounds.upper.flatten())
            coeffs = np.zeros(shape=(1,size))
            coeffs[0,op2] = 1
            coeffs[0,op1] = -1
            const = np.array([0])
            eq = Equations(coeffs, const, self.logfile)
            diff = self.back_substitution(eq, self.eqs, 'lower', ib)
            return formula if diff <= 0 else None
        elif isinstance(formula, DisjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)
            if fleft is None and fright is None:
                return None
            elif fleft is None:
                return fright
            elif fright is None:
                return fleft
            else:
                return formula
        elif isinstance(formula, ConjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)
            if fleft is None: return None
            fright = self.simplify_formula(formula.right)
            if fright is None: return None
            return ConjFormula(fleft, fright)
        elif isinstance(formula, NAryDisjFormula):
            clauses = [self.simplify_formula(f) for f in formula.clauses]
            clauses = [cl for cl in clauses if not cl is None]
            if len(clauses) == 0:
                return None
            elif len(clauses) == 1:
                return clauses[0]
            elif len(clauses) == 2:
                return DisjFormula(clauses[0], clauses[1])
            else:
                return NAryDisjFormula(clauses)
        elif isinstance(formula, NAryConjFormula):
            clauses = []
            for cl in formula.clauses:
                s_cl = self.simplify_formula(cl)
                if s_cl is None: return None
                clauses.append(s_cl)
            return  ConjFormula(clauses[0], clauses[1])
        else:
            return formula
            



