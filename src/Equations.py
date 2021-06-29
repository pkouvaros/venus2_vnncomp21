# ************
# File: Equations.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description:  The equation class provides a representation for  the bound
# equations and implements  relu relaxations and dot products on them. 
# ************

import numpy as np
import math
from src.utils.ReluApproximation import ReluApproximation
# from src.utils.Logger import get_logger

class Equations:

    # logger = None 

    def __init__(self, coeffs, const, logfile):
        """
        Arguments:

            coeffs: 2D matrix (of size nxm). Each row i represents node's i
                    equation coefficients.
            
            const:  vector (of size n). Each row i represents node's i equation
                    constant term

            logfile: str of path to logfile.
        """
        self.coeffs = coeffs
        self.const = const
        self.size = coeffs.shape[0]
        self.lower_bounds = None
        self.upper_bounds = None
        self.logfile = logfile
        self.is_slope_optimised = False
        self.lower_slope_l_bound = None
        self.lower_slope_u_bound = None
        # if Equations.logger is None:
            # Equations.logger = get_logger(__name__, logfile)

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Equations(self.coeffs.copy(), self.const.copy(), logfile=self.logfile)

    def max_values(self, lower, upper):
        """
        Computes the upper bounds of the equations.
    
        Arguments:

            lower: vector of the lower bounds for the variables in the equation
            
            upper: vector of the upper bounds for the variables in the equation
        
        Returns: 
            
            a vector for the upper bounds of the equations 
        """
        if not self.upper_bounds is None:
            return self.upper_bounds

        minus = np.clip(self.coeffs,-math.inf,0)
        plus = np.clip(self.coeffs,0,math.inf)
        self.upper_bounds = plus.dot(upper) + minus.dot(lower) + self.const
        
        return self.upper_bounds

    def min_values(self, lower, upper):
        """
        Computes the lower bounds of the equations 

        Arguments:
            
            lower: vector of the lower bounds for the variables in the equation
            
            upper: vector of the upper bounds for the variables in the equation
        
        Returns: 

            a vector of upper bounds of the equations 
        """  
        if not self.lower_bounds is None:
            return self.lower_bounds

        minus = np.clip(self.coeffs,-math.inf,0)
        plus = np.clip(self.coeffs,0,math.inf)
        self.lower_bounds = plus.dot(lower) + minus.dot(upper) + self.const
        
        return self.lower_bounds


    def set_lower_slope(self, lbound, ubound):
        """ 
        Sets the lower slopes for the equations, one for computing the lower
        bounds during back-substitution and one for computing the upper bound.

        Arguments:
            
            lbound: vector of the lower slope for the lower bound.
            
            ubound: vector of the lower slope for the upper bound.

        Returns:

            None
        """
        self.lower_slope_l_bound = lbound
        self.lower_slope_u_bound = ubound
        self.is_slope_optimised = True


    def get_lower_slope(self, l, u, approx=ReluApproximation.MIN_AREA):
        """
        Derives the slope of the lower linear relaxation of the equations.

        Arguments:

            lower: vector of lower bounds of the equations
            
            upper: vector of upper bounds of the equations

            approx: ReluApproximation

        Returns:

            vector of the slopes.
        """

        slope = np.zeros(self.size)
        l, u = l.flatten(), u.flatten()

        for i in range(self.size):
            if  l[i] >= 0:
                slope[i] = 1
            elif u[i] <= 0: 
                pass
            else:
                if approx == ReluApproximation.ZERO:
                    pass
                elif approx == ReluApproximation.IDENTITY:
                    slope[i] = 1
                elif approx == ReluApproximation.PARALLEL:
                    slope[i] = u[i] / (u[i] - l[i])
                elif approx == ReluApproximation.MIN_AREA:
                    if abs(l[i]) < u[i]: slope[i] = 1
                elif approx == ReluApproximation.VENUS_HEURISTIC:
                    if abs(l[i]) < u[i]: slope[i] = u[i] / (u[i] - l[i])
                else:
                    pass

        return slope 



    def lower_relu_relax(self, 
                         lower=None, 
                         upper=None, 
                         approx=ReluApproximation.MIN_AREA):
        """
        Derives the ReLU lower linear relaxation of the equations

        Arguments:

            lower: vector of lower bounds of the equations
            
            upper: vector of upper bounds of the equations

            approx: ReluApproximation

        Returns:

            Equations of the relaxation.
        """
        if lower is None:
            if self.lower_bounds is None:
                raise Exception("Missing lower bounds")
            else:
                lower = self.lower_bounds
        if upper is None:
            if self.upper_bounds is None:
                raise Exception("Missing upper bounds")
            else:
                upper = self.upper_bounds

        coeffs = self.coeffs.copy()
        const = self.const.copy()
        # compute the coefficients of the linear approximation of out bound
        # equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0: 
                # Inactive node - Propagate the zero function
                coeffs[i,:], const[i] =  0, 0
            else:
                # Unstable node - Propagate linear relaxation of lower bound
                # equations
                # 
                if approx == ReluApproximation.ZERO:
                    coeffs[i,:], const[i] = 0, 0
                elif approx == ReluApproximation.IDENTITY:
                    pass
                elif approx == ReluApproximation.PARALLEL:
                    coeffs[i,:], const[i] = self.parallel(coeffs[i,:],
                                                          const[i],
                                                          lower[i], 
                                                          upper[i], 
                                                          'lower') 
                elif approx == ReluApproximation.MIN_AREA:
                    coeffs[i,:], const[i] = self.min_area(coeffs[i,:],
                                                          const[i],
                                                          lower[i], 
                                                          upper[i])
                elif approx == ReluApproximation.VENUS_HEURISTIC:
                    coeffs[i,:], const[i] = self.venus_heuristic(coeffs[i,:],
                                                                 const[i],
                                                                 lower[i], 
                                                                 upper[i])
                else:
                    pass


        return Equations(coeffs, const, self.logfile)

    def upper_relu_relax(self, lower=None, upper=None):
        """
        Derives the ReLU upper linear relaxation of the equations

        Arguments:

            lower: vector of lower bounds of the equations
            
            upper: vector of upper bounds of the equations

        Returns:

            Equations of the relaxation.
        """

        if lower is None:
            if self.lower_bounds is None:
                raise Exception("Missing lower bounds")
            else:
                lower = self.lower_bounds
        if upper is None:
            if self.upper_bounds is None:
                raise Exception("Missing upper bounds")
            else:
                upper = self.upper_bounds

        coeffs = self.coeffs.copy()
        const = self.const.copy()
        # compute the coefficients of the linear approximation of out bound
        # equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0:  
                # Inactive node - Propagate the zero function
                coeffs[i,:], const[i] =  0, 0
            else:
                # Unstable node - Propagate linear relaxation of lower bound equations
                coeffs[i,:], const[i] = self.parallel(coeffs[i,:],
                                                      const[i],
                                                      lower[i],
                                                      upper[i],
                                                      'upper')

        return Equations(coeffs, const, self.logfile)


    def parallel(self, coeffs, const, l, u, bound_line):
        """
        Parallel ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
            bound_line: either 'upper' or 'lower' approximation.

        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if not bound_line in ['lower','upper']:
            raise Exception('Got invalid bound line')
        
        adj =  u / (u - l)
        coeffs *= adj
        if bound_line == 'lower':
            const *= adj
        else:
            const  = const * adj - adj * l

        return coeffs, const

    def min_area(self, coeffs, const, l, u):
        """
        Minimum Area ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if abs(l) < u:
            return coeffs, const
        else:
            return 0,0

    def venus_heuristic(self, coeffs, const, l, u):
        """
        Venus heuristic ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if abs(l) < u:
            return self.parallel(coeffs, const, l, u, 'lower')
        else:
            return 0, 0

    def dot(self, bound, eqlow, equp, slopes = None):
        """
        Computes the dot product of the equations with given lower and upper
        bound equations

        Arguments:
            
            bound: either 'lower' or 'upper'. Determines whether the lower or
            upper bound equation of the dot product will be derived.

            eqlow: lower bound equations.

            equp: upper bound equations.

            slopes: slopes for the lower bound equation.

            eqs: list of indiced of equations for which to carry out the
            procuct. If None, then all equations are considered.

        Returns:
            
            Equations of the dot product.
        """
        if not bound in ['lower','upper']:
            raise Exception('Got invalid bound line')
  
        if not slopes is None:
            l_coeffs = eqlow.coeffs * np.expand_dims(slopes, axis=1)
            l_const = eqlow.const * slopes
        elif eqlow.is_slope_optimised == True:
            if bound=='lower':
                l_coeffs = eqlow.coeffs * np.expand_dims(eqlow.lower_slope_l_bound,axis=1)
                l_const = eqlow.const * eqlow.lower_slope_l_bound
            else:
                l_coeffs = eqlow.coeffs * np.expand_dims(eqlow.lower_slope_u_bound,axis=1)
                l_const = eqlow.const * eqlow.lower_slope_u_bound
        else:
            l_coeffs = eqlow.coeffs
            l_const = eqlow.const

        plus = np.clip(self.coeffs,0,math.inf,dtype='float64')
        minus=np.clip(self.coeffs,-math.inf,0,dtype='float64')
        if bound == 'lower':
            coeffs = np.dot(plus, l_coeffs) + np.dot(minus, equp.coeffs)
            const = np.dot(plus, l_const) + np.dot(minus, equp.const) + self.const
        else:
            coeffs = np.dot(plus, equp.coeffs) + np.dot(minus, l_coeffs)
            const = np.dot(plus, equp.const) + np.dot(minus, l_const) + self.const
        
        return Equations(coeffs, const, self.logfile)

    def pool(self, in_shape, out_shape, pooling):
        """
        Derives the pooling indices of the equations.

        Arguments:
            
            in_shape: tuple of the shape of the equations.

            out_shape: tuple the shape of the equations after pooling.

            pooling: tuple of the pooling size.
        
        Returns:

            List where each item i in the list is a list of indices of the i-th
            pooling neighbourhood.
        """
            
        m,n,_ = in_shape
        M,N,K =  out_shape
        p = pooling[0]
        m_row_extent = M*N*K
        # get Starting block indices
        start_idx = np.arange(p)[:,None]*n*K + np.arange(0,p*K,K)
        # get offsetted indices across the height and width of input 
        offset_idx = np.arange(M)[:,None]*n*K*p + np.array([i*p*K + np.arange(K) for i in  range(N)]).flatten()
        # get all actual indices 
        idx = start_idx.ravel()[:,None] + offset_idx.ravel()
        return [idx[i,:] for i in idx.shape[0]]

