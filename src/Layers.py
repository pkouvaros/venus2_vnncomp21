# ************
# File: Layers.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: classes for layers suppported by Venus.
# ************

from src.Bounds import Bounds
from src.utils.Activations import Activations
from src.utils.ReluState import ReluState
from functools import reduce
import numpy as np
import itertools
import math

class Layer():
    def __init__(self, input_shape, output_shape, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            depth: depth of the layer in the network.
        """
        self.input_shape = input_shape
        self.input_size = reduce(lambda i,j : i*j, input_shape)
        self.output_shape = output_shape 
        self.output_size = reduce(lambda i,j : i*j, output_shape)
        self.depth = depth
        self.out_vars =  np.empty(0)
        self.delta_vars = np.empty(0)
        self.pre_bounds = Bounds()
        self.post_bounds = Bounds()
        self.activation = None

    def outputs(self):
        """
        Constructs a list of the indices of the nodes in the layer.

        Returns: 

            list of indices.
        """
        if len(self.output_shape)>1:
            return [i for i in itertools.product(*[range(j) for j in self.output_shape])]
        else:
            return list(range(self.output_size))

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.

        Returns 

            None
        """
        self.out_vars = np.empty(0)
        self.delta_vars = np.empty(0)

class Input(Layer):
    def __init__(self, lower, upper, name='input'):
        """
        Argumnets:
            
            lower: array of lower bounds of input nodes.
            
            upper: array of upper bounds of input nodes.
        """
        self.depth = 0
        self.out_vars = np.empty(0)
        self.delta_vars = np.empty(0)
        self.pre_bounds = self.post_bounds = Bounds(lower,upper)
        self.input_shape = self.output_shape = lower.shape
        self.input_size = self.output_size = reduce(lambda i,j : i*j, self.input_shape)
        self.activation = None
        self.name=name

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Input(self.post_bounds.lower, self.post_bounds.upper)

class GlobalAveragePooling(Layer):
    def __init__(self, input_shape, output_shape, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape,output_shape,depth)

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return GlobalAveragePooling(self.input_shape, self.output_shape, self.depth)


class MaxPooling(Layer):
    def __init__(self, input_shape, output_shape, pool_size, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            pool_size: pair of int for the width and height of the pooling.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape,output_shape,depth)
        self.pool_size = pool_size

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return MaxPooling(self.input_shape, 
                          self.output_shape, 
                          self.pool_size,
                          self.depth)

class AffineLayer(Layer):
    def __init__(self, input_shape, output_shape, weights, bias, activation, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            weights: weight matrix.

            bias: bias vector.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape,output_shape,depth)
        self.weights = weights
        self.bias = bias
        self.activation = activation
        if activation == Activations.relu:
            self.state = np.array([ReluState.UNSTABLE]*self.output_size,dtype=ReluState).reshape(self.output_shape)
            self.dep_root = np.array([False]*self.output_size).reshape(self.output_shape)
            self.dep_consistency = [True for i in range(self.output_size)]

    def set_activation(self, activation):
        """
        Sets the activation of the layer.

        Arguments:
            
            activation: Activation.
        """
        self.activation = activation
        if activation == Activations.linear:
            self.state = self.dep_root = self.dep_consistency = None
        else:
            self.state = np.array([ReluState.UNSTABLE]*self.output_size,dtype=ReluState).reshape(self.output_shape)
            self.dep_root = np.array([False]*self.output_size).reshape(self.output_shape)
            self.dep_consistency = [True for i in range(self.output_size)]


    def is_active(self, node):
        """
        Detemines whether a given ReLU node is stricly active.

        Arguments:

            node: index of the node whose active state is to be determined.

        Returns:

            bool expressing the active state of the given node.
        """
        if not self.activation == Activations.relu:
            return []
        if self.pre_bounds.lower[node] >= 0 or self.state[node] == ReluState.ACTIVE:
            return True
        else:
            return False

    def is_inactive(self, node):
        """
        Detemines whether a given ReLU node is stricly inactive.

        Arguments:

            node: index of the node whose inactive state is to be determined.

        Returns:

            bool expressing the inactive state of the given node.
        """
        if not self.activation == Activations.relu:
            return []
        if self.pre_bounds.upper[node] <= 0 or self.state[node] == ReluState.INACTIVE:
            return True
        else:
            return False


    def get_active(self):
        """
        Determines the active nodes of the layer.

        Returns: 

            A list of indices of  the active nodes.
        """
        if not self.activation == Activations.relu:
            return []
        return [i for i in range(self.output_size) is self.is_active(i)]

    def get_inactive(self):
        """
        Determines the inactive nodes of the layer.

        Returns: 

            A list of indices of  the inactive nodes.
        """
        if not self.activation == Activations.relu:
            return []
        return [i for i in range(self.output_size) if self.is_inactive(i)]

    def is_stable(self, node, delta_val=None):
        """
        Detemines whether a given ReLU node is stable.

        Arguments:

            node: index of the node whose active state is to be determined.

            delta_val: value of the binary variable associated with the node.
            if set, the value is also used in conjunction with the node's
            bounds to determined its stability.

        Returns:

            bool expressing the stability of the given node.
        """
        if not self.activation == Activations.relu:
            return None
        cond1 =  self.pre_bounds.lower[node] >= 0 or self.pre_bounds.upper[node] <= 0
        cond2 = not self.state[node] == ReluState.UNSTABLE
        cond3 = False if delta_val is None else (delta_val==0 or delta_val==1)
        return cond1 or cond2 or cond3

    def get_unstable(self, delta_vals=None):
        """
        Determines the unstable nodes of the layer.

        Arguments:

            delta_vals: values of the binary variables associated with the nodes.
            if set, the values are also used in conjunction with the nodes'
            bounds to determined their instability.
         
        Returns: 

            A list of indices of the unstable nodes.
        """
        if not self.activation == Activations.relu:
            return []
        if delta_vals is None:
            return [i for i in self.outputs() if not self.is_stable(i)]
        else:
            return [i for i in self.outputs() if not self.is_stable(i,delta_vals[i])]


    def get_stable(self, delta_vals=None):
        """
        Determines the stable nodes of the layer.

        Arguments:

            delta_vals: values of the binary variables associated with the nodes.
            if set, the values are also used in conjunction with the nodes'
            bounds to determined their stability.
         
        Returns: 

            A list of indices of the stable nodes.
        """
        if not self.activation == Activations.relu:
            return []
        if delta_vals is None:
            return [i for i in self.outputs() if self.is_stable(i)]
        else:
            return [i for i in self.outputs() if self.is_stable(i,delta_vals[i])]


class FullyConnected(AffineLayer):
    def __init__(self, input_shape, output_shape, weights, bias, activation, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            weights: weight matrix.

            bias: bias vector.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape, output_shape, weights, bias, activation, depth)
        self.vars = {'out': np.empty(0), 'delta': np.empty(0)}

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        fc =  FullyConnected(self.input_shape, self.output_shape, self.weights, self.bias, self.activation, self.depth)
        fc.pre_bounds = self.pre_bounds.copy()
        fc.post_bounds = self.post_bounds.copy()
        if self.activation == Activations.relu:
            fc.state = self.state.copy()
            fc.dep_root = self.dep_root.copy()
            fc.dep_consistency = self.dep_consistency.copy()
        return fc

    def neighbours(self, l, node):
        """
        Determines the neighbouring nodes to the given node from the previous
        layer.

        Arguments:

            node: the index of the node.

        Returns:

            a list of neighbouring nodes.
        """
        if len(l.output_shape)>1:
            return [i for i in itertools.product(*[range(j) for j in l.output_shape])]
        else:
            return list(range(self.input_size))

    def get_bias(self, node):
        """
        Returns the bias of the given node.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias 
        """
        return self.bias[node]

    def edge_weight(self, l, node1, node2):
        """
        Returns the weight of the edge between node1 of the current layer and
        node2 of the previous layer.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias.
        """
        if isinstance(l, Conv2D) and isinstance(node2, tuple) and len(node2) > 1:
            node2 = l.flattened_index(node2)

        return self.weights[node1][node2]
    
    def intra_connected(self, n1, n2):
        """
        Determines whether two given nodes share connections with nodes from
        the previous layers.

        Arguments:

            n1: index of node 1

            n2: index of node 2

        Returns:
            
            bool expressing whether the nodes are intra-connected
        """
        return True

    def joint_product(self, inp, n1, n2):
        return inp, (self.weights[n1,:], self.weights[n2,:]), (self.bias[n1],self.bias[n2])


class Conv2D(AffineLayer):
    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 kernels, 
                 bias, 
                 padding, 
                 strides, 
                 activation, 
                 depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            kernerls: kernels : matrix.

            bias: bias vector.

            padding: pair of int for the width and height of the padding.
            
            strides: pair of int for the width and height of the strides.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape, output_shape, kernels, bias, activation, depth)
        self.vars = {'out': np.empty(0), 'delta': np.empty(0)}
        self.kernels = kernels
        self.bias = bias
        self.padding = padding
        self.strides = strides
        self.depth = depth

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        conv = Conv2D(self.input_shape, self.output_shape, self.kernels, self.bias, self.padding, self.strides, self.activation, self.depth) 
        conv.pre_bounds = self.pre_bounds.copy()
        conv.post_bounds = self.post_bounds.copy()
        if self.activation == Activations.relu:
            conv.state = self.state.copy()
            conv.dep_root = self.dep_root.copy()
            conv.dep_consistency = self.dep_consistency.copy()
        return conv

    def flattened_index(self, node):
        """
        Computes the flattened index of the given nodes.

        Arguments:

            node: tuple of the index of the node.

        Returns:

            int of the flattensed index of the node.
        """
        X,Y,Z = self.output_shape
        return node[0] * Y * Z + node[1] * Z + node[2]
       
    def neighbours(self, node):
        """
        Determines the neighbouring nodes to the given node from the previous
        layer.

        Arguments:

            node: the index of the node.

        Returns:

            a list of neighbouring nodes.
        """
        x_start = node[0] * self.strides[0] - self.padding[0]
        x_rng = range(x_start, x_start + self.kernels.shape[0])
        x = [i for i in x_rng if i >= 0 and i<self.input_shape[0]]
        y_start = node[1] * self.strides[1] - self.padding[1]
        y_rng = range(y_start, y_start + self.kernels.shape[1])
        y = [i for i in y_rng if i>=0 and i<self.input_shape[1]]
        z = [i for i in range(self.kernels.shape[2])]
        return [i for i in itertools.product(*[x,y,z])]


    def intra_connected(self, n1, n2):
        """
        Determines whether two given nodes share connections with nodes from
        the previous layers.

        Arguments:

            n1: index of node 1

            n2: index of node 2

        Returns:
            
            bool expressing whether the nodes are intra-connected
        """
        n_n1 = self.neighbours(n1)
        n_n2 = self.neighbours(n2)
        return len(set(n_n1) & set(n_n2)) > 0

    def get_bias(self, node):
        """
        Returns the bias of the given node.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias 
        """
        return self.bias[node[-1]]

    def edge_weight(self, l, node1, node2):
        """
        Returns the weight of the edge between node1 of the current layer and
        node2 of the previous layer.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias.
        """
        x_start = node1[0] * self.strides[0] - self.padding[0]
        x = node2[0] - x_start
        y_start = node1[1] * self.strides[1] - self.padding[1]
        y = node2[1] - y_start
        return self.kernels[x][y][node2[2]][node1[2]]


    @staticmethod 
    def compute_output_shape(in_shape, weights_shape, pads, strides):
        """
        Computes the output shape of a convolutional layer.

        Arguments:

            in_shape: shape of the input tensor to the layer.

            weights_shape: shape of the kernels of the layer.

            padding: pair of int for the width and height of the padding.
            
            strides: pair of int for the width and height of the strides.

        Returns:

            tuple of int of the output shape
        """
        x,y,z = in_shape
        X,Y,_,K = weights_shape
        p,q  = pads
        s,r = strides
        out_x = int(math.floor( (x-X+2*p) / s + 1 ))
        out_y = int(math.floor( (y-Y+2*q) / r + 1 ))
        
        return (out_x,out_y,K)

    @staticmethod
    def pad(A, PS, values=(0,0)):
        """
        Pads a given matrix with zeroes

        Arguments:
            
            A: matrix:
            
            PS: tuple denoting the padding size

        Returns

            padded A.
        """
        if PS==(0,0):
            return A
        else:
            return np.pad(A,(PS,PS,(0,0)),'constant',constant_values=values)


    @staticmethod
    def im2col(A, KS, strides, indices=False):
        """
        MATLAB's im2col function

        Arguments:
            
            A: magtrix
            
            KS: tuple of kernel shape
        
            strides: tuple of int of strides of the convolution

        Returns:

            im2col matrix
        """
        M,N,K = A.shape
        col_extent = N - KS[1] + 1
        row_extent = M - KS[0] + 1
    
        # starting block indices
        start_idx = np.arange(KS[0])[:,None]*N*K + np.arange(KS[1]*K)

        # offsetted indices across the height and width of A
        offset_idx = np.arange(row_extent)[:,None][::strides[0]]*N*K + np.arange(0,col_extent*K,strides[1]*K)

        # actual indices 
        if indices == True:
            return start_idx.ravel()[:,None] + offset_idx.ravel()
        else:
            return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

    def joint_product(self, inp, n1, n2):
        X,Y,Z,_ = self.kernels.shape
        P = self.padding
        inpl = np.pad(inp.lower.reshape(self.input_shape),(P,P,(0,0)), 'constant', constant_values=(0,0))
        inpu = np.pad(inp.upper.reshape(self.input_shape),(P,P,(0,0)), 'constant', constant_values=(0,0))
        #row and column indices where receptive fields begin and end
        x_s = {1: n1[0]*self.strides[0], 2: n2[0]*self.strides[0]}
        x_e = {1: x_s[1] + X, 2: x_s[2] + X}
        y_s = {1: n1[1]*self.strides[1], 2: n2[1]*self.strides[1]}
        y_e = {1: y_s[1] + Y, 2: y_s[2] + Y}
        x_min = 1 if x_s[1] < x_s[2] else 2
        x_max = x_min % 2 + 1
        y_min = 1 if y_s[1] < y_s[2] else 2
        y_max = y_min % 2 + 1
        K = {1: n1[-1], 2: n2[-1]}
        # wx1, wy1: row and column ranges covered by the recepting field of n1 and only
        # wx2, wy2: row and column ranges covered by the recepting field of n1 and n2
        # # wx3, wy4: row and column ranges covered by the recepting field of n2 and only
        wx1, wx2, wx3 = x_s[x_max] - x_s[x_min],  x_e[x_min] - x_s[x_max], x_e[x_max] - x_e[x_min]
        wy1, wy2, wy3 = y_s[y_max] - y_s[y_min],  y_e[y_min] - y_s[y_max], y_e[y_max] - y_e[y_min]
        w1 = wx1*Y*Z
        w2 = w1 + wx2*wy1*Z
        w3 = w2 + wx2*wy2*Z
        w4 = w3 + wx2*wy3*Z
        w5 = w4 + wx3*Y*Z 
        # contrsuct the combined receptive field and weights
        rfl= np.empty(w5,dtype='float64')
        rfu= np.empty(w5,dtype='float64')
        weights = {1: np.empty(w5,dtype='float64'), 2:np.empty(w5,dtype='float64')}
        rfl[0:w1] = inpl[x_s[x_min]:x_s[x_max],y_s[x_min]:y_e[x_min],:].flatten() 
        rfu[0:w1] = inpu[x_s[x_min]:x_s[x_max],y_s[x_min]:y_e[x_min],:].flatten()
        weights[x_min][0:w1] = self.kernels[0:wx1,:,:,K[x_min]].flatten()
        weights[x_max][0:w1] = np.zeros(w1)
        rfl[w1:w2] = inpl[x_s[x_max]:x_e[x_min],y_s[y_min]:y_s[y_max],:].flatten()
        rfl[w2:w3] = inpl[x_s[x_max]:x_e[x_min],y_s[y_max]:y_e[y_min],:].flatten()
        rfl[w3:w4] = inpl[x_s[x_max]:x_e[x_min],y_e[y_min]:y_e[y_max],:].flatten()
        rfu[w1:w2] = inpu[x_s[x_max]:x_e[x_min],y_s[y_min]:y_s[y_max],:].flatten()
        rfu[w2:w3] = inpu[x_s[x_max]:x_e[x_min],y_s[y_max]:y_e[y_min],:].flatten()
        rfu[w3:w4] = inpu[x_s[x_max]:x_e[x_min],y_e[y_min]:y_e[y_max],:].flatten()
        if x_min == y_min:
            weights[x_min][w1:w2] = self.kernels[wx1:wx1+wx2,0:wy1,:,K[x_min]].flatten()
            weights[x_max][w1:w2] = np.zeros(w2-w1) 
            weights[x_min][w2:w3] = self.kernels[wx1:wx1+wx2,wy1:wy1+wy2,:,K[x_min]].flatten()
            weights[x_max][w2:w3] = self.kernels[0:wx2,0:wy2,:,K[x_max]].flatten()
            weights[x_min][w3:w4] = np.zeros(w4-w3)
            weights[x_max][w3:w4] = self.kernels[0:wx2,wy2:wy2+wy3,:,K[x_max]].flatten()
        else:
            weights[x_min][w1:w2] = np.zeros(w2-w1)
            weights[x_max][w1:w2] = self.kernels[0:wx2,0:wy1,:,K[x_max]].flatten()
            weights[x_min][w2:w3] = self.kernels[wx1:wx1+wx2,0:wy2,:,K[x_min]].flatten()
            weights[x_max][w2:w3] = self.kernels[0:wx2,wy1:wy1+wy2,:,K[x_max]].flatten()
            weights[x_min][w3:w4] = self.kernels[wx1:wx1+wx2,wy2:wy2+wy3,:,K[x_min]].flatten()
            weights[x_max][w3:w4] = np.zeros(w4-w3)                                         
        rfl[w4:w5] = inpl[x_e[x_min]:x_e[x_max],y_s[x_max]:y_e[x_max],:].flatten()
        rfu[w4:w5] = inpu[x_e[x_min]:x_e[x_max],y_s[x_max]:y_e[x_max],:].flatten()
        weights[x_min][w4:w5] = np.zeros(w5-w4)
        weights[x_max][w4:w5] = self.kernels[wx2:wx2+wx3,:,:,K[x_max]].flatten()

        rf = Bounds(rfl, rfu)
        return rf, (weights[1], weights[2]), (self.bias[n1[-1]],self.bias[n2[-1]])

