import numpy as np
import onnx
import onnx.numpy_helper
import math
from src.Layers import FullyConnected, Conv2D
from src.utils.Activations import Activations
from functools import reduce

class ONNXParser:

    SUPPORTED_NODES = ['Flatten', 'Shape', 'Constant', 'Concat', 'Reshape', 'Unsqueeze', 'Gather', 'Relu', 'Gemm', 'Conv', 'Transpose', 'MatMul', 'Add', 'Div', 'Sub']
 
    def __init__(self):
        """
        """
        self.mean =  0
        self.std = 1

    def load(self, model_path):
        model = onnx.load(model_path)
        layers = []
        depth = 1
        nodes = model.graph.node
        onnx_input_shape = model.graph.input[0].type.tensor_type.shape
        input_shape = self.onnx_to_venus_shape(onnx_input_shape)

        p_node = None
        for node in nodes:
            assert node.op_type in self.SUPPORTED_NODES, f'node {node.op_type} is not supported'
            if node.op_type in ['Shape', 'Sub', 'Gather', 'Concat', 'Unsqueeze', 'Reshape']:
                continue
            elif node.op_type == 'Transpose':
                for att in node.attribute:
                    if att.name == 'perm':
                        perms = [i for i in att.ints]
                        input_shape = tuple([input_shape[i-1] for i in perms[1:]])
                continue
            elif node.op_type == 'Flatten':
                input_shape = (reduce(lambda i,j : i*j, input_shape),)
                continue
            elif node.op_type == 'Gemm':
                layers.append(self.onnx_gemm_to_venus(model, node, input_shape, depth))
                depth += 1
            elif node.op_type == 'MatMul':
                layers.append(self.onnx_matmul_to_venus(model, node, input_shape, depth))
                depth +=1 
            elif node.op_type == 'Add':
                [bias] = [onnx.numpy_helper.to_array(t).astype(float)
                    for t in model.graph.initializer
                    if t.name == node.input[1]]
                if depth == 1:
                    bias = np.squeeze(bias)
                    self.mean = [bias[1],bias[2],bias[0]]
                else:
                    layers[-1].bias = bias 
            elif node.op_type == 'Div':
                if depth != 1:
                    raise Exception('Div not supported at depth {depth}')
                if p_node.op_type == 'Constant':
                    self.std = self.const.copy()
            elif node.op_type == 'Sub':
                if depth != 1:
                    raise Exception('Sub not supported at depth {depth}')
                if p_node.op_type == 'Constant':
                    self.mean = self.const.copy()
            elif node.op_type == 'Constant':
                const = node.attribute
                const =  np.squeeze(onnx.numpy_helper.to_array(const[0].t)).copy()
                self.const = [const[1], const[2], const[0]]
            elif node.op_type == 'Conv':
                layers.append(self.onnx_conv_to_venus(model, node, input_shape, depth))
                depth += 1
            elif node.op_type == 'Relu':
                layers[-1].set_activation(Activations.relu)
            
            if len(layers) > 0:
                input_shape = layers[-1].output_shape

            p_node = node

        return layers


    def onnx_gemm_to_venus(self, model, node, input_shape, depth):
        [weights] = [onnx.numpy_helper.to_array(t).astype(float)
                    for t in model.graph.initializer
                    if t.name == node.input[1]]
        for att in node.attribute:
            if att.name == 'transB' and att.i == 0:
                weights = weights.T
        self.onnx_to_venus_ff_indexing(input_shape, weights)
        [bias] = [onnx.numpy_helper.to_array(t).astype(float)
                for t in model.graph.initializer
                if t.name == node.input[2]]
        input_shape = (weights.shape[1],)
        output_shape = (weights.shape[0],)
      
        return FullyConnected(input_shape,
                              output_shape,
                              weights,
                              bias,
                              Activations.linear,
                              depth)

    def onnx_matmul_to_venus(self, model, node, input_shape, depth):
        weights = [onnx.numpy_helper.to_array(t).astype(float)
                    for t in model.graph.initializer
                    if t.name == node.input[1]]
        if len(weights) == 0:
            [weights] = [onnx.numpy_helper.to_array(t).astype(float)
                        for t in model.graph.initializer
                        if t.name == node.input[0]]
        else:
            [weights] = weights
            weights = weights.T
            for att in node.attribute:
                if att.name == 'transB' and att.i == 0:
                    weights = weights.T
        self.onnx_to_venus_ff_indexing(input_shape, weights)
        input_shape = (weights.shape[1],)
        output_shape = (weights.shape[0],)
        bias = np.zeros(output_shape)
      
        return FullyConnected(input_shape,
                              output_shape,
                              weights,
                              bias,
                              Activations.linear,
                              depth)

    def onnx_conv_to_venus(self, model, node, input_shape, depth):
        [weights] = [onnx.numpy_helper.to_array(t).astype(float)
                     for t in model.graph.initializer
                     if t.name == node.input[1]]
        weights = self.onnx_to_venus_conv_indexing(weights)
        [bias] = [onnx.numpy_helper.to_array(t).astype(float)
              for t in model.graph.initializer
              if t.name == node.input[2]]
        pads = (0,0)
        strides = (1,1)
        for att in node.attribute:
            if att.name == "pads":
                pads = tuple([i for i in att.ints[0:2]])
            elif att.name == "strides":
                strides = tuple([i for i in att.ints[0:2]])
        output_shape = Conv2D.compute_output_shape(input_shape,
                                                   weights.shape,
                                                   pads,
                                                   strides)
        return Conv2D(input_shape, 
                      output_shape,
                      weights,
                      bias,
                      pads,
                      strides,
                      Activations.linear,
                      depth)

    def onnx_to_venus_conv_indexing(self, matrix):
        return matrix.transpose(2,3,1,0) 
   
    def onnx_to_venus_ff_indexing(self, shape_in, matrix):
        assert len(shape_in)==1 or len(shape_in)==3, f'Input shape, {shape_in} is not supported'
        if len(shape_in) == 3:
            shape = (shape_in[2],shape_in[0],shape_in[1])
            for i in range(matrix.shape[0]):
                matrix[i,:] = matrix[i,:].reshape(shape).transpose(1,2,0).reshape(-1)
     
        return matrix

    def onnx_to_venus_shape(self, onnxshape):
        shape =  [i.dim_value for i in onnxshape.dim]
        if len(shape) == 4: shape  = (shape[2],shape[3],shape[1])
        return shape
