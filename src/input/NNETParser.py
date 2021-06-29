import numpy as np
import keras
import math
from src.layers import FullyConnected, Conv2D, Activations

class NNETParser:

    SUPPORTED_LAYERS = [keras.layers.Dense, 
                        keras.layers.Conv2D, 
                        keras.layers.Activation, 
                        keras.layers.Flatten, 
                        keras.layers.MaxPooling2D, 
                        keras.layers.GlobalAveragePooling2D]
    SUPPORTED_ACTIVATIONS = [keras.activations.linear,
                             keras.activations.relu]
    activations_map = {keras.activations.linear: Activations.linear,
                       keras.activations.relu: Activations.relu}
    activations_r_map = {Activations.linear : keras.activations.linear,
                         Activations.relu : keras.activations.relu}
 
    def __init__(self):
        """
        """

    def to_nnet(self, nn, filepath):
        self.nnet_headers(nn, filepath)
        for l in nn.layers:
            if isinstance(l, FullyConnected):
                self.venus_ff_to_nnet(l, filepath)
            elif isinstance(l, Conv2D):
                self.venus_conv_to_nnet(l, filepath)

    def nnet_headers(self, nn, filepath):
        for l in nn.layers:
            assert isinstance(l,FullyConnected) or isinstance(l,Conv2D), "The converter supports only fully connected and convolutional layers"
        fl = open(filepath, 'w')
        sizes = [nn.layers[0].input_size] + [l.output_size for l in nn.layers]
        fl.write('{},{},{},{},\n'.format(len(nn.layers),
                                      nn.layers[0].input_size,
                                      nn.layers[-1].output_size,
                                      max(sizes)))
        for i in sizes:
            fl.write(f'{i},')
        fl.write('\n')
        for l in nn.layers:   
            if isinstance(l,FullyConnected):
                fl.write('0,')
            elif isinstance(l,Conv2D):
                fl.write('1,')
        fl.write('\n')
        for l in nn.layers:
            if isinstance(l,Conv2D):
                fl.write('{},{},{},{},{},\n'.format(l.output_shape[-1],
                                            l.input_shape[-1],
                                            l.kernels.shape[0],
                                            l.strides[0],
                                            l.padding[0]))
        fl.close()

    def venus_ff_to_nnet(self, layer, filepath):
        fl = open(filepath,'a') 
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                fl.write(f'{layer.weights[i][j]},')
            fl.write('\n')
        for i in layer.bias:
            fl.write(f'{i},\n')
        fl.close() 

    def venus_conv_to_nnet(self, layer, filepath):
        fl = open(filepath,'a')
        for oc in range(layer.kernels.shape[3]):
            for ic in range(layer.kernels.shape[3]):
                for w in range(layer.kernels.shape[1]):
                    for h in range(layer.kernels.shape[1]):
                        fl.write(f'{layer.kernels[w][h][ic][oc]},')
            fl.write('\n') 
        for i in layer.bias:
            fl.write(f'{i},\n')
        fl.close()
