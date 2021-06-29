import numpy as np
import keras
import math
from src.Layers import FullyConnected, Conv2D
from src.utils.Activations import Activations

class KerasParser:

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
        
    def load(self, model_path):
        model = keras.models.load_model(model_path, compile=False)
        layers = [] 
        depth = 1
        for l in model.layers:
            assert type(l) in self.SUPPORTED_LAYERS, f'node {type(l)} is not supported'
            if hasattr(l,'activation'):
                assert l.activation in self.SUPPORTED_ACTIVATIONS, f'activation {l.activation} is not supported'
            
            if isinstance(l, keras.layers.Flatten):
                continue
            elif isinstance(l, keras.layers.Activation):
                layers[-1].set_activation(act)
            elif isinstance(l, keras.layers.MaxPooling2D):
                layers.append(self.keras_maxpool_to_venus(l,depth))
                depth += 1
            elif isinstance(l, keras.layers.GlobalAveragePooling2D):
                layers.append(self.keras_g_avg_pool_to_venus(l,depth))
                depth += 1
            elif isinstance(l, keras.layers.Dense):
                layers.append(self.keras_dense_to_venus(l,depth))
                depth += 1
            elif isinstance(l,keras.layers.Conv2D):
                layers.append(self.keras_conv_to_venus(l,depth))
                depth += 1

            
        return layers


    def to_keras(self, nn, filepath=None):
        model = keras.models.Sequential()
        for l in nn.layers:
            if isinstance(l, FullyConnected):
                model.add(venus_fc_to_keras(l))
            elif isinstance(l, Conv2D):
                model.add(venus_conv_to_keras(l))
        if not filepath is None:
            model.save(filepath)

        return model

                                              

    def keras_dense_to_venus(self, layer, depth):
        w = layer.get_weights()
        weights = w[0].T
        bias = w[1] if len(w) > 1 else np.zeros(layer.output_shape[1])
        act = self.activations_map[layer.activation]
        return FullyConnected((layer.input_shape[1],),
                              (layer.output_shape[1],),
                              weights,
                              bias,
                              act,
                              depth)
    
    def keras_conv_to_venus(self, layer, depth):
        w = layer.get_weights()
        weights = w[0]
        bias = w[1] if len(w) > 1 else np.zeros(weights.shape[3])
        act = self.activations_map[layer.activation]
        if layer.padding == 'same':
            i_x, i_y = layer.input_shape[1:2]
            o_x, o_y = layer.output_shape[1:2]
            k_x, k_y = layer.kernel_size
            pad_x = int((o_x - i_x + k_x - 1)/2)
            pad_y = int((o_y - i_y + k_y - 1)/2)
        else:
            pad_x=0
            pad_y=0
        return Conv2D(layer.input_shape[1:],
                      layer.output_shape[1:],
                      weights,
                      bias,
                      (pad_x,pad_y),
                      layer.strides,
                      act,
                      depth)

    def keras_max_pool_to_venus(self, layer, depth):
        return MaxPooling(layer.input_shape[1:],
                          layer.output_shape[1:],
                          layer.pool_size,
                          depth)
    
    def keras_g_avg_pool_to_venus(self, layer, depth):
        return GlobalAveragePooling((layer.input_shape[1:]),
                                    (layer.output_shape[1],),
                                    depth)


    def venus_ff_to_keras(self, layer):
        act = activations_r_map[layer.activation]
        dense = keras.layers.Dense(layer.output_size,
                                   activation=act,
                                   input_shape=layer.input_shape)
        dense.set_weights((layer.weights.T,layer.bias))
        
        return dense

    def venus_conv_to_keras(self, layer):
        assert l.padding == (0,0) or l.input_shape[0:2] == l.output_shape[0:2], \
            "Keras supports only 'valid padding', i.e., no padding, or 'same padding', i.e., padding such that the input and output shape are the same."
        act = activations_r_map[layer.activation]
        padding = 'valid' if l.padding==(0,0) else 'same'
        conv = keras.layers.Conv2D(layer.kernels[-1],
                                   layer.kernels[0:2],
                                   strides=layer.strides,
                                   padding = padding,
                                   activation=act,
                                   input_shape=layer.input_shape)
        conv.set_weights((layer.kernels,layer.bias))

        return conv
