import numpy as np
import models


def new_weight(shape, w_interval):
    return np.random.rand(*shape) * (w_interval[1] - w_interval[0]) + w_interval[0]


def compute_layer(x, W, b=0, activation=None):
    o = np.matmul(x, W) + b
    return activation(o) if activation is not None else o


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def build_basic_repeating_network(output_image_size, hidden_size=30, repeat_num=2):
    model = models.ModularCPPN(output_image_size, layer_sizes=[hidden_size], h_layer_num=repeat_num)
    repeating_layer = make_layer(model.hidden_layers[0]['weight'].shape[0], model.hidden_layers[0]['weight'].shape[1])
    model.hidden_layers = [repeating_layer for i in range(repeat_num)]
    return model


# TODO implement bias possibility
def make_layer(input_size, output_size, weight_interval=(-1, 1), activation=np.tanh):
    return {'weight': new_weight((input_size, output_size), weight_interval), 'bias': 0, 'activation':activation}


def run_layer(layer, x):
    return compute_layer(x, layer['weight'], layer['bias'], activation=layer['activation'])