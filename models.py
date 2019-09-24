import numpy as np
from utils import *


class CustomCPPN:

    def __init__(self, n_depth=4, n_channels=3, hidden_activation=np.tanh,
                 hidden_size=30, weight_interval=(-1, 1), z_size=20):

        input_size = 1

        # input and output weights
        self.io_weights = {'x_in': new_weight((input_size, hidden_size), weight_interval),
                           'y_in': new_weight((input_size, hidden_size), weight_interval),
                           'r_in': new_weight((input_size, hidden_size), weight_interval),
                           'z_in': new_weight((z_size, hidden_size), weight_interval),
                           'out': new_weight((hidden_size, n_channels), weight_interval)}

        # hidden layer weights
        self.h_weights = []
        for i in range(n_depth):
            self.h_weights.append(new_weight((hidden_size, hidden_size), weight_interval))

        self.n_channels = n_channels
        self.n_depth = n_depth
        self.activation = hidden_activation
        self.fixed_shape = False
        self.hidden_size = hidden_size

        self.x_mat, self.y_mat, self.r_mat = None, None, None

    def fix_shape(self, shape, scale):
        self.fixed_shape = True
        self.shape = shape
        self.scale = scale
        self.build_inputs(shape, scale)

    def build_inputs(self, shape, scale):
        b_size, x_num, y_num = shape[:3]

        xs = scale * (np.arange(x_num) - (x_num - 1) / 2.0) / (x_num - 1) / 0.5
        ys = scale * (np.arange(y_num) - (y_num - 1) / 2.0) / (y_num - 1) / 0.5

        x_mat = np.matmul(np.ones((y_num, 1)), xs.reshape((1, x_num)))
        y_mat = np.matmul(ys.reshape((y_num, 1)), np.ones((1, x_num)))

        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)

        self.x_mat = np.tile(x_mat.flatten(), b_size).reshape([-1, 1])
        self.y_mat = np.tile(y_mat.flatten(), b_size).reshape([-1, 1])
        self.r_mat = np.tile(r_mat.flatten(), b_size).reshape([-1, 1])

    def run_out_point(self, x, y, r, z):
        h = compute_layer(x, self.io_weights['x_in']) \
            + compute_layer(y, self.io_weights['y_in']) \
            + compute_layer(z, self.io_weights['z_in']) \
            + compute_layer(r, self.io_weights['r_in'])

        for w in self.h_weights:
            h = compute_layer(h, w, activation=self.activation)

        output = compute_layer(h, self.io_weights['out'], activation=sigmoid)

        return output

    def forward(self, zs, shape=None, scale=None):
        """ zs is of shape: [b_size, z_size]
        if 'shape' and 'scale' haven't been previously fixed you should provide them """

        if not self.fixed_shape:
            assert shape is not None
            self.shape = shape
            self.scale = scale
            self.build_inputs(shape, scale)

        b_size, x_num, y_num = self.shape[:3]

        zs = np.array(
            [np.matmul(np.ones([x_num * y_num, 1]), np.reshape(zs[i], [1, -1])) * self.scale for i in range(b_size)])

        out = self.run_out_point(self.x_mat, self.y_mat, self.r_mat, zs.reshape((b_size * x_num * y_num, -1)))
        images = out.reshape((b_size, y_num, x_num, self.n_channels))
        return images


class InputLayer:

    def __init__(self, hidden_size=30, weight_interval=(-1, 1), z_size=20, activations=None):

        self.weights = {'x_in': new_weight((1, hidden_size), weight_interval),
                           'y_in': new_weight((1, hidden_size), weight_interval),
                           'r_in': new_weight((1, hidden_size), weight_interval),
                           'z_in': new_weight((z_size, hidden_size), weight_interval)}

        self.fixed_shape = False
        self.hidden_size = hidden_size

        self.activations = [None, None, None, None] if activations is None else activations

        self.x_mat, self.y_mat, self.r_mat = None, None, None

    def fix_shape(self, shape, scale):
        self.fixed_shape = True
        self.shape = shape
        self.scale = scale
        self.build_inputs(shape, scale)

    def build_inputs(self, shape, scale):
        b_size, x_num, y_num = shape[:3]

        xs = scale * (np.arange(x_num) - (x_num - 1) / 2.0) / (x_num - 1) / 0.5
        ys = scale * (np.arange(y_num) - (y_num - 1) / 2.0) / (y_num - 1) / 0.5

        x_mat = np.matmul(np.ones((y_num, 1)), xs.reshape((1, x_num)))
        y_mat = np.matmul(ys.reshape((y_num, 1)), np.ones((1, x_num)))

        r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)

        self.x_mat = np.tile(x_mat.flatten(), b_size).reshape([-1, 1])
        self.y_mat = np.tile(y_mat.flatten(), b_size).reshape([-1, 1])
        self.r_mat = np.tile(r_mat.flatten(), b_size).reshape([-1, 1])

    def forward(self, zs, shape=None, scale=None):
        """ zs is of shape: [b_size, z_size]
        if 'shape' and 'scale' haven't been previously fixed you should provide them """

        if not self.fixed_shape:
            assert shape is not None
            self.shape = shape
            self.scale = scale
            self.build_inputs(shape, scale)

        b_size, x_num, y_num = self.shape[:3]

        zs = np.array(
            [np.matmul(np.ones([x_num * y_num, 1]), np.reshape(zs[i], [1, -1])) * self.scale for i in range(b_size)])

        h = compute_layer(self.x_mat, self.weights['x_in'], activation=self.activations[0]) \
            + compute_layer(self.y_mat, self.weights['y_in'], activation=self.activations[1]) \
            + compute_layer(self.r_mat, self.weights['r_in'], activation=self.activations[2]) \
            + compute_layer(zs.reshape((b_size * x_num * y_num, -1)), self.weights['z_in'],activation=self.activations[3])

        return h


# TODO allow for multiple parallel stacks in hidden and output layers

class OutputLayer:

    def __init__(self, input_size, shape, weight_interval=(-1, 1), activation=sigmoid):
        self.weight = new_weight((input_size, shape[-1]), weight_interval)
        self.shape = shape
        self.activation = activation

    def forward(self, x):
        b_size, x_num, y_num, n_channels = self.shape
        return compute_layer(x, self.weight, activation=self.activation).reshape((b_size, y_num, x_num, n_channels))


class ModularCPPN:

    def __init__(self, output_image_size, h_layer_num=4, n_channels=3,
                 activations=[None], weight_intervals=[None], layer_sizes=[None], z_size=20, fix_shape=True, scale=0.5):

        assert(len(weight_intervals)==h_layer_num+2 or weight_intervals == [None] or len(weight_intervals) == 1)
        assert(len(activations)==h_layer_num+2 or activations == [None] or len(activations) == 1)
        assert(len(layer_sizes)==h_layer_num+1 or layer_sizes == [None] or len(layer_sizes) == 1)   # we don't need an output layer size, it's 1


        if activations == [None]:
            activations = [None, None, None]            # input layers
            activations = activations + [np.tanh for i in range(h_layer_num)]      # hidden layers
            activations.append(sigmoid)     # output layers
        elif len(activations) == 1:
            act = activations[0]
            activations = [act, act, act]
            activations = activations + [act for i in range(h_layer_num)]
            activations.append(sigmoid)         # leaving the default output layer to sigmoid

        if layer_sizes == [None]:
            layer_sizes = [30 for i in range(h_layer_num+1)]
        elif len(layer_sizes) == 1:
            layer_sizes = [layer_sizes[0] for i in range(h_layer_num+1)]

        if weight_intervals == [None]:
            weight_intervals = [(-1, 1) for i in range(h_layer_num+2)]
        elif len(weight_intervals) == 1:
            weight_intervals = [(weight_intervals[0] for i in range(h_layer_num+2))]


        self.input_layer = InputLayer(layer_sizes[0], weight_interval=weight_intervals[0], z_size=z_size, activations=activations[0])
        self.hidden_layers = [
            make_layer(layer_sizes[i], layer_sizes[i+1], weight_interval=weight_intervals[i+1], activation=activations[i+1])
        for i in range(h_layer_num)]
        self.output_layer = OutputLayer(layer_sizes[-1], [1]+output_image_size+[n_channels], weight_interval=weight_intervals[-1], activation=activations[-1])

        # TODO change it so to allow batches without changing the shape
        if fix_shape:
            self.input_layer.fix_shape([1]+output_image_size+[n_channels], scale)

    def forward(self, zs):
        h = self.input_layer.forward(zs)
        for layer in self.hidden_layers:
            h = run_layer(layer, h)
        return self.output_layer.forward(h)

    def insert_layer(self, index, shape, activation):
        self.hidden_layers.insert(index, make_layer(shape[0], shape[1], activation=activation))
