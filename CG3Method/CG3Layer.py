import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_LAYER_UIDS = {}


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-scale, scale))


def zeros(shape, name=None):
    """All zeros."""
    return nn.Parameter(torch.zeros(*shape, dtype=torch.float32))


def ones(shape, name=None):
    """All ones."""
    return nn.Parameter(torch.ones(*shape, dtype=torch.float32))


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32).uniform_(-init_range, init_range))


def weight_variable(shape, name):
    initial = torch.empty(*shape, dtype=torch.float32).normal_(0, 0.1)
    # Truncate at 2 sigmas for parity with tf.truncated_normal.
    initial.clamp_(-0.2, 0.2)
    return nn.Parameter(initial)


def bias_variable(shape, name):
    return nn.Parameter(torch.full(shape, 0.01, dtype=torch.float32))


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors. x is a coalesced torch sparse_coo_tensor."""
    if keep_prob >= 1.0:
        return x
    indices = x._indices()
    values = x._values()
    nnz = values.shape[0]
    random_tensor = keep_prob + torch.rand(nnz, device=values.device)
    dropout_mask = torch.floor(random_tensor).bool()
    new_indices = indices[:, dropout_mask]
    new_values = values[dropout_mask] * (1.0 / keep_prob)
    return torch.sparse_coo_tensor(new_indices, new_values, x.shape).coalesce()


def dot(x, y, sparse=False):
    """Wrapper for matmul (sparse vs dense)."""
    if sparse:
        return torch.sparse.mm(x, y)
    return torch.matmul(x, y)


def conv2d(x, W):
    return F.conv2d(x, W, stride=1, padding='same')


def max_pool_2x2(x):
    return F.max_pool2d(x, kernel_size=2, stride=2, padding=0)


class CNNHSI(nn.Module):
    def __init__(self, dropout=0, act=F.softplus, filter1=None, dim=0):
        super(CNNHSI, self).__init__()
        self.act = act
        self.filter1 = filter1
        self.dim = dim
        self.dropout = dropout if dropout != 0 else 0.0
        self.vars = {}
        self.vars['W_conv'] = zeros(self.filter1, 'CNNweight_0')
        self.register_parameter('W_conv', self.vars['W_conv'])
        self.vars['b_conv'] = bias_variable([1], 'CNNbias_0')
        self.register_parameter('b_conv', self.vars['b_conv'])

    def forward(self, inputs):
        # The original TF tensor is in NHWC. We assume the same convention here for parity.
        # x: (N, H, W, C)
        x = inputs
        # Move to NCHW for PyTorch conv.
        x_nchw = x.permute(0, 3, 1, 2)
        # Original code conv on each channel slice [:,:,:,i:i+1] then concat along channel axis.
        outs = []
        for i in range(self.dim):
            channel = x_nchw[:, i:i+1, :, :]
            # Filter1 is the TF [H, W, in_c, out_c]. Convert to PyTorch [out_c, in_c, H, W].
            W = self.vars['W_conv'].permute(3, 2, 0, 1)
            conv = F.conv2d(channel, W, stride=1, padding='same') + self.vars['b_conv']
            outs.append(self.act(conv))
        # Concat along channel axis (NCHW), then permute back to NHWC.
        out_nchw = torch.cat(outs, dim=1)
        return out_nchw.permute(0, 2, 3, 1)


class SoftmaxLayer(nn.Module):
    def __init__(self, input_num, output_num, dropout=0, act=F.softplus, bias=True):
        super(SoftmaxLayer, self).__init__()
        self.use_bias = bias
        self.act = act
        self.output_num = output_num
        self.input_num = input_num
        self.vars = {}
        self.vars['weights'] = glorot(shape=[self.input_num, self.output_num], name='weight_0')
        self.register_parameter('weights', self.vars['weights'])
        self.vars['bias'] = uniform(shape=[self.output_num], name='bias_0')
        self.register_parameter('bias', self.vars['bias'])

    def forward(self, inputs):
        x = inputs
        pre_sup = torch.matmul(x, self.vars['weights'])
        return self.act(pre_sup + self.vars['bias'])


class Layer(nn.Module):
    """Base layer class."""

    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False


class GraphConvolution(Layer):
    """Graph convolution layer used by the main GCN model in CG3Model.

    `support` here is a torch sparse_coo_tensor or dense tensor (already prepared).
    """

    def __init__(self, input_dim, output_dim, support, num_features_nonzero,
                 act=F.softplus, bias=False, sparse_inputs=False, isnorm=False,
                 isSparse=False, dropout=0, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.act = act
        self.support = support  # the DAD matrix (sparse or dense)
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        # `dropout` may be a float (constant) or a getter callable returning current rate.
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero

        for i in range(1):
            w = glorot([input_dim, output_dim], name='weights_' + str(i))
            self.vars['weights_' + str(i)] = w
            self.register_parameter('weights_' + str(i), w)
        if self.use_bias:
            self.vars['bias'] = zeros([output_dim], name='bias')
            self.register_parameter('bias', self.vars['bias'])

    def _get_dropout_rate(self):
        if callable(self.dropout):
            return float(self.dropout())
        return float(self.dropout)

    def forward(self, inputs):
        x = inputs
        dropout_rate = self._get_dropout_rate()

        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - dropout_rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=dropout_rate, training=self.training)

        # convolve
        pre_sup = dot(x, self.vars['weights_0'], sparse=self.sparse_inputs)
        support = dot(self.support, pre_sup, sparse=self.isSparse)

        output = support  # add_n with one element

        if self.use_bias:
            output = output + self.vars['bias']
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)


class MLP(Layer):
    """Dense MLP layer (single matmul + bias)."""

    def __init__(self, input_dim, output_dim, act=F.softplus, bias=False,
                 sparse_inputs=False, isnorm=False, isSparse=False, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.act = act
        self.use_bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs

        for i in range(1):
            w = glorot([input_dim, output_dim], name='weights_' + str(i))
            self.vars['weights_' + str(i)] = w
            self.register_parameter('weights_' + str(i), w)
        if self.use_bias:
            self.vars['bias'] = zeros([output_dim], name='bias')
            self.register_parameter('bias', self.vars['bias'])

    def forward(self, inputs):
        x = inputs
        support = dot(x, self.vars['weights_0'], sparse=self.sparse_inputs)
        output = support
        if self.use_bias:
            output = output + self.vars['bias']
        if self.isnorm:
            output = F.normalize(output, p=2, dim=0)
        return self.act(output)