import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import numpy as np
import copy
import math

from inits import glorot, zeros
from config import FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


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


def convert_sparse_matrix_to_sparse_tensor(X):
    """Convert a scipy sparse matrix to a torch sparse_coo_tensor (float32)."""
    coo = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class Layer(nn.Module):
    """Base layer class. Defines basic API for all layer objects."""

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


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=F.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout_flag = bool(dropout)
        self.placeholders = placeholders

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias

        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
        self.register_parameter('weights', self.vars['weights'])
        if self.use_bias:
            self.vars['bias'] = zeros([output_dim], name='bias')
            self.register_parameter('bias', self.vars['bias'])

    def forward(self, inputs):
        x = inputs

        dropout_rate = self.placeholders['dropout'] if self.dropout_flag else 0.0
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - dropout_rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=dropout_rate, training=self.training)

        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        if self.use_bias:
            output = output + self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer used by the HGCN model."""

    def __init__(self, input_dim, output_dim, placeholders, support, transfer,
                 mod, layer_index, dropout=0., sparse_inputs=False, act=F.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout_flag = bool(dropout)
        self.placeholders = placeholders

        self.act = act
        self.support = support  # list of (coords, values, shape) tuples, length = channel_num
        self.transfer = transfer
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.use_bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim

        self.num_features_nonzero = placeholders['num_features_nonzero']

        if self.mod == 'coarsen' or self.mod == 'refine':
            for i in range(len(self.support)):
                w = glorot([input_dim + FLAGS.node_wgt_embed_dim, output_dim],
                           name='weights_' + str(i))
                self.vars['weights_' + str(i)] = w
                self.register_parameter('weights_' + str(i), w)
        else:
            for i in range(len(self.support)):
                w = glorot([input_dim, output_dim], name='weights_' + str(i))
                self.vars['weights_' + str(i)] = w
                self.register_parameter('weights_' + str(i), w)
        if self.use_bias:
            self.vars['bias'] = zeros([output_dim], name='bias')
            self.register_parameter('bias', self.vars['bias'])

        # Pre-build sparse support tensors as buffers (non-trainable).
        for i in range(len(self.support)):
            row = self.support[i][0][:, 0]
            col = self.support[i][0][:, 1]
            data = self.support[i][1]
            sp_support = sp.csr_matrix((data, (row, col)), shape=self.support[i][2], dtype=np.float32)
            sp_tensor = convert_sparse_matrix_to_sparse_tensor(sp_support)
            self.register_buffer('support_tensor_' + str(i), sp_tensor)

        # Pre-build the transfer tensor as a buffer.
        if self.mod == 'coarsen' or self.mod == 'input':
            transfer_opo = normalize(self.transfer.T, norm='l2', axis=1).astype(np.float32)
            self.register_buffer('transfer_tensor',
                                 convert_sparse_matrix_to_sparse_tensor(transfer_opo))
        elif self.mod == 'refine':
            self.register_buffer('transfer_tensor',
                                 convert_sparse_matrix_to_sparse_tensor(self.transfer.astype(np.float32)))
        else:
            self.transfer_tensor = None

        # 1x1 conv combining the per-channel supports.
        self.channel_combine = nn.Conv1d(in_channels=len(self.support),
                                         out_channels=1,
                                         kernel_size=1,
                                         bias=False)

    def forward(self, inputs, node_emb=None):
        if self.mod == 'coarsen' or self.mod == 'refine':
            x = torch.cat([inputs, node_emb], dim=1)
            print('layer_index ', self.layer_index + 1)
            print('input shape:   ', list(inputs.shape))
        elif self.mod == 'input' or self.mod == 'output':
            x = inputs

        dropout_rate = self.placeholders['dropout'] if self.dropout_flag else 0.0
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - dropout_rate, self.num_features_nonzero)
        else:
            x = F.dropout(x, p=dropout_rate, training=self.training)

        # convolve
        supports = []
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            sp_tensor = getattr(self, 'support_tensor_' + str(i))
            support_ans = dot(sp_tensor, pre_sup, sparse=True)
            supports.append(support_ans)

        # supports: list of (N, output_dim) tensors -> stack to (N, output_dim, num_channels)
        supports = torch.stack(supports, dim=2)
        # Conv1d expects (batch, channels, length). Permute to (N, num_channels, output_dim).
        supports = supports.permute(0, 2, 1)
        output = self.channel_combine(supports)  # (N, 1, output_dim)
        output = output.squeeze(1)  # (N, output_dim)

        if self.use_bias:
            output = output + self.vars['bias']
        output = self.act(output)

        gcn_output = output

        if self.mod == 'output':
            print('layer_index ', self.layer_index + 1)
            print('input shape:   ', list(inputs.shape))
            print('output shape:    ', list(output.shape))
            return output, gcn_output

        if self.mod == 'coarsen' or self.mod == 'input':
            output = dot(self.transfer_tensor, gcn_output, sparse=True)
        elif self.mod == 'refine':
            output = dot(self.transfer_tensor, gcn_output, sparse=True)

        print('output shape:    ', list(output.shape))
        return output, gcn_output