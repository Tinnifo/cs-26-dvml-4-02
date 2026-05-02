import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution
from metrics import masked_softmax_cross_entropy, masked_accuracy
from config import FLAGS


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = nn.ModuleList()
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.embed = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()"""
        self._build()

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


class HGCN(Model):
    def __init__(self, placeholders, input_dim, transfer_list, adj_list, node_wgt_list, **kwargs):
        super(HGCN, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].shape[1]

        self.transfer_list = transfer_list
        self.adj_list = adj_list
        self.node_wgt_list = node_wgt_list

        # W_node_wgt: lookup table of shape [max_node_wgt, node_wgt_embed_dim]
        bound = math.sqrt(6 / (3 * FLAGS.node_wgt_embed_dim + 3 * self.input_dim))
        self.W_node_wgt = nn.Parameter(
            torch.empty(FLAGS.max_node_wgt, FLAGS.node_wgt_embed_dim).uniform_(-bound, bound)
        )

        # Register node weight indices as buffers (long tensors for indexing).
        for i, nw in enumerate(self.node_wgt_list):
            self.register_buffer(
                'node_wgt_idx_' + str(i),
                torch.from_numpy(nw.astype('int64'))
            )

        self.build()

        # The TF version uses placeholders['features'] as the input to the model. We
        # will be passed the features explicitly via forward(); record it on the model
        # for parity with the original API.
        self.inputs = placeholders['features']

        # Outputs/embed are set when forward() runs.
        self.outputs = None
        self.embed = None
        self.loss = torch.tensor(0.0)

    def _build(self):
        FCN_hidden_list = [FLAGS.hidden] * 100

        # Input layer (mod='input')
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FCN_hidden_list[0],
                                            placeholders=self.placeholders,
                                            support=self.adj_list[0] * FLAGS.channel_num,
                                            transfer=self.transfer_list[0],
                                            mod='input',
                                            layer_index=0,
                                            act=F.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # Coarsen layers
        for i in range(FLAGS.coarsen_level - 1):
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i],
                                                output_dim=FCN_hidden_list[i + 1],
                                                placeholders=self.placeholders,
                                                support=self.adj_list[i + 1] * FLAGS.channel_num,
                                                transfer=self.transfer_list[i + 1],
                                                mod='coarsen',
                                                layer_index=i + 1,
                                                act=F.relu,
                                                dropout=True,
                                                logging=self.logging))

        # Refine layers
        for i in range(FLAGS.coarsen_level, FLAGS.coarsen_level * 2):
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i - 1],
                                                output_dim=FCN_hidden_list[i],
                                                placeholders=self.placeholders,
                                                support=self.adj_list[2 * FLAGS.coarsen_level - i] * FLAGS.channel_num,
                                                transfer=self.transfer_list[2 * FLAGS.coarsen_level - 1 - i],
                                                mod='refine',
                                                layer_index=i,
                                                act=F.relu,
                                                dropout=True,
                                                logging=self.logging))

        # Output layer (mod='output')
        self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[FLAGS.coarsen_level * 2 - 1],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.adj_list[0] * FLAGS.channel_num,
                                            transfer=self.transfer_list[0],
                                            mod='output',
                                            layer_index=FLAGS.coarsen_level * 2,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def _node_emb_for_layer(self, layer_idx):
        """Return embedding lookup for the layer (matching the TF code's index choice)."""
        # input layer (idx=0): node_wgt_list[0]
        # coarsen layers (idx=1..coarsen_level-1): node_wgt_list[idx]
        # refine layers (idx=coarsen_level..coarsen_level*2-1): node_wgt_list[2*coarsen_level - idx]
        # output layer (idx=coarsen_level*2): not used (input/output mods don't take node_emb)
        if layer_idx == 0:
            idx_buf = self.node_wgt_idx_0
        elif layer_idx < FLAGS.coarsen_level:
            idx_buf = getattr(self, 'node_wgt_idx_' + str(layer_idx))
        elif layer_idx < FLAGS.coarsen_level * 2:
            idx_buf = getattr(self, 'node_wgt_idx_' + str(2 * FLAGS.coarsen_level - layer_idx))
        else:
            return None
        return self.W_node_wgt[idx_buf]

    def forward(self, features):
        activations = [features]
        gcn_layers = []

        for i, layer in enumerate(self.layers):
            mod = layer.mod
            if mod == 'coarsen' or mod == 'refine':
                node_emb = self._node_emb_for_layer(i)
                hidden, pre_GCN = layer(activations[-1], node_emb=node_emb)
            else:
                hidden, pre_GCN = layer(activations[-1])
            gcn_layers.append(pre_GCN)
            if i >= FLAGS.coarsen_level and i < FLAGS.coarsen_level * 2:
                hidden = hidden + gcn_layers[FLAGS.coarsen_level * 2 - i - 1]
            activations.append(hidden)

        self.outputs = activations[-1]
        self.embed = activations[-2]
        self._loss()
        return self.outputs

    def _loss(self):
        # Weight decay loss only on layer parameters (matches the TF version).
        loss = torch.tensor(0.0, device=self.W_node_wgt.device)
        for layer in self.layers:
            for var in layer.vars.values():
                loss = loss + FLAGS.weight_decay * 0.5 * torch.sum(var ** 2)
        self.loss = loss

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs,
                                        self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def predict(self):
        return F.softmax(self.outputs, dim=1)