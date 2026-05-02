"""Configuration object replacing tf.app.flags."""


class Config:
    dataset = 'citeseer'
    model = 'hgcn'
    seed1 = 123
    seed2 = 123
    hidden = 32
    node_wgt_embed_dim = 5
    weight_decay = 0.1
    coarsen_level = 4
    max_node_wgt = 50
    channel_num = 4


FLAGS = Config()