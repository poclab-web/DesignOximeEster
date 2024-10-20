import torch.nn as nn
from networks.layers.base_layer import BaseLayer
from networks.layers.dense_layer import DenseLayer


class DenseNetwork(BaseLayer):
    def __init__(self, first_node):
        super().__init__()
        self.layers = nn.ModuleDict()

        ### defining model ###
        self.num_of_layers = 13
        use_dropout = False
        use_batch_norm1 = True
        use_batch_norm2 = False
        activation1 = 'LeakyReLU'
        final_activation = None
        self.layers[str(1)] = DenseLayer(first_node, 4096, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(2)] = DenseLayer(4096, 2048, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(3)] = DenseLayer(2048, 1024, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(4)] = DenseLayer(1024, 512, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(5)] = DenseLayer(512, 256, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(6)] = DenseLayer(256, 128, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(7)] = DenseLayer(128, 64, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(8)] = DenseLayer(64, 32, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(9)] = DenseLayer(32, 16, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(10)] = DenseLayer(16, 8, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(11)] = DenseLayer(8, 4, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(12)] = DenseLayer(4, 2, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(13)] = DenseLayer(2, 1, activation=final_activation, use_dropout=use_dropout, use_batchnorm=use_batch_norm2)

    def forward(self, x):
        for i in range(self.num_of_layers):
            x = self.layers[str(i+1)](x)
        return x

    def set_channel(self, pre_ch, mag):
        """
        pre_ch -> 一個前の層のチャンネルの出力
        next_ch -> 次の層のチャンネルの入力
        基本的に1/magしていく
        """
        next_ch = pre_ch / mag
        return next_ch
