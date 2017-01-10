# -*- coding: utf-8 -*-
"""
ニューラルネットワークモデル定義
"""
from cntk.models import Sequential, LayerStack
from cntk.layers import default_options, Convolution, MaxPooling, Dense, Dropout
from cntk.ops import relu
from cntk.initializer import he_uniform

class MLP:
    """
    多層パーセプトロンネットワークモデル
    """
    def __init__(self, input_dim, output_dim):
        """
        コンストラクタ
        params:
            input_dim(integer): 入力パラメータ数
            output_dim(integer): 出力数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = Sequential([
            LayerStack(3, lambda i: [
                Dense([256, 128, 64][i], init=he_uniform(), activation=relu)
            ]),
            Dense(self.output_dim, activation=None)
        ])

class CNN:
    """
    畳み込みニューラルネットワークモデル
    """
    def __init__(self, input_dim, output_dim):
        """
        コンストラクタ
        params:
            input_dim(integer): 入力パラメータ数
            output_dim(integer): 出力数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        with default_options(activation=relu, pad=False):
            self.layers = Sequential([
                LayerStack(3, lambda i: [
                    Convolution((3, 3), [32, 48, 64][i], init=he_uniform(), pad=True),
                    MaxPooling((3, 3), strides=(2, 2)),
                ]),
                Dense(96, init=he_uniform()),
                Dropout(0.5),
                Dense(self.output_dim, activation=None)
            ])

