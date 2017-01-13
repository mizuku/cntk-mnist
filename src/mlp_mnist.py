# -*- coding: utf-8 -*-
"""
MNISTデータセットを多層パーセプトロンで学習します
"""
import os
import numpy as np
from cntk import Trainer
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, \
                    INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.ops import input_variable, constant, element_times, cross_entropy_with_softmax, classification_error
from cntk.learner import learning_rate_schedule, UnitType, sgd
from cntk.utils import get_train_loss, get_train_eval_criterion

from models import MLP

abs_path = os.path.dirname(os.path.abspath(__file__))

def create_reader(path, is_training, input_dim, label_dim):
    """
    データセットを読み込みます
    """
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels=StreamDef(field='labels', shape=label_dim, is_sparse=False)
    )), randomize=is_training, epoch_size=INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

def mlp_mnist(input_dim, num_output_classes, reader_train, reader_test):
    """
    多層パーセプトロンでMNISTデータセットを学習します
    """
    # 入力パラメータ数と出力ラベル数をCNTKの型でラップする
    input_var = input_variable(input_dim, np.float32) # input_dim は入力パラメータ数
    label_var = input_variable(num_output_classes, np.float32) # num_output_classes は出力ラベル数
    # 入力パラメータを作成
    scaled_input = element_times(constant(0.00390625), input_var)
    # MLPで実装したモデルを入力パラメータと接続
    nn = MLP(input_dim, num_output_classes)
    z = nn.layers(scaled_input)
    
    # Trainerを生成 (層の出力, 損失関数, 評価関数, 学習関数)
    lr_per_minibatch = learning_rate_schedule(1e-1, UnitType.minibatch)
    trainer = Trainer(z,
                      cross_entropy_with_softmax(z, label_var),
                      classification_error(z, label_var),
                      sgd(z.parameters, lr=lr_per_minibatch))

    # 教師データを入力として 20週 学習する
    # 教師データとラベルを対応付けるハッシュ配列。 reader_train は create_reader で読み込んだ教師データ
    train_t = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }
    epoch_size = 60000 # epoch はデータの周回単位 epoch_size は１週の大きさ
    minibatch_size = 128 # 一度に処理するデータ数
    max_epochs = 20 # 学習期間
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            mb_data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=train_t)
            trainer.train_minibatch(mb_data)
            sample_count += mb_data[label_var].num_samples

        # Debug print
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Epoch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            epoch, training_loss, eval_crit))

    # 学習が終わったモデルを使用してテストデータを対象に推論を行ない、誤回答の確率を求める
    # テストデータとラベルを対応付けるハッシュ配列。 test_train は create_reader で読み込んだテストデータ
    test_t = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb_data = reader_test.next_minibatch(test_minibatch_size, input_map=test_t)
        eval_error = trainer.test_minibatch(mb_data)
        test_result = test_result + eval_error

    return test_result / num_minibatches_to_test

if __name__ == "__main__":
    input_dim = 784 # 入力データ1件のデータサイズ
    num_output_classes = 10 # 出力数（分類するクラス数 0～9）

    # 教師付きデータの読み込み
    train_path = os.path.normpath(os.path.join(abs_path, "..", "datasets", "Train-28x28_cntk_text.txt"))
    reader_train = create_reader(train_path, True, input_dim, num_output_classes)

    # テストデータの読み込み
    test_path = os.path.normpath(os.path.join(abs_path, "..", "datasets", "Test-28x28_cntk_text.txt"))
    reader_test = create_reader(test_path, False, input_dim, num_output_classes)

    error = mlp_mnist(input_dim, num_output_classes, reader_train, reader_test)
    print("Error: %f" % error)


