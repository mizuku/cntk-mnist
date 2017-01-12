# -*- coding: utf-8 -*-
"""
MNISTデータセットを畳み込みニューラルネットで学習します
"""
import os
import numpy as np
from cntk import Trainer
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, \
                    INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.ops import input_variable, constant, element_times, cross_entropy_with_softmax, classification_error
from cntk.device import cpu, set_default_device
from cntk.learner import learning_rate_schedule, UnitType, sgd, adagrad, momentum_sgd, adam_sgd, momentum_as_time_constant_schedule
from cntk.utils import get_train_loss, get_train_eval_criterion

from models import CNN

abs_path = os.path.dirname(os.path.abspath(__file__))

def check_path(path):
    """
    ファイルパスをチェックします
    """
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

def create_reader(path, is_training, input_dim, label_dim):
    """
    データセットを読み込みます
    """
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=input_dim),
        labels=StreamDef(field='labels', shape=label_dim)
    )), randomize=is_training, epoch_size=INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

def convnet_mnist(input_dim, num_output_classes, reader_train, reader_test):
    """
    畳み込みニューラルネットでMNISTデータセットを学習します
    """
    # Create CNN model
    nn = CNN(input_dim, num_output_classes)
    # Input variables denoting the features and label data
    input_var = input_variable(input_dim, np.float32)
    label_var = input_variable(num_output_classes, np.float32)
    train_t = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }
    test_t = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }
    # Connect
    scaled_input = element_times(constant(0.00390625), input_var)
    z = nn.layers(scaled_input)
    # Instantiate the trainer object to drive the model training
    epoch_size = 60000
    minibatch_size = 128
    lr_per_sample = [0.001]*10 + [0.0005]*10 + [0.0001]
    lr_per_minibatch = learning_rate_schedule(lr_per_sample, UnitType.sample, epoch_size)
    # Momentum schedule
    mm_time_constant = [0]*5 + [1024]
    mm_schedule = momentum_as_time_constant_schedule(momentum=mm_time_constant, epoch_size=epoch_size)

    trainer = Trainer(z,
                      cross_entropy_with_softmax(z, label_var),
                      classification_error(z, label_var),
                      adam_sgd(z.parameters, lr=lr_per_minibatch, momentum=mm_schedule))

    # Get minibatches of images to train with and perform model training
    max_epochs = 20
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            mb = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=train_t)
            trainer.train_minibatch(mb)
            sample_count += mb[label_var].num_samples

        # Debug print
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Epoch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            epoch, training_loss, eval_crit))

    # Test data for trained model
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=test_t)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test

if __name__ == "__main__":
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    set_default_device(cpu())

    # input_dim = 784 # 入力データ1件のデータサイズ
    input_dim = (1, 28, 28) # 入力データ1件のデータサイズ(チャンネル数, 高さ, 幅)
    input_data = input_dim[0] * input_dim[1] * input_dim[2]
    num_output_classes = 10 # 分類数

    train_path = os.path.normpath(os.path.join(abs_path, "..", "datasets", "Train-28x28_cntk_text.txt"))
    check_path(train_path)
    reader_train = create_reader(train_path, True, input_data, num_output_classes)

    test_path = os.path.normpath(os.path.join(abs_path, "..", "datasets", "Test-28x28_cntk_text.txt"))
    check_path(test_path)
    reader_test = create_reader(test_path, False, input_data, num_output_classes)

    error = convnet_mnist(input_dim, num_output_classes, reader_train, reader_test)
    print("Error: %f" % error)


