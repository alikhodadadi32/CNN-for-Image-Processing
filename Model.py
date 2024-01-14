import os
import time
from typing import List

import numpy as np
import tensorflow as tf

from Network import MyNetwork
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, sess: tf.Session, configs: object):
        self.sess = sess
        self.configs = configs

    def loss(self, _alpha: float, _beta: float, _lambda: float) -> tf.Tensor:
        L1 = (
            _alpha
            * _lambda
            * tf.add_n(
                [
                    tf.abs(tf.reduce_sum(v))
                    for v in tf.trainable_variables()
                    if "kernel" in v.name
                ]
            )
        )
        L2 = (
            (1 - _alpha)
            * _beta
            * tf.add_n(
                [
                    tf.reduce_sum(tf.square(v))
                    for v in tf.trainable_variables()
                    if "kernel" in v.name
                ]
            )
        )

        CrossEntropyLoss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.labels, self.configs.num_classes),
            logits=self.unscaled_logits,
            name=None,
        )

        CrossEntropyLoss = tf.reduce_mean(CrossEntropyLoss, 0)
        regularizations = tf.add(L1, L2)
        return tf.add(CrossEntropyLoss, regularizations)

    def setup(self, training: bool):
        print("---Setup input interfaces...")
        self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.labels = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)

        print("---Setup the network...")
        Network = MyNetwork(
            self.configs.First_layer_num_filter,
            self.configs.Kernel_size,
            self.configs.Resnet_version,
            self.configs.Num_residual_blocks,
            self.configs.List_residual_layers,
        )

        if training:
            print("---Setup training components...")
            self.logits, self.unscaled_logits = Network(self.inputs, True)
            self.losses = self.loss(
                self.configs._alpha,
                self.configs._beta,
                self.configs._lambda,
            )

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=0.9
            )
            self.train_op = optimizer.minimize(self.losses)

            print("---Setup the Saver for saving models...")
            self.global_variables = tf.global_variables()
            self.saver = tf.train.Saver(max_to_keep=0, name="my-model")
        else:
            print("---Setup testing components...")
            self.logits, _ = Network(self.inputs, False)
            self.preds = tf.argmax(self.logits, axis=-1)

            print("---Setup the Saver for loading models...")
            self.loader = tf.train.Saver(max_to_keep=0, name="my-model")

    def train(self, x_train: np.ndarray, y_train: np.ndarray, max_epoch: int):
        print("###Train###")
        self.setup(True)
        self.sess.run(tf.global_variables_initializer())

        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = int(num_samples / self.configs.batch_size)
        print("---Run...")
        n = 0
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            n += 1
            learning_rate = self.configs.start_learning_rate
            splited_x = np.split(
                curr_x_train,
                np.arange(
                    self.configs.batch_size,
                    num_batches * self.configs.batch_size,
                    self.configs.batch_size,
                ),
            )
            splited_y = np.split(
                curr_y_train,
                np.arange(
                    self.configs.batch_size,
                    num_batches * self.configs.batch_size,
                    self.configs.batch_size,
                ),
            )

            for i in range(num_batches):
                x_batch = splited_x[i]
                y_batch = splited_y[i]
                x_list = []
                for j in range(len(x_batch)):
                    x_list.append(parse_record(x_batch[j], True))
                x_batch_new = np.stack(x_list)
                # Run
                feed_dict = {
                    self.inputs: x_batch_new,
                    self.labels: y_batch,
                    self.learning_rate: learning_rate,
                }

                loss, _ = self.sess.run(
                    [self.losses, self.train_op], feed_dict=feed_dict
                )

                print(
                    "Batch {:d}/{:d} Loss {:.6f} \n".format(i, num_batches, loss[0][0]),
                    end="\r",
                    flush=True,
                )

            duration = time.time() - start_time
            print(
                "Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.".format(
                    epoch, loss[0][0], duration
                )
            )

            if epoch % self.configs.save_interval == 0:
                self.save(self.saver, epoch)

    def test_or_validate(
        self, x: np.ndarray, y: np.ndarray, checkpoint_num_list: List[int]
    ) -> None:
        print("###Test or Validation###")

        self.setup(False)
        self.sess.run(tf.global_variables_initializer())

        # load checkpoint
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = (
                self.configs.mymodel
                + "/"
                + self.configs.modeldir
                + "/model.ckptstep-"
                + str(checkpoint_num)
            )
            self.load(self.loader, checkpointfile)
            preds = []
            for i in range(x.shape[0]):
                x_new = np.zeros(shape=(1, 32, 32, 3))
                x_new[0, :, :, :] = parse_record(x[i], False)
                feed_dict = {self.inputs: x_new, self.labels: y[i]}
                pred = self.sess.run(self.preds, feed_dict=feed_dict)
                preds.append(pred)
            preds = np.array(preds).reshape(np.shape(y))
            print("Test accuracy: {:.4f}".format(np.sum(preds == y) / np.shape(y)[0]))

    def save(self, saver: tf.train.Saver, step: int):
        """Save weights."""
        model_name = "model.ckpt" + "step"
        checkpoint_path = os.path.join(
            self.configs.mymodel, self.configs.modeldir, model_name
        )
        if not os.path.exists(self.configs.mymodel):
            os.makedirs(self.configs.mymodel + "/" + self.configs.modeldir)
            if not os.path.exists(self.configs.modeldir):
                os.makedirs(self.configs.modeldir)
        saver.save(self.sess, checkpoint_path, global_step=step)
        print("The checkpoint has been created.")

    def load(self, loader: tf.train.Saver, filename: str):
        """Load trained weights."""
        loader.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))
