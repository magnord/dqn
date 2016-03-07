from __future__ import print_function
from builtins import *
import os
import tensorflow as tf


class Checkpointer(object):
    """Saving and restoring a TensorFlow model."""

    def __init__(self):
        pass

    def get_model_dir(self):
        model_dir = self.dataset
        for attr in self.attributes:
            if hasattr(self, attr):
                model_dir += "_%s:%s" % (attr, getattr(self, attr))
        return model_dir

    def save(self, checkpoint_dir, global_step=None):
        self.saver = tf.train.Saver()

        print("Saving checkpoints...")
        model_name = type(self).__name__ or "Checkpointer"
        model_dir = self.get_model_dir()

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def initialize(self, log_dir="./logs"):
        self.no_op = tf.no_op()  # Used instead of merged_sum when no summaries are needed
        self.merged_sum = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(log_dir)

        tf.initialize_all_variables().run()
        if self.f.load_checkpoint:
            self.load(self.checkpoint_dir)

    def load(self, checkpoint_dir):
        self.saver = tf.train.Saver()

        print("Loading checkpoints...")
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Loading from ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load success")
            return True
        else:
            print("Load failed")
            return False
