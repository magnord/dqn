from __future__ import print_function
from builtins import *
import os
import tensorflow as tf


class Checkpointer(object):
    """Saving and restoring a TensorFlow model."""

    def __init__(self):
        pass

    def get_model_dir(self):
        model_dir = self.game.name
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

    def initialize(self):
        self.no_op = tf.no_op()  # Used instead of merged_sum when no summaries are needed
        if not self.f.no_logging:
            self.merged_summaries = tf.merge_all_summaries()
            final_log_dir = os.path.join(self.log_dir, self.get_model_dir())
            if not os.path.exists(final_log_dir):
                os.makedirs(final_log_dir)
            self.writer = tf.train.SummaryWriter(final_log_dir)

        tf.initialize_all_variables().run()

        if not self.f.no_logging:
            self.writer.add_graph(self.sess.graph_def)

        if self.f.load_checkpoint:
            self.load(self.checkpoint_dir)

    def load(self, checkpoint_dir):
        self.saver = tf.train.Saver()

        print("Loading checkpoints...")
        model_dir = self.get_model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print('Loading from ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Load success")
            return True
        else:
            print("Load failed")
            return False
