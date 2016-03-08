from __future__ import print_function
import pprint as pp

import tensorflow as tf

from dqn import DQN
from games.bad_balls import BadBallsGame as Game

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 10000000, "Max steps to train [10000000]")
flags.DEFINE_float("start_epsilon", 0.20, "Initial probability for exploration [0.2]")
flags.DEFINE_float("final_epsilon", 0.01, "Final probability for exploration [0.01]")
flags.DEFINE_float("learning_rate", 0.0003, "Learning rate [0.0003]")
flags.DEFINE_integer("batch_size", 256, "The size of mini batches [256]")
flags.DEFINE_integer("memory_size", 100000, "The size of the experience replay memory [100000]")
flags.DEFINE_integer("repeat_action", 1, "The size of the experience replay memory [1]")
flags.DEFINE_integer("update_target_network", 5000, "Copy primary network to target network every X steps [5000]")
flags.DEFINE_string("game_name", "game", "The name of game [game]")
flags.DEFINE_string("dir", "default", "Sub-directory to save checkpoints and logs for this training run [default]")
flags.DEFINE_boolean("no_logging", False, "Do not log any summaries [False]")
flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
flags.DEFINE_boolean("visualize", False, "Visualize the game [False]")
flags.DEFINE_boolean("load_checkpoint", False, "Load previous checkpoint [False]")


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    game = Game()

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            dqn = DQN(sess, game, flags.FLAGS)
            dqn.train()


if __name__ == '__main__':
    tf.app.run()
