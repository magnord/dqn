from __future__ import print_function
from builtins import *
import time
import random
import math
import numpy as np
import tensorflow as tf
from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from checkpointer import Checkpointer


def weight_variable(name, input_size, output_size):
    xavier = -6.0 / math.sqrt(input_size + output_size)
    initializer = tf.random_uniform_initializer(-xavier, xavier)
    return tf.get_variable(name, (input_size, output_size), initializer=initializer)


def bias_variable(name, size):
    return tf.get_variable(name, (size,), initializer=tf.constant_initializer(0))


class DQN(Checkpointer):
    """Deep Neural Netwok
    """

    def __init__(self, sess, game, f):
        self.sess = sess
        self.memory = deque()
        self.epsilon = self.start_epsilon = f.start_epsilon
        self.final_epsilon = f.final_epsilon
        self.observe = 500
        self.discount = 0.99  # Gamma
        self.num_action_per_step = 1
        self.memory_size = f.memory_size
        self.batch_size = f.batch_size
        self.max_steps = f.max_steps
        self.learning_rate = f.learning_rate

        self.game = game
        self.dataset = game.name
        self.f = f
        self.num_actions = len(self.game.actions)
        self.observation_size = self.game.observation_size

        self.attributes = ['learning_rate', 'final_epsilon', 'gamma', 'memory_size', 'batch_size']
        self.checkpoint_dir = f.checkpoint_dir

        if f.visualize:
            game.init_visualization()

    def build_model(self):
        """Model that implements activate function that can take in observation vector or a batch
            and returns scores (of unbounded values) for each action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, num_actions]
        """

        layer_size = 128
        # TODO: add namespace so multiple copies of network can be created
        observation = tf.placeholder(tf.float32, [None, self.observation_size], name='observation')

        # hidden layers
        W_fc1 = weight_variable('W_fc1', self.observation_size, layer_size)
        b_fc1 = bias_variable('b_fc1', layer_size)
        h_fc1 = tf.nn.relu(tf.matmul(observation, W_fc1) + b_fc1)

        W_fc2 = weight_variable('W_fc2', layer_size, self.num_actions)
        b_fc2 = bias_variable('b_fc2', self.num_actions)

        action_scores = tf.matmul(h_fc1, W_fc2) + b_fc2

        tf.histogram_summary("action_scores", action_scores)
        tf.scalar_summary("action_scores_mean", tf.reduce_mean(action_scores))

        return observation, action_scores, h_fc1

    def train(self):
        """Train a Deep Q Network.

        """
        f = self.f
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.actions = tf.placeholder(tf.float32, [self.batch_size, self.num_actions])
        self.true_reward = tf.placeholder(tf.float32, [self.batch_size])

        self.input, self.action_scores, self.h_fc1 = self.build_model()
        # TODO: add target network
        self.action_reward = tf.reduce_sum(tf.mul(self.action_scores, self.actions), reduction_indices=1)
        self.loss = tf.reduce_sum(tf.square(self.true_reward - self.action_reward))
        self.optim = tf.train.AdamOptimizer(f.learning_rate).minimize(self.loss, global_step=self.global_step)
        # self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9).
        # minimize(self.loss, global_step=self.global_step)

        tf.scalar_summary("loss", self.loss)
        tf.scalar_summary("rewards_mean", tf.reduce_mean(self.true_reward))

        action = np.zeros(self.num_actions)
        action[0] = 1

        self.initialize()

        self.start_time = time.time()
        start_iter = self.global_step.eval()
        print('start_iter' + str(start_iter))
        final_step = start_iter + self.max_steps

        self.state_t = self.game.observation()
        print("Start")

        if f.visualize:
            _ = FuncAnimation(self.game.fig, self.animate, interval=30)
            plt.show()
        else:
            # Simulation step without animation
            for step_no in range(start_iter, final_step):
                self.do_step(step_no)

    def animate(self, frame):
        self.do_step(frame)
        self.game.update_visualization()

    def do_step(self, step_no):
        action_scores = self.sess.run([self.action_scores], feed_dict={self.input: np.expand_dims(self.state_t, axis=0)})
        action_t = np.zeros([self.num_actions])

        # Epsilon greedy action selection (no random actions if checkpoint is loaded)
        if not self.f.load_checkpoint and (random.random() <= self.epsilon or step_no <= self.observe):
            action_idx = random.randrange(0, self.num_actions)
        else:
            action_idx = np.argmax(action_scores[0])

        action_t[action_idx] = 1

        # Decrease epsilon (TODO: Improve decreasing formula)
        if self.epsilon > self.final_epsilon and step_no > self.observe:
            self.epsilon -= (self.start_epsilon - self.final_epsilon) / (100 * self.observe)  # ?

        # Execute an action
        state_t1, reward_t, is_terminal = self.game.do(action_idx)
        # print(action_idx, action_scores[0], state_t1, reward_t, is_terminal, len(self.memory), self.epsilon)

        # And save it in the experience replay memory
        self.memory.append((self.state_t, action_t, reward_t, state_t1, is_terminal))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

        # Q-learning updates (mini-batched)
        if len(self.memory) > self.observe:  # Only train when we have enough data in replay memory
            batch = random.sample(self.memory, self.batch_size)

            s = [mem[0] for mem in batch]
            a = [mem[1] for mem in batch]
            r = [mem[2] for mem in batch]
            s2 = [mem[3] for mem in batch]
            terminal = [mem[4] for mem in batch]

            # Predicted reward for next state
            y = []
            predicted_reward = self.action_scores.eval(feed_dict={self.input: s2})
            for i in range(0, self.batch_size):
                if terminal[i]:
                    # If terminal only equals current reward
                    y.append(r[i])
                else:
                    # Otherwise discounted future reward of best action
                    y.append(r[i] + self.discount * np.max(predicted_reward[i]))

            write_summaries = step_no % 100 == 0

            # Run a training step in the network
            _, loss, summaries = self.sess.run([self.optim,
                                                self.loss,
                                                self.merged_sum if write_summaries else self.no_op],
                                               feed_dict={
                                                   self.input: s,
                                                   self.true_reward: y,
                                                   self.actions: a
                                               })

            # Save checkpoint
            if step_no % 10000 == 0:
                self.save(self.f.checkpoint_dir, step_no)

            # Print progress
            if step_no % 100 == 0:
                print("Step: [%2d/%7d] time: %4.2f, loss: %.4f, action score mean: %.4f, e: %.6f" % (
                    step_no, self.max_steps, time.time() - self.start_time, loss, np.mean(action_scores[0]), self.epsilon))

            # Write summaries for Tensorboard
            if write_summaries:
                self.writer.add_summary(summaries, step_no)

        if is_terminal:
            _, _, _ = self.game.new_game()

        self.state_t = state_t1
