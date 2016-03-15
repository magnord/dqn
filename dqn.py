from __future__ import print_function

import datetime
import math
import os
import random
import time
from builtins import *
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from checkpointer import Checkpointer


class DQN(Checkpointer):
    """Deep Neural Netwok
    """

    def __init__(self, sess, game, f):
        self.sess = sess
        self.memory = deque()
        self.epsilon = self.start_epsilon = f.start_epsilon
        self.final_epsilon = f.final_epsilon
        self.final_epsilon_step = f.final_epsilon_step
        self.discount = 0.99  # Gamma
        self.memory_size = f.memory_size
        self.batch_size = f.batch_size
        self.max_steps = f.max_steps
        self.learning_rate = f.learning_rate
        self.ddqn = f.ddqn

        self.game = game
        self.f = f
        self.num_actions = len(self.game.actions)
        self.observation_size = self.game.observation_size
        self.repeat_action = f.repeat_action
        self.observe = 1000 * f.repeat_action

        self.attributes = ['learning_rate', 'final_epsilon', 'gamma', 'memory_size', 'batch_size', 'repeat_action', 'ddqn']
        self.checkpoint_dir = os.path.join('checkpoints', f.dir)
        self.log_dir = os.path.join('logs', f.dir)

        if f.visualize:
            game.init_visualization()

    def train(self):
        """Train a Deep Q Network.

        """
        f = self.f
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.actions = tf.placeholder(tf.float32, [self.batch_size, self.num_actions], name='actions')
        self.target_actions = tf.placeholder(tf.float32, [self.batch_size, self.num_actions], name='target_actions')
        self.true_reward = tf.placeholder(tf.float32, [self.batch_size])  # True reward (y) in Q-learning

        # Create primary neural network
        self.input, self.action_scores, self.network_last_hidden_layer = build_model(self.observation_size,
                                                                                     self.num_actions, 'network')
        # Create target neural network
        self.target_input, self.target_action_scores, self.target_network_last_hidden_layer = \
            build_model(self.observation_size, self.num_actions, 'target_network')

        self.action_reward = tf.reduce_sum(tf.mul(self.action_scores, self.actions), reduction_indices=1)
        self.loss = tf.reduce_sum(tf.square(self.true_reward - self.action_reward))  # TODO: Clip to (-1,1)?
        tf.scalar_summary("loss", self.loss)

        # Optimizer
        # TODO: Clip gradients to 10 (as in Double DQN article)?
        self.optim = tf.train.AdamOptimizer(f.learning_rate).minimize(self.loss, global_step=self.global_step)
        # self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9).
        # minimize(self.loss, global_step=self.global_step)

        # Update target network
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            network_vars = [v for v in tf.all_variables() if v.name.startswith('network')]
            target_network_vars = [v for v in tf.all_variables() if v.name.startswith('target_network')]
            # print([(x.name, y.name) for (x,y) in zip(network_vars, target_network_vars)])
            for source, target in zip(network_vars, target_network_vars):
                update_op = target.assign(source)
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # Keep track of actual rewards with exponential moving average (ema)
        self.ema_decay = 0.9999
        self.reward_ema = 0.0
        self.tf_reward_ema = tf.placeholder(tf.float32, name='reward_ema')
        tf.scalar_summary("reward ema", self.tf_reward_ema)

        # Init checkpointing and logging
        self.initialize()

        # Init target network
        self.sess.run(self.target_network_update)

        self.start_time = time.time()
        self.start_iter = self.global_step.eval()
        print('start_iter: ' + str(self.start_iter))
        final_step = self.start_iter + self.max_steps

        # Start simulation adn training
        print("Start")
        self.state_t = self.game.observation()
        self.total_reward = 0
        self.step_no = self.start_iter

        if f.visualize:
            _ = FuncAnimation(self.game.fig, self.animate, interval=30)
            plt.show()
        else:
            # Simulation step without animation
            while self.step_no <= final_step:
                self.do_step()

    def animate(self, frame):
        self.do_step()
        self.game.update_visualization()

    def do_step(self):

        # Calculate action scores for action selection
        action_scores = self.sess.run([self.action_scores],
                                      feed_dict={self.input: np.expand_dims(self.state_t, axis=0)})

        # Epsilon greedy action selection (no random actions if checkpoint is loaded)
        if not self.f.load_checkpoint and (random.random() <= self.epsilon or self.step_no < self.observe):
            action_idx = random.randrange(0, self.num_actions)
        else:
            action_idx = np.argmax(action_scores[0])

        # Set the selected action in a one-hot vector
        action_t = np.zeros([self.num_actions])
        action_t[action_idx] = 1

        # Execute an action (1 or more times according to repeat_action setting)
        reward_t = 0.0
        for i in range(self.repeat_action):
            state_t1 = self.game.do(action_idx)
            reward_t += self.game.get_score()
            self.reward_ema -= (1.0 - self.ema_decay) * (self.reward_ema - reward_t)
            # Decrease epsilon
            # TODO: Track epsilon in Tensorboard
            if self.epsilon > self.final_epsilon and self.step_no > self.observe:
                self.epsilon -= (self.start_epsilon - self.final_epsilon) / self.final_epsilon_step
            self.step_no += 1
        is_terminal = self.game.terminal()

        # Print accumulated score during replay
        if reward_t != 0 and self.f.load_checkpoint:
            self.total_reward += reward_t
            print('{!s}: reward: {:+.0f}, total: {: .0f}'.format(
                datetime.timedelta(seconds=int(time.time() - self.start_time)),
                reward_t,
                self.total_reward))

        # print(str(reward_t) + '=' + str(self.reward.eval()), end=' ')
        # print(action_idx, action_scores[0], state_t1, reward_t, is_terminal, len(self.memory), self.epsilon)

        if not self.f.load_checkpoint:  # Don't train if we have loaded a checkpoint

            # And save it in the experience replay memory
            self.memory.append((self.state_t, action_t, reward_t, state_t1, is_terminal))
            if len(self.memory) > self.memory_size:
                self.memory.popleft()

            # Q-learning training (mini-batched)
            if len(self.memory) > self.observe:  # Only train when we have enough data in replay memory
                batch = random.sample(self.memory, self.batch_size)

                s = [mem[0] for mem in batch]
                a = [mem[1] for mem in batch]
                r = [mem[2] for mem in batch]
                s_t1 = [mem[3] for mem in batch]
                terminal = [mem[4] for mem in batch]

                # Predicted reward for next state calculated using the target network
                # TODO: This could be moved into the TF graph. Is it worth it?
                target_reward = self.target_action_scores.eval(feed_dict={self.target_input: s_t1})

                # If Double Q-Learning
                # Select action with primary network, but calculate action value with target network
                if self.ddqn:
                    predicted_reward = self.action_scores.eval(feed_dict={self.input: s_t1})

                # TODO: Try out dynamic discount
                ys = []
                for i in range(0, self.batch_size):
                    if terminal[i]:  # If terminal only equals current reward
                        y = r[i]
                    else:  # Otherwise discounted future reward of best action
                        if self.ddqn:
                            # Double DQN action value
                            ddqn_action_idx = np.argmax(predicted_reward[i])
                            y = r[i] + self.discount * target_reward[i][ddqn_action_idx]
                        else:
                            # Ordinary DQN reward
                            y = r[i] + self.discount * np.max(target_reward[i])
                    ys.append(y)

                write_summaries = self.step_no % 100 == 0 and not self.f.load_checkpoint and self.step_no >= self.observe

                # Run a training step in the network
                _, loss, summaries = self.sess.run([self.optim,
                                                    self.loss,
                                                    self.merged_summaries if write_summaries else self.no_op],
                                                   feed_dict={
                                                       self.input: s,
                                                       self.target_input: s,  # Just for summaries to work
                                                       self.true_reward: ys,   # Q from target network!
                                                       self.actions: a,
                                                       self.tf_reward_ema: self.reward_ema  # Reward moving average
                                                   })

                # Save checkpoint
                if self.step_no % 50000 == 0:
                    self.save(self.checkpoint_dir, self.step_no)

                # Update target network
                if self.step_no % self.f.update_target_network == 0:
                    self.sess.run(self.target_network_update)

                # Print progress
                if self.step_no % 100 == 0:
                    n = self.step_no if not self.f.visualize else self.step_no + self.start_iter
                    print("Step: [%2d/%7d] time: %4.2f, loss: %.4f, ac.sc: %.4f, e: %.6f, reward ema: %.6f" % (
                        n,
                        self.max_steps,
                        time.time() - self.start_time,
                        loss,
                        np.mean(action_scores[0]),
                        self.epsilon,
                        self.reward_ema))

                # Write summaries for Tensorboard
                if write_summaries:
                    self.writer.add_summary(summaries, self.step_no)

        if is_terminal:
            _, _, _ = self.game.new_game()

        self.state_t = state_t1


def build_model(observation_size, num_actions, namespace):
    """Model that implements actiion function that can take in observation vector or a batch
        and returns scores (of unbounded values) for each action for each observation.
        input shape:  [batch_size, observation_size]
        output shape: [batch_size, num_actions]
        :param namespace: The name space of this network
    """

    layer_size = 128
    # TODO: move network definition to its own module
    # TODO: Test layer size
    # TODO: Do we need two fully connected layers for simple tasks? Gow much does it give?
    # TODO: Test other non-linearities, e.g. tanh.
    # TODO: Implement dueling network architecture
    with tf.variable_scope(namespace):

        observation = tf.placeholder(tf.float32, [None, observation_size], name='observation')

        # Input layer
        w_fc0 = weight_variable('W_fc0', observation_size, layer_size)
        b_fc0 = bias_variable('b_fc0', layer_size)
        h_fc0 = tf.nn.relu(tf.matmul(observation, w_fc0) + b_fc0)

        # Hidden layer 1
        w_fc1 = weight_variable('W_fc1', layer_size, layer_size)
        b_fc1 = bias_variable('b_fc1', layer_size)
        h_fc1 = tf.nn.relu(tf.matmul(h_fc0, w_fc1) + b_fc1)

        # Hidden layer 2
        w_fc2 = weight_variable('W_fc2', layer_size, num_actions)
        b_fc2 = bias_variable('b_fc2', num_actions)

        # Output
        action_scores = tf.matmul(h_fc1, w_fc2) + b_fc2

        # tf.histogram_summary(action_scores.name, action_scores)
        tf.scalar_summary(namespace + "/action_score_mean", tf.reduce_mean(action_scores))

        return observation, action_scores, w_fc2


def weight_variable(name, input_size, output_size):
    xavier = -6.0 / math.sqrt(input_size + output_size)
    initializer = tf.random_uniform_initializer(-xavier, xavier)
    return tf.get_variable(name, (input_size, output_size), initializer=initializer)


def bias_variable(name, size):
    return tf.get_variable(name, (size,), initializer=tf.constant_initializer(0))

