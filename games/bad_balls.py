from __future__ import print_function, generators

import numpy as np
import random
import matplotlib.pyplot as plt

"""
Catch bouncing balls, avoid bad balls. Actions are 'up', 'down', 'left and 'right'.
"""

force = 0.05  # Magnitude of force applied by action
num_good_balls = 20
num_bad_balls = 10
num_balls = num_good_balls + num_bad_balls
radius = 0.04
max_x = 1.0
max_y = 1.0
max_ball_speed = 0.005
ray_length = 0.5
num_rays = 16
angles = np.linspace(0.0, 2 * np.pi, num_rays, endpoint=False)
arc_width = 2 * np.pi / num_rays
lx = ray_length * np.cos(angles)
ly = ray_length * np.sin(angles)


def create_balls():
    bx = np.random.uniform(0.0, max_x, num_balls)
    by = np.random.uniform(0.0, max_y, num_balls)
    bxv = np.random.uniform(-max_ball_speed, max_ball_speed, num_balls)
    byv = np.random.uniform(-max_ball_speed, max_ball_speed, num_balls)
    bt = np.zeros(num_balls)
    bt[:num_bad_balls] = 1  # 1 represents a bad ball
    return bx, by, bxv, byv, bt


class BadBallsGame(object):
    def new_game(self):
        self.px = max_x / 2
        self.py = max_y / 2
        self.pxv = 0.0
        self.pyv = 0.0
        self.score = 0
        self.fig = None
        self.bx, self.by, self.bxv, self.byv, self.bt = create_balls()

        return self.observation(), self.get_score(), self.terminal()

    def __init__(self):
        self.name = "bad_balls"
        self.actions = ['noop', 'up', 'down', 'left', 'right']
        self.action_forces = [(0.0, 0.0), (0.0, -force), (0.0, force), (-force, 0.0), (force, 0.0)]
        self.observation_size = 2 * 3 * num_rays + 4

        _, _, _ = self.new_game()

    def init_visualization(self):
        # Setup visualization
        self.fig = plt.figure(figsize=(8, 8))
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, max_x), ax.set_xticks([])
        ax.set_ylim(0, max_y), ax.set_yticks([])

        # Use a scatter plot for visualization
        self.good_balls_vis = ax.scatter([0], [0], c='green', s=500)
        self.bad_balls_vis = ax.scatter([0], [0], c='red', s=500)
        self.player_vis = ax.scatter([0], [0], c='blue', s=500)

    def update_visualization(self):
        self.player_vis.set_offsets([(self.px, self.py)])
        self.good_balls_vis.set_offsets(list(zip(self.bx[num_bad_balls:], self.by[num_bad_balls:])))
        self.bad_balls_vis.set_offsets(list(zip(self.bx[:num_bad_balls], self.by[:num_bad_balls])))

    def get_score(self):
        score = self.score
        self.score = 0
        return score

    def terminal(self):
        return False

    def observation(self):
        # Do raycasting to observe balls
        dist_good = np.empty(num_rays)
        dist_good.fill(max_x * 2.0)  # No observation = 2 * world size
        dist_bad = np.empty(num_rays)
        dist_bad.fill(max_x * 2.0)
        good_ball_xv = np.zeros(num_rays)
        good_ball_yv = np.zeros(num_rays)
        bad_ball_xv = np.zeros(num_rays)
        bad_ball_yv = np.zeros(num_rays)
        delta_x = self.px - self.bx
        delta_y = self.py - self.by
        angle_b = np.arctan2(delta_y, delta_x) + np.pi
        squared_dist_to_balls = np.square(delta_x) + np.square(delta_y)
        relevant_balls = np.where(squared_dist_to_balls < ray_length * ray_length)[0]
        relevant_good_balls = relevant_balls[relevant_balls >= num_bad_balls]
        relevant_bad_balls = relevant_balls[relevant_balls < num_bad_balls]
        # print((relevant_balls, relevant_good_balls, relevant_bad_balls))
        # print((delta_x, delta_y, angle_b * 180 / np.pi))
        # TODO: Calculate distance and angle to each ball, then observe the closest ball in that "ray sector"

        for i in list(relevant_good_balls):
            # Calculate which arc the ball is in
            arc = int(angle_b[i] / arc_width)
            if squared_dist_to_balls[i] < dist_good[arc]:  # Ball distance less than smallest so far
                dist_good[arc] = squared_dist_to_balls[i]  # Distance to closest ball
                good_ball_xv[arc] = self.bxv[i]              # Velocity of closest ball
                good_ball_yv[arc] = self.byv[i]
        for i in list(relevant_bad_balls):
            # Calculate which arc the ball is in
            arc = int(angle_b[i] / arc_width)
            if squared_dist_to_balls[i] < dist_bad[arc]:  # Ball distance less than smallest so far
                dist_bad[arc] = squared_dist_to_balls[i]  # Distance to closest ball
                bad_ball_xv[arc] = self.bxv[i]              # Velocity of closest ball
                bad_ball_yv[arc] = self.byv[i]

        # TODO: Optimization: remove balls already seen (closest seen) from relevant_balls lists
        # print((dist_good, dist_bad))
        return np.hstack(([self.px, self.py, self.pxv, self.pyv],
                          dist_good, good_ball_xv, good_ball_yv,
                          dist_bad, bad_ball_xv, bad_ball_yv))

    def do(self, action_idx):
        # Update balls
        self.bx += self.bxv
        self.by += self.byv

        # Update player according to action
        (xv, yv) = self.action_forces[action_idx]
        self.pxv += xv * force
        self.pyv += yv * force
        self.pxv *= 0.8  # Friction
        self.pyv *= 0.8
        self.px += self.pxv
        self.py += self.pyv

        # Bounce ball if outside playing field
        np.putmask(self.bxv, np.logical_or(self.bx < 0.0, self.bx > max_x), -self.bxv)
        np.putmask(self.byv, np.logical_or(self.by < 0.0, self.by > max_y), -self.byv)

        # Bound player position
        self.px = max(min(self.px, max_x), 0.0)
        self.py = max(min(self.py, max_y), 0.0)

        # Check for player-ball collision
        dist_x = np.square(self.bx - self.px)
        dist_y = np.square(self.by - self.py)
        dist = np.sqrt(dist_x + dist_y)
        collision_idx = np.where(dist < radius)
        for i in collision_idx[0]:
            # Increase score and respawn ball
            self.score += 1 if self.bt[i] == 0 else -5
            # print(self.score)
            # print(self.observation())
            self.bx[i] = random.uniform(0.0, max_x)
            self.by[i] = random.uniform(0.0, max_y)
            self.bxv[i] = random.uniform(-max_ball_speed, max_ball_speed)
            self.byv[i] = random.uniform(-max_ball_speed, max_ball_speed)

        return self.observation()

