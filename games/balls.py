from __future__ import print_function, generators
from builtins import *
import numpy as np
import random
import matplotlib.pyplot as plt
from euclid import Point2, LineSegment2

"""
Catch bouncing balls. Actions are 'up', 'down', 'left and 'right'.
"""

force = 0.05  # Magnitude of force applied by action
num_balls = 10
radius = 0.04
max_x = 1.0
max_y = 1.0
max_ball_speed = 0.005
ray_length = 0.5


def create_balls():
    bx = np.random.uniform(0.0, max_x, num_balls)
    by = np.random.uniform(0.0, max_y, num_balls)
    bxv = np.random.uniform(-max_ball_speed, max_ball_speed, num_balls)
    byv = np.random.uniform(-max_ball_speed, max_ball_speed, num_balls)
    return bx, by, bxv, byv


class BallsGame(object):
    def new_game(self):
        self.px = max_x / 2
        self.py = max_y / 2
        self.pxv = 0.0
        self.pyv = 0.0
        self.score = 0
        self.fig = None
        self.bx, self.by, self.bxv, self.byv = create_balls()

        return self.observation(), self.get_score(), self.terminal()

    def __init__(self):
        self.name = "balls"
        self.actions = ['up', 'down', 'left', 'right']
        self.action_forces = [(0.0, -force), (0.0, force), (-force, 0.0), (force, 0.0)]
        self.observation_size = 16
        _, _, _ = self.new_game()

    def init_visualization(self):
        # Setup visualization
        self.fig = plt.figure(figsize=(8, 8))
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, max_x), ax.set_xticks([])
        ax.set_ylim(0, max_y), ax.set_yticks([])

        # Use a scatter plot for visualization
        self.balls_vis = ax.scatter([0], [0], c='green', s=500)
        self.player_vis = ax.scatter([0], [0], c='blue', s=500)

    def update_visualization(self):
        self.player_vis.set_offsets([(self.px, self.py)])
        self.balls_vis.set_offsets(list(zip(self.bx, self.by)))

    def get_score(self):
        score = self.score
        self.score = 0
        return score

    def terminal(self):
        return False

    def observation(self):
        # Do raycasting to observe balls
        angles = np.linspace(0.0, 2*np.pi, self.observation_size, endpoint=False)
        lx = ray_length * np.cos(angles)
        ly = ray_length * np.sin(angles)
        obs_ends_x = self.px + lx
        obs_ends_y = self.py + ly
        dist = np.empty(self.observation_size);
        dist.fill(10)  # No observation = distance: 10
        for i in range(self.observation_size):
            for j in range(num_balls):
                d = line_point_dist(self.px,
                                    self.py,
                                    float(obs_ends_x[i]),
                                    float(obs_ends_y[i]),
                                    float(self.bx[j]),
                                    float(self.by[j]))
                dist[i] = min(d, dist[i])
        return dist

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
            self.score += 1
            self.bx[i] = random.uniform(0.0, max_x)
            self.by[i] = random.uniform(0.0, max_y)
            self.bxv[i] = random.uniform(-max_ball_speed, max_ball_speed)
            self.byv[i] = random.uniform(-max_ball_speed, max_ball_speed)

        return self.observation(), self.get_score(), self.terminal()


# Shortest distance from line (x1,y1) - (x2,y2) to point (x, y)
def line_point_dist(x1, y1, x2, y2, x, y):
    line = LineSegment2(Point2(x1, y1), Point2(x2, y2))
    return line.distance(Point2(x, y))
