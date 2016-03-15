from __future__ import print_function, generators
from builtins import *
import numpy as np
import random
import matplotlib.pyplot as plt
from euclid import Point2, LineSegment2

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
angles = np.linspace(0.0, 2*np.pi, num_rays, endpoint=False)
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
        self.observation_size = 4 * num_rays + 4

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
        obs_ends_x = self.px + lx
        obs_ends_y = self.py + ly
        dist = np.empty(num_rays)
        dist.fill(2.0)  # No observation = distance: 2
        ball_type = np.zeros(num_rays)
        ball_xv = np.zeros(num_rays)
        ball_yv = np.zeros(num_rays)
        squared_dist_to_balls = np.square(self.bx - self.px) + np.square(self.by - self.py)
        relevant_balls = np.where(squared_dist_to_balls < ray_length * ray_length)[0]
        # relevant_good_balls = relevant_balls[relevant_balls >= num_bad_balls]
        # relevant_bad_balls = relevant_balls[relevant_balls < num_bad_balls]
        # print((relevant_balls, relevant_good_balls, relevant_bad_balls))
        # TODO: Use two 'layers' of rays, one for good balls and one for bad, and skip ball type
        # TODO: Calculate distance and angle to each ball, then observe the closest ball in that "ray sector"
        for i in range(num_rays):
            for j in list(relevant_balls):
                # Calculate distance from ball to ray
                d = line_point_dist(self.px,
                                    self.py,
                                    float(obs_ends_x[i]),
                                    float(obs_ends_y[i]),
                                    float(self.bx[j]),
                                    float(self.by[j]))
                if d < radius and d < dist[i]:  # Ball distance to ray less than ball radius and smallest seen so far
                    dist[i] = d                 # Distance to closest ball
                    ball_type[i] = self.bt[j]   # Type of the closest ball
                    ball_xv[i] = self.bxv[j]    # Velocity of closest ball
                    ball_yv[i] = self.byv[j]

        return np.hstack(([self.px, self.py, self.pxv, self.pyv], dist, ball_type, ball_xv, ball_yv))

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


# Shortest distance from line (x1,y1) - (x2,y2) to point (x, y)
def line_point_dist(x1, y1, x2, y2, x, y):
    line = LineSegment2(Point2(x1, y1), Point2(x2, y2))
    return line.distance(Point2(x, y))
